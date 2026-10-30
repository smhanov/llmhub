package openaichat

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/smhanov/llmhub"
	"github.com/smhanov/llmhub/auth"
	"github.com/smhanov/llmhub/internal/telemetry"
)

type rotatingTokenSource struct {
	mu          sync.Mutex
	current     string
	invalidated bool
}

func (s *rotatingTokenSource) Token(ctx context.Context) (*auth.Token, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return &auth.Token{AccessToken: s.current}, nil
}

func (s *rotatingTokenSource) Invalidate(accessToken string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.invalidated = true
	if s.current == accessToken {
		s.current = "fresh-token"
	}
}

// TestGenerateRecordsTokenRefreshAttempts covers the 401 invalidation path:
// both httpretry.Do calls share the context session, so their attempts land in
// order without duplicates.
func TestGenerateRecordsTokenRefreshAttempts(t *testing.T) {
	var attempts atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		attempts.Add(1)
		if r.Header.Get("Authorization") == "Bearer expired-token" {
			w.WriteHeader(http.StatusUnauthorized)
			io.WriteString(w, `{"error":{"message":"invalid token"}}`)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		io.WriteString(w, `{
			"id":"chatcmpl-refresh",
			"choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],
			"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}
		}`)
	}))
	defer server.Close()

	source := &rotatingTokenSource{current: "expired-token"}
	client := NewClient(ClientConfig{
		ProviderName: "test-provider",
		BaseConfig: llmhub.NewConfig(
			llmhub.WithBaseURL(server.URL),
			llmhub.WithHTTPClient(server.Client()),
			llmhub.WithTokenSource(source),
			llmhub.WithModel("test-model"),
		),
	})

	ctx, session := telemetry.WithSession(context.Background())
	resp, err := client.Generate(ctx, []*llmhub.Message{llmhub.NewUserMessage(llmhub.Text("hi"))})
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if got := resp.Text(); got != "ok" {
		t.Fatalf("Text = %q, want ok", got)
	}
	if got := attempts.Load(); got != 2 {
		t.Fatalf("server attempts = %d, want 2", got)
	}

	source.mu.Lock()
	invalidated := source.invalidated
	source.mu.Unlock()
	if !invalidated {
		t.Error("token source was not invalidated after the 401")
	}

	recorded := session.Attempts()
	if len(recorded) != 2 {
		t.Fatalf("recorded attempts = %+v, want 2 entries", recorded)
	}
	if got := recorded[0].StatusCode; got != http.StatusUnauthorized {
		t.Errorf("first status = %d, want 401", got)
	}
	if got := recorded[1].StatusCode; got != http.StatusOK {
		t.Errorf("second status = %d, want 200", got)
	}
	for i, attempt := range recorded {
		if attempt.Err != "" {
			t.Errorf("attempt %d Err = %q, want empty", i, attempt.Err)
		}
		if attempt.Backoff != 0 {
			t.Errorf("attempt %d Backoff = %v, want 0 (token refresh is not backoff)", i, attempt.Backoff)
		}
	}
}

package gemini

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/smhanov/llmhub"
)

func TestGeminiGenerate(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if got := r.URL.Query().Get("key"); got != "secret" {
			t.Fatalf("missing api key query param, got %q", got)
		}
		defer r.Body.Close()
		body, _ := io.ReadAll(r.Body)
		var req geminiRequest
		if err := json.Unmarshal(body, &req); err != nil {
			t.Fatalf("decode request: %v", err)
		}
		if len(req.Contents) == 0 {
			t.Fatalf("expected contents")
		}
		io.WriteString(w, `{"candidates":[{"content":{"parts":[{"text":"pong"}]}}],"usage_metadata":{"prompt_token_count":1,"candidates_token_count":2,"total_token_count":3}}`)
	}))
	defer server.Close()

	provider, err := New("secret", llmhub.WithBaseURL(server.URL), llmhub.WithModel("flash-test"))
	if err != nil {
		t.Fatalf("provider new: %v", err)
	}
	resp, err := provider.Generate(context.Background(), []*llmhub.Message{llmhub.NewUserMessage(llmhub.Text("ping"))})
	if err != nil {
		t.Fatalf("generate: %v", err)
	}
	if resp.Text() != "pong" {
		t.Fatalf("unexpected response: %s", resp.Text())
	}
}

func TestGeminiStream(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprintln(w, `{"candidates":[{"content":{"parts":[{"text":"foo"}]}}]}`)
		fmt.Fprintln(w, `{"candidates":[{"content":{"parts":[{"text":"bar"}]}}]}`)
	}))
	defer server.Close()

	provider, err := New("secret", llmhub.WithBaseURL(server.URL))
	if err != nil {
		t.Fatalf("new provider: %v", err)
	}
	stream, err := provider.Stream(context.Background(), []*llmhub.Message{llmhub.NewUserMessage(llmhub.Text("hi"))})
	if err != nil {
		t.Fatalf("stream: %v", err)
	}
	first := <-stream
	second := <-stream
	if first.Delta != "foo" || second.Delta != "bar" {
		t.Fatalf("unexpected deltas: %v %v", first, second)
	}
	done := <-stream
	if !done.Done {
		t.Fatalf("expected done chunk")
	}
}

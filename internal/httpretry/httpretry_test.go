package httpretry

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"

	"github.com/smhanov/llmhub"
	"github.com/smhanov/llmhub/internal/telemetry"
)

func TestDoRetries429ByDefault(t *testing.T) {
	var attempts atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		n := attempts.Add(1)
		if n == 1 {
			w.WriteHeader(http.StatusTooManyRequests)
			io.WriteString(w, "rate limited")
			return
		}
		w.WriteHeader(http.StatusOK)
		io.WriteString(w, "ok")
	}))
	defer server.Close()

	cfg := DefaultConfig()
	cfg.BaseDelay = time.Millisecond
	cfg.MaxDelay = time.Millisecond

	resp, err := Do(context.Background(), server.Client(), func() (*http.Request, error) {
		return http.NewRequestWithContext(context.Background(), http.MethodGet, server.URL, nil)
	}, cfg)
	if err != nil {
		t.Fatalf("Do: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200", resp.StatusCode)
	}
	if attempts.Load() != 2 {
		t.Fatalf("attempts = %d, want 2", attempts.Load())
	}
}

func TestDoDoesNotRetry429WhenDisabled(t *testing.T) {
	var attempts atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		attempts.Add(1)
		w.WriteHeader(http.StatusTooManyRequests)
		io.WriteString(w, "rate limited")
	}))
	defer server.Close()

	cfg := DefaultConfig()
	cfg.RetryOnStatus = map[int]bool{http.StatusTooManyRequests: false}

	resp, err := Do(context.Background(), server.Client(), func() (*http.Request, error) {
		return http.NewRequestWithContext(context.Background(), http.MethodGet, server.URL, nil)
	}, cfg)
	if err != nil {
		t.Fatalf("Do: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusTooManyRequests {
		t.Fatalf("status = %d, want 429", resp.StatusCode)
	}
	if attempts.Load() != 1 {
		t.Fatalf("attempts = %d, want 1", attempts.Load())
	}
}

func TestDoRetries500WhenEnabled(t *testing.T) {
	var attempts atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		n := attempts.Add(1)
		if n == 1 {
			w.WriteHeader(http.StatusInternalServerError)
			io.WriteString(w, "boom")
			return
		}
		w.WriteHeader(http.StatusOK)
		io.WriteString(w, "ok")
	}))
	defer server.Close()

	cfg := DefaultConfig()
	cfg.BaseDelay = time.Millisecond
	cfg.MaxDelay = time.Millisecond
	cfg.RetryOnStatus = map[int]bool{http.StatusInternalServerError: true}

	resp, err := Do(context.Background(), server.Client(), func() (*http.Request, error) {
		return http.NewRequestWithContext(context.Background(), http.MethodGet, server.URL, nil)
	}, cfg)
	if err != nil {
		t.Fatalf("Do: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200", resp.StatusCode)
	}
	if attempts.Load() != 2 {
		t.Fatalf("attempts = %d, want 2", attempts.Load())
	}
}

func TestFromLLMHubCopiesOverrides(t *testing.T) {
	cfg := llmhub.NewConfig(llmhub.WithRetryOnStatus(429, false))
	rc := FromLLMHub(cfg)
	if rc.RetryOnStatus[429] {
		t.Fatalf("expected 429 retry disabled")
	}
	rc.RetryOnStatus[429] = true
	if cfg.RetryOnStatus[429] {
		t.Fatalf("FromLLMHub mutation leaked into llmhub.Config")
	}
}

func TestDoRecordsRetryAttempts(t *testing.T) {
	var attempts atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if attempts.Add(1) == 1 {
			w.Header().Set("Retry-After", "1")
			w.WriteHeader(http.StatusTooManyRequests)
			io.WriteString(w, "rate limited")
			return
		}
		w.WriteHeader(http.StatusOK)
		io.WriteString(w, "ok")
	}))
	defer server.Close()

	cfg := Config{MaxRetries: 1, BaseDelay: time.Millisecond, MaxDelay: time.Millisecond}
	ctx, session := telemetry.WithSession(context.Background())
	resp, err := Do(ctx, server.Client(), func() (*http.Request, error) {
		return http.NewRequestWithContext(ctx, http.MethodGet, server.URL, nil)
	}, cfg)
	if err != nil {
		t.Fatalf("Do: %v", err)
	}
	defer resp.Body.Close()

	recorded := session.Attempts()
	if len(recorded) != 2 {
		t.Fatalf("recorded attempts = %d, want 2", len(recorded))
	}
	first, last := recorded[0], recorded[1]
	if first.StatusCode != http.StatusTooManyRequests {
		t.Errorf("first status = %d, want 429", first.StatusCode)
	}
	if first.Duration <= 0 {
		t.Errorf("first duration = %v, want > 0", first.Duration)
	}
	if first.Backoff != time.Millisecond {
		t.Errorf("first backoff = %v, want 1ms (capped)", first.Backoff)
	}
	if first.RetryAfter != time.Second {
		t.Errorf("first retry-after = %v, want 1s (raw header)", first.RetryAfter)
	}
	if first.Err != "" {
		t.Errorf("first Err = %q, want empty", first.Err)
	}
	if last.StatusCode != http.StatusOK {
		t.Errorf("last status = %d, want 200", last.StatusCode)
	}
	if last.Backoff != 0 || last.RetryAfter != 0 {
		t.Errorf("final attempt backoff/retry-after = %v/%v, want 0/0", last.Backoff, last.RetryAfter)
	}
	if last.Err != "" {
		t.Errorf("last Err = %q, want empty", last.Err)
	}
}

func TestDoRecordsTransportError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))
	url := server.URL
	client := server.Client()
	server.Close()

	ctx, session := telemetry.WithSession(context.Background())
	_, err := Do(ctx, client, func() (*http.Request, error) {
		return http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	}, DefaultConfig())
	if err == nil {
		t.Fatal("Do succeeded against a closed server")
	}

	recorded := session.Attempts()
	if len(recorded) != 1 {
		t.Fatalf("recorded attempts = %d, want 1", len(recorded))
	}
	if recorded[0].StatusCode != 0 {
		t.Errorf("status = %d, want 0 for a transport error", recorded[0].StatusCode)
	}
	if recorded[0].Err == "" {
		t.Error("Err is empty, want the transport error")
	}
}

func TestDoWithoutSessionDoesNotPanic(t *testing.T) {
	var attempts atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if attempts.Add(1) == 1 {
			w.WriteHeader(http.StatusTooManyRequests)
			return
		}
		w.WriteHeader(http.StatusOK)
	}))
	defer server.Close()

	cfg := DefaultConfig()
	cfg.BaseDelay = time.Millisecond
	cfg.MaxDelay = time.Millisecond

	resp, err := Do(context.Background(), server.Client(), func() (*http.Request, error) {
		return http.NewRequestWithContext(context.Background(), http.MethodGet, server.URL, nil)
	}, cfg)
	if err != nil {
		t.Fatalf("Do: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200", resp.StatusCode)
	}
	if attempts.Load() != 2 {
		t.Fatalf("attempts = %d, want 2", attempts.Load())
	}
}

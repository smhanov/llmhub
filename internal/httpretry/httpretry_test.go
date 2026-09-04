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

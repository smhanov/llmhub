package llmhub_test

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"

	"github.com/smhanov/llmhub"
	"github.com/smhanov/llmhub/providers/openai"
)

func newOpenAIClient(t *testing.T, server *httptest.Server) *llmhub.Client {
	t.Helper()
	provider, err := openai.New("test-key",
		llmhub.WithBaseURL(server.URL),
		llmhub.WithHTTPClient(server.Client()),
		llmhub.WithModel("test-model"),
	)
	if err != nil {
		t.Fatalf("openai.New: %v", err)
	}
	return llmhub.Wrap(provider)
}

func TestTimingRetryGenerate(t *testing.T) {
	var attempts atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if attempts.Add(1) == 1 {
			w.WriteHeader(http.StatusTooManyRequests)
			io.WriteString(w, `{"error":{"message":"rate limited"}}`)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		io.WriteString(w, `{
			"id":"chatcmpl-retry",
			"choices":[{"index":0,"message":{"role":"assistant","content":"hello"},"finish_reason":"stop"}],
			"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}
		}`)
	}))
	defer server.Close()

	client := newOpenAIClient(t, server)
	resp, err := client.Generate(context.Background(), []*llmhub.Message{
		llmhub.NewUserMessage(llmhub.Text("hi")),
	})
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if got := resp.Text(); got != "hello" {
		t.Fatalf("Text = %q, want hello", got)
	}
	if resp.Timing == nil {
		t.Fatal("response missing Timing")
	}
	timing := resp.Timing
	if len(timing.Attempts) != 2 {
		t.Fatalf("Attempts = %+v, want 2 entries", timing.Attempts)
	}
	if timing.Retries != 1 {
		t.Errorf("Retries = %d, want 1", timing.Retries)
	}
	if timing.RetryDelay <= 0 {
		t.Errorf("RetryDelay = %v, want > 0", timing.RetryDelay)
	}
	if timing.Total < timing.RetryDelay {
		t.Errorf("Total = %v < RetryDelay = %v", timing.Total, timing.RetryDelay)
	}
	if got := timing.Attempts[0].StatusCode; got != http.StatusTooManyRequests {
		t.Errorf("first status = %d, want 429", got)
	}
	if got := timing.Attempts[len(timing.Attempts)-1].StatusCode; got != http.StatusOK {
		t.Errorf("last status = %d, want 200", got)
	}
	if timing.Attempts[0].Backoff != timing.RetryDelay {
		t.Errorf("attempt backoff = %v, RetryDelay = %v", timing.Attempts[0].Backoff, timing.RetryDelay)
	}
	if timing.Attempts[1].Backoff != 0 {
		t.Errorf("final attempt backoff = %v, want 0", timing.Attempts[1].Backoff)
	}
}

func TestTimingRetryStream(t *testing.T) {
	var attempts atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if attempts.Add(1) == 1 {
			w.WriteHeader(http.StatusTooManyRequests)
			io.WriteString(w, `{"error":{"message":"rate limited"}}`)
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		fmt.Fprint(w, "data: {\"id\":\"chatcmpl-stream\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hello\"},\"finish_reason\":null}]}\n\n")
		fmt.Fprint(w, "data: {\"id\":\"chatcmpl-stream\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n")
		fmt.Fprint(w, "data: [DONE]\n\n")
		if flusher, ok := w.(http.Flusher); ok {
			flusher.Flush()
		}
	}))
	defer server.Close()

	client := newOpenAIClient(t, server)
	stream, err := client.Stream(context.Background(), []*llmhub.Message{
		llmhub.NewUserMessage(llmhub.Text("hi")),
	})
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}

	var (
		content  string
		terminal *llmhub.StreamChunk
	)
	for chunk := range stream {
		if chunk.Err != nil {
			t.Fatalf("stream error: %v", chunk.Err)
		}
		content += chunk.Delta
		if chunk.Done {
			copied := chunk
			terminal = &copied
		}
	}
	if content != "hello" {
		t.Fatalf("content = %q, want hello", content)
	}
	if terminal == nil {
		t.Fatal("stream produced no terminal chunk")
	}
	if terminal.Timing == nil {
		t.Fatal("terminal chunk missing Timing")
	}
	timing := terminal.Timing
	if len(timing.Attempts) != 2 {
		t.Fatalf("Attempts = %+v, want 2 entries", timing.Attempts)
	}
	if timing.Retries != 1 {
		t.Errorf("Retries = %d, want 1", timing.Retries)
	}
	if timing.RetryDelay <= 0 {
		t.Errorf("RetryDelay = %v, want > 0", timing.RetryDelay)
	}
	if timing.Total < timing.RetryDelay {
		t.Errorf("Total = %v < RetryDelay = %v", timing.Total, timing.RetryDelay)
	}
	if got := timing.Attempts[0].StatusCode; got != http.StatusTooManyRequests {
		t.Errorf("first status = %d, want 429", got)
	}
	if got := timing.Attempts[len(timing.Attempts)-1].StatusCode; got != http.StatusOK {
		t.Errorf("last status = %d, want 200", got)
	}
}

func TestTimingNoRetry(t *testing.T) {
	var attempts atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		attempts.Add(1)
		w.Header().Set("Content-Type", "application/json")
		io.WriteString(w, `{
			"id":"chatcmpl-once",
			"choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],
			"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}
		}`)
	}))
	defer server.Close()

	client := newOpenAIClient(t, server)
	resp, err := client.Generate(context.Background(), []*llmhub.Message{
		llmhub.NewUserMessage(llmhub.Text("hi")),
	})
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if got := attempts.Load(); got != 1 {
		t.Fatalf("server attempts = %d, want 1", got)
	}
	if resp.Timing == nil {
		t.Fatal("response missing Timing")
	}
	timing := resp.Timing
	if len(timing.Attempts) != 1 {
		t.Fatalf("Attempts = %+v, want 1 entry", timing.Attempts)
	}
	if timing.Retries != 0 {
		t.Errorf("Retries = %d, want 0", timing.Retries)
	}
	if timing.RetryDelay != 0 {
		t.Errorf("RetryDelay = %v, want 0", timing.RetryDelay)
	}
	if got := timing.Attempts[0].StatusCode; got != http.StatusOK {
		t.Errorf("status = %d, want 200", got)
	}
	if timing.Attempts[0].Backoff != 0 {
		t.Errorf("backoff = %v, want 0", timing.Attempts[0].Backoff)
	}
}

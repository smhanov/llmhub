package openrouter

import (
	"context"
	"fmt"
	"io"
	"math"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/smhanov/llmhub"
)

func TestOpenRouterGenerateWithReturnedCost(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" {
			t.Fatalf("unexpected path: %s", r.URL.Path)
		}
		if r.Header.Get("Authorization") != "Bearer test-key" {
			t.Fatalf("unexpected auth: %s", r.Header.Get("Authorization"))
		}
		io.WriteString(w, `{
			"id": "gen-123",
			"choices": [{"message": {"role": "assistant", "content": "hello world"}}],
			"usage": {
				"prompt_tokens": 100,
				"completion_tokens": 50,
				"total_tokens": 150,
				"cost": 0.00042
			}
		}`)
	}))
	defer server.Close()

	provider, err := New("test-key", llmhub.WithBaseURL(server.URL))
	if err != nil {
		t.Fatalf("new: %v", err)
	}

	client := llmhub.Wrap(provider, llmhub.WithCost(10.0, 30.0))
	resp, err := client.Generate(context.Background(), []*llmhub.Message{
		llmhub.NewUserMessage(llmhub.Text("hi")),
	})
	if err != nil {
		t.Fatalf("generate: %v", err)
	}

	if resp.Text() != "hello world" {
		t.Fatalf("unexpected text: %s", resp.Text())
	}
	// The provider-returned cost of 0.00042 must override WithCost rates (which would have been 0.0025)
	expectedCost := 0.00042
	if math.Abs(resp.Usage.Cost-expectedCost) > 1e-9 {
		t.Fatalf("expected provider cost %f, got %f", expectedCost, resp.Usage.Cost)
	}
}

func TestOpenRouterStream(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, _ := w.(http.Flusher)
		fmt.Fprintf(w, "data: {\"id\":\"x\",\"choices\":[{\"delta\":{\"content\":\"streamed text\"}}]}\n\n")
		flusher.Flush()
		fmt.Fprintf(w, "data: [DONE]\n\n")
		flusher.Flush()
	}))
	defer server.Close()

	provider, err := New("test-key", llmhub.WithBaseURL(server.URL))
	if err != nil {
		t.Fatalf("new: %v", err)
	}

	stream, err := provider.Stream(context.Background(), []*llmhub.Message{llmhub.NewUserMessage(llmhub.Text("hi"))})
	if err != nil {
		t.Fatalf("stream: %v", err)
	}
	chunk := <-stream
	if chunk.Delta != "streamed text" {
		t.Fatalf("unexpected delta: %+v", chunk)
	}
}

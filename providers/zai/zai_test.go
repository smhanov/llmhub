package zai

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/smhanov/llmhub"
)

func TestZAIGenerateUsesExactBaseURL(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/paas/v4/chat/completions" {
			t.Fatalf("unexpected path: %s", r.URL.Path)
		}
		if r.Header.Get("Authorization") != "Bearer test-key" {
			t.Fatalf("unexpected auth: %s", r.Header.Get("Authorization"))
		}
		io.WriteString(w, `{"id":"z","choices":[{"message":{"role":"assistant","content":"ok"}}],"usage":{"prompt_tokens":1,"completion_tokens":2,"total_tokens":3}}`)
	}))
	defer server.Close()

	provider, err := New("test-key", llmhub.WithBaseURL(server.URL+"/api/paas/v4"))
	if err != nil {
		t.Fatalf("new: %v", err)
	}
	resp, err := provider.Generate(context.Background(), []*llmhub.Message{
		llmhub.NewUserMessage(llmhub.Text("hi")),
	})
	if err != nil {
		t.Fatalf("generate: %v", err)
	}
	if resp.Text() != "ok" {
		t.Fatalf("unexpected text: %s", resp.Text())
	}
}

func TestZAICodingPlanPath(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/coding/paas/v4/chat/completions" {
			t.Fatalf("unexpected path: %s", r.URL.Path)
		}
		io.WriteString(w, `{"id":"z","choices":[{"message":{"role":"assistant","content":"plan"}}],"usage":{}}`)
	}))
	defer server.Close()

	provider, err := NewCodingPlan("test-key", llmhub.WithBaseURL(server.URL+"/api/coding/paas/v4"))
	if err != nil {
		t.Fatalf("new: %v", err)
	}
	resp, err := provider.Generate(context.Background(), []*llmhub.Message{
		llmhub.NewUserMessage(llmhub.Text("hi")),
	})
	if err != nil {
		t.Fatalf("generate: %v", err)
	}
	if resp.Text() != "plan" {
		t.Fatalf("unexpected text: %s", resp.Text())
	}
}

func TestZAIStreamHTTPErrorIsSynchronous(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		io.WriteString(w, `{"error":{"message":"bad request"}}`)
	}))
	defer server.Close()

	provider, err := New("test-key", llmhub.WithBaseURL(server.URL+"/api/paas/v4"))
	if err != nil {
		t.Fatalf("new: %v", err)
	}
	_, err = provider.Stream(context.Background(), []*llmhub.Message{
		llmhub.NewUserMessage(llmhub.Text("hi")),
	})
	if err == nil {
		t.Fatal("expected stream error")
	}
	if !strings.Contains(err.Error(), "http 400") || !strings.Contains(err.Error(), "bad request") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestZAIStream(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, _ := w.(http.Flusher)
		fmt.Fprintf(w, "data: {\"id\":\"x\",\"choices\":[{\"delta\":{\"content\":\"hello\"}}]}\n\n")
		flusher.Flush()
		fmt.Fprintf(w, "data: [DONE]\n\n")
		flusher.Flush()
	}))
	defer server.Close()

	provider, err := New("test-key", llmhub.WithBaseURL(server.URL+"/api/paas/v4"))
	if err != nil {
		t.Fatalf("new: %v", err)
	}
	stream, err := provider.Stream(context.Background(), []*llmhub.Message{
		llmhub.NewUserMessage(llmhub.Text("hi")),
	})
	if err != nil {
		t.Fatalf("stream: %v", err)
	}
	chunk := <-stream
	if chunk.Delta != "hello" {
		t.Fatalf("unexpected delta: %+v", chunk)
	}
}

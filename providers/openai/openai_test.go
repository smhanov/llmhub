package openai

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

func TestEnsureV1Suffix(t *testing.T) {
	tests := []struct {
		input string
		want  string
	}{
		{"https://api.openai.com", "https://api.openai.com/v1"},
		{"https://api.openai.com/", "https://api.openai.com/v1"},
		{"https://api.openai.com/v1", "https://api.openai.com/v1"},
		{"https://api.openai.com/v1/", "https://api.openai.com/v1"},
		{"http://localhost:11434", "http://localhost:11434/v1"},
		{"http://localhost:11434/v1", "http://localhost:11434/v1"},
		{"https://my-proxy.example.com/llm", "https://my-proxy.example.com/llm/v1"},
	}
	for _, tt := range tests {
		t.Run(tt.input, func(t *testing.T) {
			got := ensureV1Suffix(tt.input)
			if got != tt.want {
				t.Errorf("ensureV1Suffix(%q) = %q, want %q", tt.input, got, tt.want)
			}
		})
	}
}

func TestBaseURLGetsV1Suffix(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" {
			t.Fatalf("unexpected path: %s", r.URL.Path)
		}
		io.WriteString(w, `{"id":"x","choices":[{"message":{"role":"assistant","content":"ok"}}],"usage":{}}`)
	}))
	defer server.Close()

	provider, err := New("testkey", llmhub.WithModel("m"), llmhub.WithBaseURL(server.URL))
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
		t.Fatalf("unexpected response: %s", resp.Text())
	}
}

func TestDefaultModelResolution(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/v1/models":
			io.WriteString(w, `{"data":[{"id":"my-local-model"},{"id":"another-model"}]}`)
		case "/v1/chat/completions":
			defer r.Body.Close()
			data, _ := io.ReadAll(r.Body)
			var req completionRequest
			if err := json.Unmarshal(data, &req); err != nil {
				t.Fatalf("decode request: %v", err)
			}
			if req.Model != "my-local-model" {
				t.Fatalf("expected resolved model 'my-local-model', got %q", req.Model)
			}
			io.WriteString(w, `{"id":"x","choices":[{"message":{"role":"assistant","content":"resolved"}}],"usage":{}}`)
		default:
			t.Fatalf("unexpected path: %s", r.URL.Path)
		}
	}))
	defer server.Close()

	provider, err := New("testkey", llmhub.WithModel("default"), llmhub.WithBaseURL(server.URL))
	if err != nil {
		t.Fatalf("new: %v", err)
	}
	resp, err := provider.Generate(context.Background(), []*llmhub.Message{
		llmhub.NewUserMessage(llmhub.Text("hi")),
	})
	if err != nil {
		t.Fatalf("generate: %v", err)
	}
	if resp.Text() != "resolved" {
		t.Fatalf("unexpected response: %s", resp.Text())
	}
}

func TestDefaultModelNoModelsAvailable(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		io.WriteString(w, `{"data":[]}`)
	}))
	defer server.Close()

	_, err := New("testkey", llmhub.WithModel("default"), llmhub.WithBaseURL(server.URL))
	if err == nil {
		t.Fatal("expected error when no models available")
	}
}

func TestOpenAIGenerate(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" {
			t.Fatalf("unexpected path: %s", r.URL.Path)
		}
		defer r.Body.Close()
		data, _ := io.ReadAll(r.Body)
		var req completionRequest
		if err := json.Unmarshal(data, &req); err != nil {
			t.Fatalf("decode request: %v", err)
		}
		if req.Model != "test-model" {
			t.Fatalf("unexpected model: %s", req.Model)
		}
		io.WriteString(w, `{"id":"chatcmpl-1","choices":[{"message":{"role":"assistant","content":"hello world"}}],"usage":{"prompt_tokens":5,"completion_tokens":7,"total_tokens":12}}`)
	}))
	defer server.Close()

	provider, err := New("key", llmhub.WithModel("test-model"), llmhub.WithBaseURL(server.URL))
	if err != nil {
		t.Fatalf("new provider: %v", err)
	}
	resp, err := provider.Generate(context.Background(), []*llmhub.Message{
		llmhub.NewSystemMessage(llmhub.Text("You are friendly")),
		llmhub.NewUserMessage(llmhub.Text("hi")),
	})
	if err != nil {
		t.Fatalf("generate: %v", err)
	}
	if resp.Text() != "hello world" {
		t.Fatalf("unexpected response: %s", resp.Text())
	}
}

func TestOpenAIGenerateReasoningContent(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		io.WriteString(w, `{"id":"chatcmpl-1","choices":[{"message":{"role":"assistant","content":[{"type":"text","text":"answer"},{"type":"reasoning","reasoning":"thought-a"}],"reasoning_content":"thought-b"}}],"usage":{"prompt_tokens":5,"completion_tokens":7,"total_tokens":12}}`)
	}))
	defer server.Close()

	provider, err := New("key", llmhub.WithModel("test-model"), llmhub.WithBaseURL(server.URL))
	if err != nil {
		t.Fatalf("new provider: %v", err)
	}
	resp, err := provider.Generate(context.Background(), []*llmhub.Message{llmhub.NewUserMessage(llmhub.Text("hi"))})
	if err != nil {
		t.Fatalf("generate: %v", err)
	}
	if resp.Text() != "answer" {
		t.Fatalf("unexpected response text: %q", resp.Text())
	}
	if resp.ReasoningText() != "thought-athought-b" {
		t.Fatalf("unexpected reasoning text: %q", resp.ReasoningText())
	}
}

func TestOpenAIStream(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, _ := w.(http.Flusher)
		fmt.Fprintf(w, "data: {\"id\":\"x\",\"choices\":[{\"delta\":{\"content\":[{\"type\":\"text\",\"text\":\"hello\"}]}}]}\n\n")
		flusher.Flush()
		fmt.Fprintf(w, "data: [DONE]\n\n")
		flusher.Flush()
	}))
	defer server.Close()

	provider, err := New("key", llmhub.WithModel("m"), llmhub.WithBaseURL(server.URL))
	if err != nil {
		t.Fatalf("new provider: %v", err)
	}
	stream, err := provider.Stream(context.Background(), []*llmhub.Message{llmhub.NewUserMessage(llmhub.Text("hi"))})
	if err != nil {
		t.Fatalf("stream: %v", err)
	}
	chunk := <-stream
	if chunk.Delta != "hello" {
		t.Fatalf("unexpected delta: %+v", chunk)
	}
	finalChunk := <-stream
	if !finalChunk.Done {
		t.Fatalf("expected done chunk: %+v", finalChunk)
	}
}

func TestOpenAIStreamReasoning(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, _ := w.(http.Flusher)
		fmt.Fprintf(w, "data: {\"id\":\"x\",\"choices\":[{\"delta\":{\"content\":[{\"type\":\"reasoning\",\"reasoning\":\"r1\"}]}}]}\n\n")
		flusher.Flush()
		fmt.Fprintf(w, "data: [DONE]\n\n")
		flusher.Flush()
	}))
	defer server.Close()

	provider, err := New("key", llmhub.WithModel("m"), llmhub.WithBaseURL(server.URL))
	if err != nil {
		t.Fatalf("new provider: %v", err)
	}
	stream, err := provider.Stream(context.Background(), []*llmhub.Message{llmhub.NewUserMessage(llmhub.Text("hi"))})
	if err != nil {
		t.Fatalf("stream: %v", err)
	}
	chunk := <-stream
	if chunk.ReasoningDelta != "r1" {
		t.Fatalf("unexpected reasoning delta: %+v", chunk)
	}
	finalChunk := <-stream
	if !finalChunk.Done {
		t.Fatalf("expected done chunk: %+v", finalChunk)
	}
}

func TestOpenAIStreamStringDeltaContent(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, _ := w.(http.Flusher)
		fmt.Fprintf(w, "data: {\"id\":\"x\",\"choices\":[{\"delta\":{\"content\":\"hello\"}}]}\n\n")
		flusher.Flush()
		fmt.Fprintf(w, "data: [DONE]\n\n")
		flusher.Flush()
	}))
	defer server.Close()

	provider, err := New("key", llmhub.WithModel("m"), llmhub.WithBaseURL(server.URL))
	if err != nil {
		t.Fatalf("new provider: %v", err)
	}
	stream, err := provider.Stream(context.Background(), []*llmhub.Message{llmhub.NewUserMessage(llmhub.Text("hi"))})
	if err != nil {
		t.Fatalf("stream: %v", err)
	}
	chunk := <-stream
	if chunk.Delta != "hello" {
		t.Fatalf("unexpected delta: %+v", chunk)
	}
}

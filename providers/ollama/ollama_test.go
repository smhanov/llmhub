package ollama

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/smhanov/llmhub"
)

func TestOllamaGenerate(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != chatEndpoint {
			t.Fatalf("unexpected path: %s", r.URL.Path)
		}
		defer r.Body.Close()
		var req chatRequest
		data, _ := io.ReadAll(r.Body)
		if err := json.Unmarshal(data, &req); err != nil {
			t.Fatalf("unmarshal: %v", err)
		}
		if !req.Stream {
			w.Header().Set("Content-Type", "application/json")
			io.WriteString(w, `{"message":{"role":"assistant","content":"pong"},"prompt_eval_count":1,"eval_count":2,"done":true}`)
			return
		}
	}))
	defer server.Close()

	provider, err := New("", llmhub.WithBaseURL(server.URL), llmhub.WithModel("local"))
	if err != nil {
		t.Fatalf("new provider: %v", err)
	}
	resp, err := provider.Generate(context.Background(), []*llmhub.Message{llmhub.NewUserMessage(llmhub.Text("ping"))})
	if err != nil {
		t.Fatalf("generate: %v", err)
	}
	if resp.Text() != "pong" {
		t.Fatalf("unexpected response: %s", resp.Text())
	}
}

func TestOllamaStream(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		defer r.Body.Close()
		var req chatRequest
		data, _ := io.ReadAll(r.Body)
		json.Unmarshal(data, &req)
		if !req.Stream {
			io.WriteString(w, `{"message":{"role":"assistant","content":"full"},"done":true}`)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		io.WriteString(w, `{"message":{"role":"assistant","content":"hel"},"done":false}`+"\n")
		io.WriteString(w, `{"message":{"role":"assistant","content":"lo"},"done":true}`+"\n")
	}))
	defer server.Close()

	provider, err := New("", llmhub.WithBaseURL(server.URL))
	if err != nil {
		t.Fatalf("new provider: %v", err)
	}
	stream, err := provider.Stream(context.Background(), []*llmhub.Message{llmhub.NewUserMessage(llmhub.Text("ping"))})
	if err != nil {
		t.Fatalf("stream: %v", err)
	}
	first := <-stream
	second := <-stream
	if first.Delta+second.Delta != "hello" {
		t.Fatalf("unexpected concatenation: %s%s", first.Delta, second.Delta)
	}
	done := <-stream
	if !done.Done {
		t.Fatalf("expected done chunk")
	}
}

package anthropic

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

func TestAnthropicGenerate(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != messagesPath {
			t.Fatalf("unexpected path: %s", r.URL.Path)
		}
		defer r.Body.Close()
		data, _ := io.ReadAll(r.Body)
		var req anthropicRequest
		if err := json.Unmarshal(data, &req); err != nil {
			t.Fatalf("decode request: %v", err)
		}
		if req.Model != "claude-test" {
			t.Fatalf("unexpected model: %s", req.Model)
		}
		io.WriteString(w, `{"id":"msg","content":[{"type":"text","text":"hello world"}],"usage":{"input_tokens":5,"output_tokens":7}}`)
	}))
	defer server.Close()

	provider, err := New("key", llmhub.WithModel("claude-test"), llmhub.WithBaseURL(server.URL))
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

func TestAnthropicGenerateWithCost(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		io.WriteString(w, `{"id":"msg","content":[{"type":"text","text":"hello"}],"usage":{"input_tokens":5,"output_tokens":7,"cost":0.0002}}`)
	}))
	defer server.Close()

	provider, err := New("key", llmhub.WithModel("claude-test"), llmhub.WithBaseURL(server.URL))
	if err != nil {
		t.Fatalf("new provider: %v", err)
	}
	resp, err := provider.Generate(context.Background(), []*llmhub.Message{
		llmhub.NewUserMessage(llmhub.Text("hi")),
	})
	if err != nil {
		t.Fatalf("generate: %v", err)
	}
	if resp.Usage.Cost != 0.0002 {
		t.Fatalf("expected cost 0.0002, got %v", resp.Usage.Cost)
	}
}

func TestAnthropicGenerateReasoning(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		io.WriteString(w, `{"id":"msg","content":[{"type":"thinking","thinking":"hidden steps"},{"type":"text","text":"final"}],"usage":{"input_tokens":5,"output_tokens":7}}`)
	}))
	defer server.Close()

	provider, err := New("key", llmhub.WithModel("claude-test"), llmhub.WithBaseURL(server.URL))
	if err != nil {
		t.Fatalf("new provider: %v", err)
	}
	resp, err := provider.Generate(context.Background(), []*llmhub.Message{llmhub.NewUserMessage(llmhub.Text("hi"))})
	if err != nil {
		t.Fatalf("generate: %v", err)
	}
	if resp.Text() != "final" {
		t.Fatalf("unexpected response: %s", resp.Text())
	}
	if resp.ReasoningText() != "hidden steps" {
		t.Fatalf("unexpected reasoning: %s", resp.ReasoningText())
	}
}

func TestAnthropicGenerateToolCalls(t *testing.T) {
	tool := llmhub.NewTool("weather", "Get weather", map[string]interface{}{"type": "object"})
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		defer r.Body.Close()
		data, _ := io.ReadAll(r.Body)
		var req anthropicRequest
		if err := json.Unmarshal(data, &req); err != nil {
			t.Fatalf("decode request: %v", err)
		}
		if len(req.Tools) != 1 || req.Tools[0].Name != "weather" {
			t.Fatalf("expected weather tool, got %+v", req.Tools)
		}
		if req.ToolChoice == nil || req.ToolChoice.Type != "tool" || req.ToolChoice.Name != "weather" {
			t.Fatalf("unexpected tool choice: %+v", req.ToolChoice)
		}
		if len(req.Messages) != 3 || req.Messages[1].Content[0].Type != "tool_use" || req.Messages[2].Content[0].ToolUseID != "toolu-1" {
			t.Fatalf("unexpected tool messages: %+v", req.Messages)
		}
		io.WriteString(w, `{"id":"msg","content":[{"type":"tool_use","id":"toolu-2","name":"weather","input":{"city":"Toronto"}}],"usage":{"input_tokens":5,"output_tokens":7}}`)
	}))
	defer server.Close()

	provider, err := New("key", llmhub.WithModel("claude-test"), llmhub.WithBaseURL(server.URL), llmhub.WithTools(tool), llmhub.WithToolChoice(llmhub.NamedToolChoice("weather")))
	if err != nil {
		t.Fatalf("new provider: %v", err)
	}
	resp, err := provider.Generate(context.Background(), []*llmhub.Message{
		llmhub.NewUserMessage(llmhub.Text("weather?")),
		llmhub.NewAssistantMessage(llmhub.ToolCall("toolu-1", "weather", `{"city":"Paris"}`)),
		llmhub.NewToolResultMessage("toolu-1", "weather", llmhub.Text(`{"temp":20}`)),
	})
	if err != nil {
		t.Fatalf("generate: %v", err)
	}
	calls := resp.ToolCalls()
	if len(calls) != 1 || calls[0].ID != "toolu-2" || calls[0].Arguments != `{"city":"Toronto"}` {
		t.Fatalf("unexpected tool calls: %+v", calls)
	}
}

func TestAnthropicStream(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, _ := w.(http.Flusher)
		fmt.Fprintf(w, "event: content_block_delta\ndata: {\"delta\":{\"text\":\"hello\"}}\n\n")
		flusher.Flush()
		fmt.Fprintf(w, "event: message_stop\ndata: {}\n\n")
		flusher.Flush()
	}))
	defer server.Close()

	provider, err := New("key", llmhub.WithBaseURL(server.URL))
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

func TestAnthropicStreamReasoning(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, _ := w.(http.Flusher)
		fmt.Fprintf(w, "event: content_block_delta\ndata: {\"delta\":{\"thinking\":\"draft\"}}\n\n")
		flusher.Flush()
		fmt.Fprintf(w, "event: message_stop\ndata: {}\n\n")
		flusher.Flush()
	}))
	defer server.Close()

	provider, err := New("key", llmhub.WithBaseURL(server.URL))
	if err != nil {
		t.Fatalf("new provider: %v", err)
	}
	stream, err := provider.Stream(context.Background(), []*llmhub.Message{llmhub.NewUserMessage(llmhub.Text("hi"))})
	if err != nil {
		t.Fatalf("stream: %v", err)
	}
	chunk := <-stream
	if chunk.ReasoningDelta != "draft" {
		t.Fatalf("unexpected reasoning delta: %+v", chunk)
	}
}

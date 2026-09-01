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

func TestGeminiGenerateWithCost(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		io.WriteString(w, `{"candidates":[{"content":{"parts":[{"text":"pong"}]}}],"usageMetadata":{"promptTokenCount":1,"candidatesTokenCount":2,"totalTokenCount":3,"cost":0.0003}}`)
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
	if resp.Usage.Cost != 0.0003 {
		t.Fatalf("expected cost 0.0003, got %v", resp.Usage.Cost)
	}
}

func TestGeminiGenerateReasoning(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		io.WriteString(w, `{"candidates":[{"content":{"parts":[{"text":"scratch","thought":true},{"text":"answer"}]}}],"usage_metadata":{"prompt_token_count":1,"candidates_token_count":2,"total_token_count":3}}`)
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
	if resp.Text() != "answer" {
		t.Fatalf("unexpected response: %s", resp.Text())
	}
	if resp.ReasoningText() != "scratch" {
		t.Fatalf("unexpected reasoning: %s", resp.ReasoningText())
	}
}

func TestGeminiGenerateToolCalls(t *testing.T) {
	tool := llmhub.NewTool("weather", "Get weather", map[string]interface{}{"type": "object"})
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		defer r.Body.Close()
		body, _ := io.ReadAll(r.Body)
		var req geminiRequest
		if err := json.Unmarshal(body, &req); err != nil {
			t.Fatalf("decode request: %v", err)
		}
		if len(req.Tools) != 1 || len(req.Tools[0].FunctionDeclarations) != 1 || req.Tools[0].FunctionDeclarations[0].Name != "weather" {
			t.Fatalf("expected weather tool, got %+v", req.Tools)
		}
		if req.ToolConfig == nil || req.ToolConfig.FunctionCallingConfig.Mode != "ANY" {
			t.Fatalf("unexpected tool config: %+v", req.ToolConfig)
		}
		if len(req.Contents) != 3 || req.Contents[1].Parts[0].FunctionCall == nil || req.Contents[2].Parts[0].FunctionResponse == nil {
			t.Fatalf("unexpected tool contents: %+v", req.Contents)
		}
		io.WriteString(w, `{"candidates":[{"content":{"parts":[{"functionCall":{"name":"weather","args":{"city":"Toronto"}}}]}}],"usage_metadata":{"prompt_token_count":1,"candidates_token_count":2,"total_token_count":3}}`)
	}))
	defer server.Close()

	provider, err := New("secret", llmhub.WithBaseURL(server.URL), llmhub.WithModel("flash-test"), llmhub.WithTools(tool), llmhub.WithToolChoice(llmhub.NamedToolChoice("weather")))
	if err != nil {
		t.Fatalf("provider new: %v", err)
	}
	resp, err := provider.Generate(context.Background(), []*llmhub.Message{
		llmhub.NewUserMessage(llmhub.Text("weather?")),
		llmhub.NewAssistantMessage(llmhub.ToolCall("", "weather", `{"city":"Paris"}`)),
		llmhub.NewToolResultMessage("", "weather", llmhub.Text(`{"temp":20}`)),
	})
	if err != nil {
		t.Fatalf("generate: %v", err)
	}
	calls := resp.ToolCalls()
	if len(calls) != 1 || calls[0].Name != "weather" || calls[0].Arguments != `{"city":"Toronto"}` {
		t.Fatalf("unexpected tool calls: %+v", calls)
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

func TestGeminiStreamReasoning(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprintln(w, `{"candidates":[{"content":{"parts":[{"text":"draft","thought":true}]}}]}`)
		fmt.Fprintln(w, `{"candidates":[{"content":{"parts":[{"text":"done"}]}}]}`)
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
	if first.ReasoningDelta != "draft" || second.Delta != "done" {
		t.Fatalf("unexpected chunks: %v %v", first, second)
	}
}

func TestGeminiStreamSSEDataLines(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprintln(w, `data: {"candidates":[{"content":{"parts":[{"text":"hello"}]}}]}`)
		fmt.Fprintln(w)
		fmt.Fprintln(w, `data: [DONE]`)
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
	if first.Delta != "hello" {
		t.Fatalf("unexpected chunk: %+v", first)
	}
}

func TestGeminiStreamArrayPayload(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprintln(w, `[{"candidates":[{"content":{"parts":[{"text":"hello"}]}}]}]`)
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
	if first.Delta != "hello" {
		t.Fatalf("unexpected chunk: %+v", first)
	}
}

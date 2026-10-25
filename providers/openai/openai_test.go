package openai

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

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
		// A base URL that already pins a version segment (z.ai uses /v4) must
		// not get "/v1" appended.
		{"https://api.z.ai/api/paas/v4", "https://api.z.ai/api/paas/v4"},
		{"https://api.z.ai/api/paas/v4/", "https://api.z.ai/api/paas/v4"},
		{"https://example.test/api/v2", "https://example.test/api/v2"},
		{"https://example.test/v1beta", "https://example.test/v1beta/v1"},
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

func TestOpenAIGeneratePromptCacheDisable(t *testing.T) {
	prompt := []*llmhub.Message{
		llmhub.NewUserMessage(llmhub.Text("hi")),
	}

	run := func(t *testing.T, opts ...llmhub.Option) map[string]json.RawMessage {
		t.Helper()
		var received map[string]json.RawMessage
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			defer r.Body.Close()
			data, _ := io.ReadAll(r.Body)
			if err := json.Unmarshal(data, &received); err != nil {
				t.Errorf("decode request: %v", err)
			}
			io.WriteString(w, `{"id":"chatcmpl-1","choices":[{"message":{"role":"assistant","content":"ok"}}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`)
		}))
		defer server.Close()

		provider, err := New("key", append([]llmhub.Option{
			llmhub.WithModel("test-model"),
			llmhub.WithBaseURL(server.URL),
		}, opts...)...)
		if err != nil {
			t.Fatalf("new provider: %v", err)
		}
		if _, err := provider.Generate(context.Background(), prompt); err != nil {
			t.Fatalf("generate: %v", err)
		}
		return received
	}

	// Opting out disables OpenAI's server-side automatic caching natively.
	received := run(t, llmhub.WithPromptCache(false))
	if string(received["prompt_cache_options"]) != `{"mode":"explicit"}` {
		t.Fatalf("expected prompt_cache_options explicit mode, got %s", string(received["prompt_cache_options"]))
	}
	if _, ok := received["prompt_cache_key"]; ok {
		t.Fatalf("prompt_cache_key must be absent when caching is disabled")
	}

	// Default (caching enabled) emits no cache-control fields.
	received = run(t)
	if _, ok := received["prompt_cache_options"]; ok {
		t.Fatalf("prompt_cache_options must be absent by default, got %s", string(received["prompt_cache_options"]))
	}
	if _, ok := received["prompt_cache_key"]; ok {
		t.Fatalf("prompt_cache_key must be absent without a cache key")
	}
}

func TestOpenAIGenerateWithCost(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		io.WriteString(w, `{"id":"chatcmpl-1","choices":[{"message":{"role":"assistant","content":"hello"}}],"usage":{"prompt_tokens":5,"completion_tokens":7,"total_tokens":12,"cost":0.00015}}`)
	}))
	defer server.Close()

	provider, err := New("key", llmhub.WithModel("test-model"), llmhub.WithBaseURL(server.URL))
	if err != nil {
		t.Fatalf("new provider: %v", err)
	}
	resp, err := provider.Generate(context.Background(), []*llmhub.Message{
		llmhub.NewUserMessage(llmhub.Text("hi")),
	})
	if err != nil {
		t.Fatalf("generate: %v", err)
	}
	if resp.Usage.Cost != 0.00015 {
		t.Fatalf("expected cost 0.00015, got %v", resp.Usage.Cost)
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

func TestOpenAIGenerateToolCalls(t *testing.T) {
	tool := llmhub.NewTool("weather", "Get weather", map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"city": map[string]interface{}{"type": "string"},
		},
	})
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		defer r.Body.Close()
		data, _ := io.ReadAll(r.Body)
		var req completionRequest
		if err := json.Unmarshal(data, &req); err != nil {
			t.Fatalf("decode request: %v", err)
		}
		if len(req.Tools) != 1 || req.Tools[0].Function.Name != "weather" {
			t.Fatalf("expected weather tool, got %+v", req.Tools)
		}
		if len(req.Messages) != 3 || len(req.Messages[1].ToolCalls) != 1 || req.Messages[2].ToolCallID != "call-1" {
			t.Fatalf("unexpected tool messages: %+v", req.Messages)
		}
		io.WriteString(w, `{"id":"chatcmpl-tool","choices":[{"message":{"role":"assistant","content":null,"tool_calls":[{"id":"call-2","type":"function","function":{"name":"weather","arguments":"{\"city\":\"Toronto\"}"}}]}}],"usage":{}}`)
	}))
	defer server.Close()

	provider, err := New("key", llmhub.WithModel("test-model"), llmhub.WithBaseURL(server.URL), llmhub.WithTools(tool), llmhub.WithToolChoice(llmhub.NamedToolChoice("weather")))
	if err != nil {
		t.Fatalf("new provider: %v", err)
	}
	resp, err := provider.Generate(context.Background(), []*llmhub.Message{
		llmhub.NewUserMessage(llmhub.Text("weather?")),
		llmhub.NewAssistantMessage(llmhub.ToolCall("call-1", "weather", `{"city":"Paris"}`)),
		llmhub.NewToolResultMessage("call-1", "weather", llmhub.Text(`{"temp":20}`)),
	})
	if err != nil {
		t.Fatalf("generate: %v", err)
	}
	calls := resp.ToolCalls()
	if len(calls) != 1 || calls[0].ID != "call-2" || calls[0].Name != "weather" {
		t.Fatalf("unexpected tool calls: %+v", calls)
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

func TestOpenAIStreamToolCalls(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, _ := w.(http.Flusher)
		fmt.Fprintf(w, "data: {\"id\":\"x\",\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call-1\",\"type\":\"function\",\"function\":{\"name\":\"weather\",\"arguments\":\"{\\\"city\\\":\\\"Toronto\\\"}\"}}]}}]}\n\n")
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
	if len(chunk.ToolCalls) != 1 || chunk.ToolCalls[0].Name != "weather" || chunk.ToolCalls[0].Index != 0 {
		t.Fatalf("unexpected tool call chunk: %+v", chunk)
	}
}

func TestOpenAIStreamParallelToolCallIndices(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, _ := w.(http.Flusher)
		fmt.Fprintf(w, "data: {\"id\":\"x\",\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":1,\"function\":{\"arguments\":\"true\"}}]}}]}\n\n")
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
	if len(chunk.ToolCalls) != 1 || chunk.ToolCalls[0].Index != 1 || chunk.ToolCalls[0].Arguments != "true" {
		t.Fatalf("unexpected indexed tool call chunk: %+v", chunk)
	}
}

func TestOpenAICustomHeaders(t *testing.T) {
	var gotAuth, gotCustom string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotAuth = r.Header.Get("Authorization")
		gotCustom = r.Header.Get("X-Custom-Header")
		io.WriteString(w, `{"id":"x","choices":[{"message":{"role":"assistant","content":"ok"}}],"usage":{}}`)
	}))
	defer server.Close()

	provider, err := New("testkey",
		llmhub.WithBaseURL(server.URL),
		llmhub.WithHeader("X-Custom-Header", "my-value"),
	)
	if err != nil {
		t.Fatalf("new: %v", err)
	}
	_, err = provider.Generate(context.Background(), []*llmhub.Message{llmhub.NewUserMessage(llmhub.Text("hi"))})
	if err != nil {
		t.Fatalf("generate: %v", err)
	}
	if gotAuth != "Bearer testkey" {
		t.Fatalf("expected auth 'Bearer testkey', got %q", gotAuth)
	}
	if gotCustom != "my-value" {
		t.Fatalf("expected custom header 'my-value', got %q", gotCustom)
	}
}

func TestOpenAIHTTPError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
		io.WriteString(w, `{"error":{"message":"invalid api key"}}`)
	}))
	defer server.Close()

	provider, err := New("badkey", llmhub.WithBaseURL(server.URL))
	if err != nil {
		t.Fatalf("new: %v", err)
	}
	_, err = provider.Generate(context.Background(), []*llmhub.Message{llmhub.NewUserMessage(llmhub.Text("hi"))})
	if err == nil {
		t.Fatal("expected error on 401")
	}
	if !strings.Contains(err.Error(), "http 401") || !strings.Contains(err.Error(), "invalid api key") {
		t.Fatalf("unexpected error message: %v", err)
	}

	_, err = provider.Stream(context.Background(), []*llmhub.Message{llmhub.NewUserMessage(llmhub.Text("hi"))})
	if err == nil {
		t.Fatal("expected stream error on 401")
	}
	if !strings.Contains(err.Error(), "http 401") || !strings.Contains(err.Error(), "invalid api key") {
		t.Fatalf("unexpected stream error: %v", err)
	}
}

func TestOpenAIContextCancellation(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		time.Sleep(100 * time.Millisecond)
		io.WriteString(w, `{"id":"x","choices":[{"message":{"role":"assistant","content":"ok"}}],"usage":{}}`)
	}))
	defer server.Close()

	provider, err := New("testkey", llmhub.WithBaseURL(server.URL))
	if err != nil {
		t.Fatalf("new: %v", err)
	}

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	_, err = provider.Generate(ctx, []*llmhub.Message{llmhub.NewUserMessage(llmhub.Text("hi"))})
	if err == nil {
		t.Fatal("expected error on canceled context")
	}
}

func TestOpenAICustomHTTPClient(t *testing.T) {
	called := false
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		io.WriteString(w, `{"id":"x","choices":[{"message":{"role":"assistant","content":"ok"}}],"usage":{}}`)
	}))
	defer server.Close()

	customTransport := &testRoundTripper{
		roundTrip: func(req *http.Request) (*http.Response, error) {
			called = true
			return http.DefaultTransport.RoundTrip(req)
		},
	}
	customClient := &http.Client{Transport: customTransport}

	provider, err := New("testkey", llmhub.WithBaseURL(server.URL), llmhub.WithHTTPClient(customClient))
	if err != nil {
		t.Fatalf("new: %v", err)
	}
	_, err = provider.Generate(context.Background(), []*llmhub.Message{llmhub.NewUserMessage(llmhub.Text("hi"))})
	if err != nil {
		t.Fatalf("generate: %v", err)
	}
	if !called {
		t.Fatal("custom HTTP client was not called")
	}
}

type testRoundTripper struct {
	roundTrip func(req *http.Request) (*http.Response, error)
}

func (t *testRoundTripper) RoundTrip(req *http.Request) (*http.Response, error) {
	return t.roundTrip(req)
}

// TestClientStreamHTTPTraceTTFB runs a full llmhub.New("openai") client
// against an httptest SSE server and verifies the built-in timing telemetry:
// FirstByte comes from httptrace, FirstToken from the first content chunk,
// and Total from the terminal frame.
func TestClientStreamHTTPTraceTTFB(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, _ := w.(http.Flusher)
		fmt.Fprintf(w, "data: {\"id\":\"x\",\"choices\":[{\"delta\":{\"content\":[{\"type\":\"text\",\"text\":\"hello\"}]}}]}\n\n")
		flusher.Flush()
		time.Sleep(10 * time.Millisecond)
		fmt.Fprintf(w, "data: [DONE]\n\n")
		flusher.Flush()
	}))
	defer server.Close()

	client, err := llmhub.New("openai", "key", llmhub.WithModel("m"), llmhub.WithBaseURL(server.URL))
	if err != nil {
		t.Fatalf("new client: %v", err)
	}
	stream, err := client.Stream(context.Background(), []*llmhub.Message{llmhub.NewUserMessage(llmhub.Text("hi"))})
	if err != nil {
		t.Fatalf("stream: %v", err)
	}
	var final llmhub.StreamChunk
	for chunk := range stream {
		if chunk.Err != nil {
			t.Fatalf("chunk error: %v", chunk.Err)
		}
		if chunk.Timing == nil {
			t.Fatalf("chunk %q missing Timing", chunk.Delta)
		}
		if chunk.Done {
			final = chunk
			break
		}
	}
	tm := final.Timing
	if tm.FirstByte <= 0 {
		t.Errorf("FirstByte = %v, want > 0 (httptrace not reported)", tm.FirstByte)
	}
	if tm.FirstToken <= 0 {
		t.Errorf("FirstToken = %v, want > 0", tm.FirstToken)
	}
	if tm.FirstByte > tm.FirstToken {
		t.Errorf("FirstByte = %v after FirstToken = %v; first byte must precede first token", tm.FirstByte, tm.FirstToken)
	}
	if tm.Total < tm.FirstToken {
		t.Errorf("Total = %v < FirstToken = %v", tm.Total, tm.FirstToken)
	}
	if tm.Chunks != 1 {
		t.Errorf("Chunks = %d, want 1", tm.Chunks)
	}
	if tm.Connected < 0 {
		t.Errorf("Connected = %v is negative", tm.Connected)
	}
}

// strictToolChoiceServer rejects tool_choice without tools the way strict
// OpenAI-compatible servers do, and otherwise serves a normal response.
func strictToolChoiceServer(t *testing.T, stream bool) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		defer r.Body.Close()
		data, _ := io.ReadAll(r.Body)
		var body map[string]json.RawMessage
		if err := json.Unmarshal(data, &body); err != nil {
			t.Errorf("decode request: %v", err)
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		_, hasChoice := body["tool_choice"]
		_, hasTools := body["tools"]
		if hasChoice && !hasTools {
			w.WriteHeader(http.StatusBadRequest)
			io.WriteString(w, `{"error":{"message":"When using tool_choice, tools must be set","type":"invalid_request_error"}}`)
			return
		}
		if !stream {
			io.WriteString(w, `{"id":"chatcmpl-1","choices":[{"message":{"role":"assistant","content":"ok"}}],"usage":{}}`)
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, _ := w.(http.Flusher)
		fmt.Fprintf(w, "data: {\"id\":\"x\",\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\n")
		flusher.Flush()
		fmt.Fprintf(w, "data: [DONE]\n\n")
		flusher.Flush()
	}))
}

func TestOpenAIGenerateToolChoiceWithoutToolsOmitted(t *testing.T) {
	server := strictToolChoiceServer(t, false)
	defer server.Close()

	provider, err := New("key", llmhub.WithModel("test-model"), llmhub.WithBaseURL(server.URL), llmhub.WithToolChoice(llmhub.AutoToolChoice()))
	if err != nil {
		t.Fatalf("new provider: %v", err)
	}
	resp, err := provider.Generate(context.Background(), []*llmhub.Message{llmhub.NewUserMessage(llmhub.Text("hi"))})
	if err != nil {
		t.Fatalf("generate must succeed without tools: %v", err)
	}
	if resp.Text() != "ok" {
		t.Fatalf("unexpected response text: %q", resp.Text())
	}
}

func TestOpenAIStreamToolChoiceWithoutToolsOmitted(t *testing.T) {
	server := strictToolChoiceServer(t, true)
	defer server.Close()

	provider, err := New("key", llmhub.WithModel("test-model"), llmhub.WithBaseURL(server.URL), llmhub.WithToolChoice(llmhub.AutoToolChoice()))
	if err != nil {
		t.Fatalf("new provider: %v", err)
	}
	stream, err := provider.Stream(context.Background(), []*llmhub.Message{llmhub.NewUserMessage(llmhub.Text("hi"))})
	if err != nil {
		t.Fatalf("stream must succeed without tools: %v", err)
	}
	chunk := <-stream
	if chunk.Delta != "ok" {
		t.Fatalf("unexpected delta: %+v", chunk)
	}
	finalChunk := <-stream
	if !finalChunk.Done {
		t.Fatalf("expected done chunk: %+v", finalChunk)
	}
}

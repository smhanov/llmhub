package anthropic

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
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

func TestAnthropicGenerateThinkingLevel(t *testing.T) {
	cases := []struct {
		name       string
		level      llmhub.ThinkingLevel
		wantNil    bool
		wantEffort string
	}{
		{name: "low", level: llmhub.ThinkingLevelLow, wantEffort: "low"},
		{name: "minimal maps to low", level: llmhub.ThinkingLevelMinimal, wantEffort: "low"},
		{name: "xhigh", level: llmhub.ThinkingLevelXHigh, wantEffort: "xhigh"},
		{name: "unset omits output_config", wantNil: true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				defer r.Body.Close()
				data, _ := io.ReadAll(r.Body)
				var req anthropicRequest
				if err := json.Unmarshal(data, &req); err != nil {
					t.Fatalf("decode request: %v", err)
				}
				if tc.wantNil {
					if req.OutputConfig != nil {
						t.Fatalf("output_config must be absent when ThinkingLevel is unset, got %+v", req.OutputConfig)
					}
				} else {
					if req.OutputConfig == nil || req.OutputConfig.Effort != tc.wantEffort {
						t.Fatalf("expected output_config.effort %q, got %+v", tc.wantEffort, req.OutputConfig)
					}
				}
				io.WriteString(w, `{"id":"msg","content":[{"type":"text","text":"ok"}],"usage":{"input_tokens":1,"output_tokens":1}}`)
			}))
			defer server.Close()

			var opts []llmhub.Option
			opts = append(opts, llmhub.WithModel("claude-test"), llmhub.WithBaseURL(server.URL))
			if tc.level != "" {
				opts = append(opts, llmhub.WithThinkingLevel(tc.level))
			}
			provider, err := New("key", opts...)
			if err != nil {
				t.Fatalf("new provider: %v", err)
			}
			if _, err := provider.Generate(context.Background(), []*llmhub.Message{
				llmhub.NewUserMessage(llmhub.Text("hi")),
			}); err != nil {
				t.Fatalf("generate: %v", err)
			}
		})
	}
}

func TestBuildRequestPayload_PromptCacheControl(t *testing.T) {
	prompt := []*llmhub.Message{
		llmhub.NewUserMessage(llmhub.Text("hi")),
	}

	// Case 1: default config - the automatic caching marker is present.
	payload, err := buildRequestPayload(prompt, llmhub.NewConfig(llmhub.WithModel("claude-test")), false)
	if err != nil {
		t.Fatalf("buildRequestPayload failed: %v", err)
	}
	var decoded map[string]json.RawMessage
	if err := json.Unmarshal(payload, &decoded); err != nil {
		t.Fatalf("unmarshal payload failed: %v", err)
	}
	var cacheControl struct {
		Type string `json:"type"`
	}
	if err := json.Unmarshal(decoded["cache_control"], &cacheControl); err != nil {
		t.Fatalf("unmarshal cache_control failed: %v", err)
	}
	if cacheControl.Type != "ephemeral" {
		t.Fatalf("expected cache_control {\"type\":\"ephemeral\"}, got %s", string(decoded["cache_control"]))
	}

	// Case 2: WithPromptCache(false) - the marker is absent.
	optOutCfg := llmhub.NewConfig(
		llmhub.WithModel("claude-test"),
		llmhub.WithPromptCache(false),
	)
	payload, err = buildRequestPayload(prompt, optOutCfg, false)
	if err != nil {
		t.Fatalf("buildRequestPayload failed: %v", err)
	}
	decoded = nil
	if err := json.Unmarshal(payload, &decoded); err != nil {
		t.Fatalf("unmarshal payload failed: %v", err)
	}
	if _, ok := decoded["cache_control"]; ok {
		t.Fatalf("cache_control must be absent when prompt caching is disabled, got %s", string(decoded["cache_control"]))
	}

	// Case 3: opt-out with a cache key set - the marker stays absent and
	// the CacheKey is not rendered on this lane (Anthropic has no
	// per-request cache-key field).
	optOutKeyedCfg := llmhub.NewConfig(
		llmhub.WithModel("claude-test"),
		llmhub.WithCacheKey("conv-1"),
		llmhub.WithPromptCache(false),
	)
	payload, err = buildRequestPayload(prompt, optOutKeyedCfg, false)
	if err != nil {
		t.Fatalf("buildRequestPayload failed: %v", err)
	}
	decoded = nil
	if err := json.Unmarshal(payload, &decoded); err != nil {
		t.Fatalf("unmarshal payload failed: %v", err)
	}
	if _, ok := decoded["cache_control"]; ok {
		t.Fatalf("cache_control must be absent when prompt caching is disabled, got %s", string(decoded["cache_control"]))
	}
	if _, ok := decoded["prompt_cache_key"]; ok {
		t.Fatalf("prompt_cache_key must never be rendered on the anthropic lane, got %s", string(decoded["prompt_cache_key"]))
	}
}

func TestBuildRequestPayload_Temperature(t *testing.T) {
	prompt := []*llmhub.Message{
		llmhub.NewUserMessage(llmhub.Text("hi")),
	}

	// Case 1: unset temperature - the field is absent from the payload.
	payload, err := buildRequestPayload(prompt, llmhub.NewConfig(llmhub.WithModel("claude-test")), false)
	if err != nil {
		t.Fatalf("buildRequestPayload failed: %v", err)
	}
	var decoded map[string]json.RawMessage
	if err := json.Unmarshal(payload, &decoded); err != nil {
		t.Fatalf("unmarshal payload failed: %v", err)
	}
	if _, ok := decoded["temperature"]; ok {
		t.Fatalf("temperature must be absent when unset, got %s", string(decoded["temperature"]))
	}

	// Case 2: explicit value - rendered verbatim.
	cfg := llmhub.NewConfig(
		llmhub.WithModel("claude-test"),
		llmhub.WithTemperature(0.5),
	)
	payload, err = buildRequestPayload(prompt, cfg, false)
	if err != nil {
		t.Fatalf("buildRequestPayload failed: %v", err)
	}
	decoded = nil
	if err := json.Unmarshal(payload, &decoded); err != nil {
		t.Fatalf("unmarshal payload failed: %v", err)
	}
	var temp float64
	if err := json.Unmarshal(decoded["temperature"], &temp); err != nil {
		t.Fatalf("unmarshal temperature failed: %v", err)
	}
	if temp != 0.5 {
		t.Fatalf("expected temperature 0.5, got %v", temp)
	}

	// Case 3: explicit zero - present as 0, not omitted.
	zeroCfg := llmhub.NewConfig(
		llmhub.WithModel("claude-test"),
		llmhub.WithTemperature(0),
	)
	payload, err = buildRequestPayload(prompt, zeroCfg, false)
	if err != nil {
		t.Fatalf("buildRequestPayload failed: %v", err)
	}
	decoded = nil
	if err := json.Unmarshal(payload, &decoded); err != nil {
		t.Fatalf("unmarshal payload failed: %v", err)
	}
	raw, ok := decoded["temperature"]
	if !ok {
		t.Fatal("explicit WithTemperature(0) must render temperature on the wire")
	}
	if string(raw) != "0" {
		t.Fatalf("expected temperature 0, got %s", string(raw))
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

func TestAnthropicStreamPreservesWhitespace(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, _ := w.(http.Flusher)
		fmt.Fprintf(w, "event: content_block_delta\ndata: {\"delta\":{\"text\":\"function \"}}\n\n")
		flusher.Flush()
		fmt.Fprintf(w, "event: content_block_delta\ndata: {\"delta\":{\"text\":\"foo()\"}}\n\n")
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
	var text string
	for chunk := range stream {
		text += chunk.Delta
	}
	if text != "function foo()" {
		t.Fatalf("unexpected concatenated text: %q", text)
	}
}

func TestAnthropicStreamUsageEvents(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, _ := w.(http.Flusher)
		fmt.Fprintf(w, "event: message_start\ndata: {\"message\":{\"usage\":{\"input_tokens\":12,\"cache_read_input_tokens\":3,\"cache_creation_input_tokens\":1}}}\n\n")
		flusher.Flush()
		fmt.Fprintf(w, "event: content_block_delta\ndata: {\"delta\":{\"text\":\"hi\"}}\n\n")
		flusher.Flush()
		fmt.Fprintf(w, "event: message_delta\ndata: {\"usage\":{\"output_tokens\":4}}\n\n")
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
	var chunks []llmhub.StreamChunk
	for chunk := range stream {
		chunks = append(chunks, chunk)
	}
	if len(chunks) < 3 {
		t.Fatalf("expected usage and text chunks, got %+v", chunks)
	}
	if chunks[0].Usage == nil || chunks[0].Usage.PromptTokens != 12 || chunks[0].Usage.CacheReadTokens != 3 || chunks[0].Usage.CacheCreationTokens != 1 {
		t.Fatalf("unexpected message_start usage: %+v", chunks[0])
	}
	if chunks[1].Delta != "hi" {
		t.Fatalf("unexpected text chunk: %+v", chunks[1])
	}
	if chunks[2].Usage == nil || chunks[2].Usage.CompletionTokens != 4 {
		t.Fatalf("unexpected message_delta usage: %+v", chunks[2])
	}
}

func TestAnthropicStreamFinishReason(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, _ := w.(http.Flusher)
		fmt.Fprintf(w, "event: message_start\ndata: {\"message\":{\"id\":\"msg_123\",\"usage\":{\"input_tokens\":4,\"output_tokens\":1}}}\n\n")
		flusher.Flush()
		fmt.Fprintf(w, "event: content_block_delta\ndata: {\"delta\":{\"text\":\"ok\"}}\n\n")
		flusher.Flush()
		fmt.Fprintf(w, "event: message_delta\ndata: {\"delta\":{\"stop_reason\":\"tool_use\"},\"usage\":{\"output_tokens\":2}}\n\n")
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
	var chunks []llmhub.StreamChunk
	for chunk := range stream {
		chunks = append(chunks, chunk)
	}

	// Chunk order: id (from message_start), text, finish_reason+usage, done.
	var idFound bool
	var reasonFound bool
	for _, chunk := range chunks {
		if chunk.ID == "msg_123" {
			idFound = true
		}
		if chunk.FinishReason == "tool_calls" {
			reasonFound = true
			if chunk.Usage == nil {
				t.Fatalf("expected finish reason chunk to carry usage, got %+v", chunk)
			}
		}
	}
	if !idFound {
		t.Fatalf("expected message id chunk, got %+v", chunks)
	}
	if !reasonFound {
		t.Fatalf("expected tool_calls finish reason chunk, got %+v", chunks)
	}
}

func TestAnthropicStreamStopReasonMapsToStop(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, _ := w.(http.Flusher)
		fmt.Fprintf(w, "event: content_block_delta\ndata: {\"delta\":{\"text\":\"yes\"}}\n\n")
		flusher.Flush()
		fmt.Fprintf(w, "event: message_delta\ndata: {\"delta\":{\"stop_reason\":\"end_turn\"}}\n\n")
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
	var chunks []llmhub.StreamChunk
	for chunk := range stream {
		chunks = append(chunks, chunk)
	}

	var reasonFound bool
	for _, chunk := range chunks {
		if chunk.FinishReason == "stop" {
			reasonFound = true
		}
	}
	if !reasonFound {
		t.Fatalf("expected stop finish reason chunk, got %+v", chunks)
	}
}

func TestAnthropicStreamHTTPErrorIsSynchronous(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		io.WriteString(w, `{"error":{"message":"bad request"}}`)
	}))
	defer server.Close()

	provider, err := New("key", llmhub.WithBaseURL(server.URL))
	if err != nil {
		t.Fatalf("new provider: %v", err)
	}
	_, err = provider.Stream(context.Background(), []*llmhub.Message{llmhub.NewUserMessage(llmhub.Text("hi"))})
	if err == nil {
		t.Fatal("expected stream error")
	}
	if !strings.Contains(err.Error(), "http 400") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestAnthropicRoundTripReasoning(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		defer r.Body.Close()
		data, _ := io.ReadAll(r.Body)
		var req anthropicRequest
		if err := json.Unmarshal(data, &req); err != nil {
			t.Fatalf("decode request: %v", err)
		}
		if len(req.Messages) != 1 {
			t.Fatalf("expected one message, got %+v", req.Messages)
		}
		foundReasoning := false
		for _, part := range req.Messages[0].Content {
			if part.Type == "text" && part.Text == "prior thought" {
				foundReasoning = true
			}
		}
		if !foundReasoning {
			t.Fatalf("expected prior reasoning as text, got %+v", req.Messages[0].Content)
		}
		io.WriteString(w, `{"id":"msg","content":[{"type":"text","text":"ok"}],"usage":{"input_tokens":1,"output_tokens":1}}`)
	}))
	defer server.Close()

	provider, err := New("key", llmhub.WithModel("claude-test"), llmhub.WithBaseURL(server.URL))
	if err != nil {
		t.Fatalf("new provider: %v", err)
	}
	resp, err := provider.Generate(context.Background(), []*llmhub.Message{
		llmhub.NewAssistantMessage(llmhub.Reasoning("prior thought"), llmhub.Text("previous answer")),
	})
	if err != nil {
		t.Fatalf("generate: %v", err)
	}
	if resp.Text() != "ok" {
		t.Fatalf("unexpected response: %s", resp.Text())
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

func TestAnthropicStreamErrorEvent(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, _ := w.(http.Flusher)
		fmt.Fprintf(w, "event: content_block_delta\ndata: {\"delta\":{\"text\":\"partial\"}}\n\n")
		flusher.Flush()
		fmt.Fprintf(w, "event: error\ndata: {\"type\":\"error\",\"error\":{\"type\":\"overloaded_error\",\"message\":\"Overloaded\",\"code\":\"overloaded\",\"param\":\"model\"}}\n\n")
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

	var chunks []llmhub.StreamChunk
	for chunk := range stream {
		chunks = append(chunks, chunk)
	}
	if len(chunks) != 2 {
		t.Fatalf("expected partial content plus terminal error, got %d: %+v", len(chunks), chunks)
	}
	if chunks[0].Delta != "partial" {
		t.Fatalf("expected partial content before the error, got %+v", chunks[0])
	}
	terminal := chunks[1]
	if !terminal.Done || terminal.Err == nil {
		t.Fatalf("expected Done=true with Err set, got %+v", terminal)
	}

	var streamErr *llmhub.StreamError
	if !errors.As(terminal.Err, &streamErr) {
		t.Fatalf("expected *llmhub.StreamError, got %T: %v", terminal.Err, terminal.Err)
	}
	if streamErr.Provider != "anthropic" {
		t.Fatalf("expected provider anthropic, got %q", streamErr.Provider)
	}
	if streamErr.Message != "Overloaded" {
		t.Fatalf("expected message Overloaded, got %q", streamErr.Message)
	}
	if streamErr.Type != "overloaded_error" {
		t.Fatalf("expected error type to survive, got %q", streamErr.Type)
	}
	if streamErr.Code != "overloaded" {
		t.Fatalf("expected error code to survive, got %q", streamErr.Code)
	}
	if streamErr.Param != "model" {
		t.Fatalf("expected error param to survive, got %q", streamErr.Param)
	}
}

func TestBuildRequestPayload_ToolChoiceOmittedWithoutTools(t *testing.T) {
	prompt := []*llmhub.Message{
		llmhub.NewUserMessage(llmhub.Text("hi")),
	}

	// Case 1: tool choice without tools - no tool_choice key on the wire.
	cfg := llmhub.NewConfig(
		llmhub.WithModel("claude-test"),
		llmhub.WithToolChoice(llmhub.RequiredToolChoice()),
	)
	payload, err := buildRequestPayload(prompt, cfg, false)
	if err != nil {
		t.Fatalf("buildRequestPayload failed: %v", err)
	}
	var decoded map[string]json.RawMessage
	if err := json.Unmarshal(payload, &decoded); err != nil {
		t.Fatalf("unmarshal payload failed: %v", err)
	}
	if _, ok := decoded["tool_choice"]; ok {
		t.Fatalf("tool_choice must be absent without tools, got %s", string(decoded["tool_choice"]))
	}

	// Case 2: with tools - rendered with the same value as before.
	tool := llmhub.NewTool("weather", "Get weather", map[string]interface{}{"type": "object"})
	cfg = llmhub.NewConfig(
		llmhub.WithModel("claude-test"),
		llmhub.WithTools(tool),
		llmhub.WithToolChoice(llmhub.RequiredToolChoice()),
	)
	payload, err = buildRequestPayload(prompt, cfg, false)
	if err != nil {
		t.Fatalf("buildRequestPayload failed: %v", err)
	}
	decoded = nil
	if err := json.Unmarshal(payload, &decoded); err != nil {
		t.Fatalf("unmarshal payload failed: %v", err)
	}
	var choice struct {
		Type string `json:"type"`
	}
	if err := json.Unmarshal(decoded["tool_choice"], &choice); err != nil {
		t.Fatalf("unmarshal tool_choice failed: %v", err)
	}
	if choice.Type != "any" {
		t.Fatalf("expected tool_choice type 'any', got %q", choice.Type)
	}
}

func TestAnthropicGenerateUpstreamInfo(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("request-id", "req_anthropic_1")
		w.Header().Set("anthropic-ratelimit-requests-remaining", "42")
		io.WriteString(w, `{
			"id": "msg_upstream",
			"model": "claude-resolved",
			"content": [{"type": "text", "text": "hello"}],
			"usage": {"input_tokens": 4, "output_tokens": 5}
		}`)
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
	if resp.Text() != "hello" {
		t.Fatalf("Text() = %q, want hello", resp.Text())
	}
	if resp.Upstream == nil {
		t.Fatal("Upstream is nil")
	}
	if resp.Upstream.StatusCode != http.StatusOK {
		t.Fatalf("StatusCode = %d, want 200", resp.Upstream.StatusCode)
	}
	if got := resp.Upstream.Headers.Get("request-id"); got != "req_anthropic_1" {
		t.Fatalf("request-id = %q, want req_anthropic_1", got)
	}
	if got := resp.Upstream.Headers.Get("anthropic-ratelimit-requests-remaining"); got != "42" {
		t.Fatalf("anthropic-ratelimit-requests-remaining = %q, want 42", got)
	}
	if resp.Upstream.Model != "claude-resolved" {
		t.Fatalf("Model = %q, want claude-resolved", resp.Upstream.Model)
	}
	if resp.Upstream.Provider != "" || resp.Upstream.SystemFingerprint != "" {
		t.Fatalf("unexpected identity fields: %+v", resp.Upstream)
	}
}

func TestAnthropicStreamUpstreamInfo(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("request-id", "req_anthropic_stream")
		flusher, _ := w.(http.Flusher)
		fmt.Fprintf(w, "event: message_start\ndata: {\"message\":{\"id\":\"msg_1\",\"model\":\"claude-resolved\",\"usage\":{\"input_tokens\":3}}}\n\n")
		flusher.Flush()
		fmt.Fprintf(w, "event: content_block_delta\ndata: {\"delta\":{\"text\":\"hello\"}}\n\n")
		flusher.Flush()
		fmt.Fprintf(w, "event: message_stop\ndata: {}\n\n")
		flusher.Flush()
	}))
	defer server.Close()

	provider, err := New("key", llmhub.WithModel("claude-test"), llmhub.WithBaseURL(server.URL))
	if err != nil {
		t.Fatalf("new provider: %v", err)
	}
	stream, err := provider.Stream(context.Background(), []*llmhub.Message{
		llmhub.NewUserMessage(llmhub.Text("hi")),
	})
	if err != nil {
		t.Fatalf("stream: %v", err)
	}

	var chunks []llmhub.StreamChunk
	var text strings.Builder
	for chunk := range stream {
		chunks = append(chunks, chunk)
		text.WriteString(chunk.Delta)
	}
	if text.String() != "hello" {
		t.Fatalf("streamed text = %q, want hello", text.String())
	}
	if len(chunks) == 0 {
		t.Fatal("no chunks received")
	}
	for i, chunk := range chunks {
		if chunk.Upstream == nil {
			t.Fatalf("chunk %d has nil Upstream", i)
		}
		if chunk.Upstream.StatusCode != http.StatusOK {
			t.Fatalf("chunk %d StatusCode = %d, want 200", i, chunk.Upstream.StatusCode)
		}
		if got := chunk.Upstream.Headers.Get("request-id"); got != "req_anthropic_stream" {
			t.Fatalf("chunk %d request-id = %q", i, got)
		}
		if chunk.Upstream.Model != "claude-resolved" {
			t.Fatalf("chunk %d Model = %q, want claude-resolved", i, chunk.Upstream.Model)
		}
	}
	if !chunks[len(chunks)-1].Done {
		t.Fatal("terminal chunk is not Done")
	}
	if chunks[0].Upstream == chunks[1].Upstream {
		t.Fatal("expected a distinct Upstream snapshot per chunk")
	}
}

func TestAnthropicGenerateRawJSON(t *testing.T) {
	const body = `{"id":"msg_raw","model":"claude-resolved","vendor_only":{"keep":true},"content":[{"type":"text","text":"hello"}],"usage":{"input_tokens":1,"output_tokens":1}}`
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		io.WriteString(w, body)
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
	if got := string(resp.RawJSON); got != body {
		t.Fatalf("RawJSON = %q, want exact body %q", got, body)
	}
	if !strings.Contains(string(resp.RawJSON), `"vendor_only"`) {
		t.Fatalf("RawJSON dropped an unknown field: %s", resp.RawJSON)
	}
	if resp.Text() != "hello" {
		t.Fatalf("Text = %q, want hello", resp.Text())
	}
}

func TestAnthropicStreamRawJSON(t *testing.T) {
	const frame = `{"delta":{"text":"hello"}}`
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprintf(w, "event: content_block_delta\ndata: %s\n\n", frame)
		fmt.Fprint(w, "event: message_stop\ndata: {}\n\n")
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
	var chunks []llmhub.StreamChunk
	for chunk := range stream {
		chunks = append(chunks, chunk)
	}
	if len(chunks) != 2 {
		t.Fatalf("expected 2 chunks, got %d: %+v", len(chunks), chunks)
	}
	if got := string(chunks[0].RawJSON); got != frame {
		t.Fatalf("RawJSON = %q, want %q", got, frame)
	}
	if chunks[1].RawJSON != nil {
		t.Fatalf("Done chunk RawJSON = %q, want nil", chunks[1].RawJSON)
	}
}

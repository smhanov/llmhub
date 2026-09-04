package openaichat

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/smhanov/llmhub"
)

func TestExactBaseURL(t *testing.T) {
	if got := ExactBaseURL("https://api.z.ai/api/paas/v4/"); got != "https://api.z.ai/api/paas/v4" {
		t.Fatalf("ExactBaseURL = %q", got)
	}
}

func TestEnsureV1Suffix(t *testing.T) {
	tests := []struct {
		input string
		want  string
	}{
		// No version segment: "/v1" is appended.
		{"https://api.openai.com", "https://api.openai.com/v1"},
		{"https://api.openai.com/", "https://api.openai.com/v1"},
		// Already versioned: unchanged, including non-v1 versions.
		{"https://api.openai.com/v1", "https://api.openai.com/v1"},
		{"https://api.openai.com/v1/", "https://api.openai.com/v1"},
		{"https://api.z.ai/api/paas/v4", "https://api.z.ai/api/paas/v4"},
		{"https://api.z.ai/api/paas/v4/", "https://api.z.ai/api/paas/v4"},
		{"https://example.test/api/v2", "https://example.test/api/v2"},
		{"https://example.test/api/v10", "https://example.test/api/v10"},
		// Version-looking segments that are not version segments still get /v1.
		{"https://example.test/v1beta", "https://example.test/v1beta/v1"},
		{"https://example.test/av1", "https://example.test/av1/v1"},
		{"https://example.test/llm", "https://example.test/llm/v1"},
	}
	for _, tt := range tests {
		t.Run(tt.input, func(t *testing.T) {
			if got := EnsureV1Suffix(tt.input); got != tt.want {
				t.Errorf("EnsureV1Suffix(%q) = %q, want %q", tt.input, got, tt.want)
			}
		})
	}
}

func TestClientStream_HTTPErrorIsSynchronous(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		io.WriteString(w, `{"error":{"message":"bad request"}}`)
	}))
	defer server.Close()

	client := NewClient(ClientConfig{
		ProviderName: "test-provider",
		BaseConfig: llmhub.NewConfig(
			llmhub.WithBaseURL(server.URL),
			llmhub.WithHTTPClient(server.Client()),
			llmhub.WithAPIKey("test-key"),
			llmhub.WithModel("test-model"),
		),
	})
	_, err := client.Stream(context.Background(), []*llmhub.Message{
		llmhub.NewUserMessage(llmhub.Text("Hi")),
	})
	if err == nil {
		t.Fatal("expected stream error")
	}
	if !strings.Contains(err.Error(), "http 400") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestClientGenerate_NoRetryOn429(t *testing.T) {
	var attempts atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		attempts.Add(1)
		w.WriteHeader(http.StatusTooManyRequests)
		io.WriteString(w, `{"error":{"message":"rate limited"}}`)
	}))
	defer server.Close()

	client := NewClient(ClientConfig{
		ProviderName: "test-provider",
		BaseConfig: llmhub.NewConfig(
			llmhub.WithBaseURL(server.URL),
			llmhub.WithHTTPClient(server.Client()),
			llmhub.WithAPIKey("test-key"),
			llmhub.WithModel("test-model"),
			llmhub.WithRetryOnStatus(http.StatusTooManyRequests, false),
		),
	})
	_, err := client.Generate(context.Background(), []*llmhub.Message{
		llmhub.NewUserMessage(llmhub.Text("Hi")),
	})
	if err == nil {
		t.Fatal("expected generate error")
	}
	if !strings.Contains(err.Error(), "http 429") {
		t.Fatalf("unexpected error: %v", err)
	}
	if attempts.Load() != 1 {
		t.Fatalf("attempts = %d, want 1", attempts.Load())
	}
}

func TestToolCallsFromAPIPreservesIndex(t *testing.T) {
	calls := ToolCallsFromAPI([]OpenAIToolCall{{
		Index: 2,
		ID:    "call-2",
		Function: OpenAIFunctionCall{
			Name:      "lookup",
			Arguments: `{"q":"x"}`,
		},
	}})
	if len(calls) != 1 || calls[0].Index != 2 || calls[0].ID != "call-2" {
		t.Fatalf("unexpected tool calls: %+v", calls)
	}
}

func TestBuildRequestPayload_ExtraBodyMerging(t *testing.T) {
	prompt := []*llmhub.Message{
		llmhub.NewUserMessage(llmhub.Text("Hello world")),
	}

	// Case 1: No ExtraBody
	baseCfg := llmhub.NewConfig(
		llmhub.WithModel("gpt-4o"),
		llmhub.WithTemperature(0.7),
	)
	payload, err := BuildRequestPayload(prompt, baseCfg, false)
	if err != nil {
		t.Fatalf("BuildRequestPayload failed: %v", err)
	}
	var decoded map[string]interface{}
	if err := json.Unmarshal(payload, &decoded); err != nil {
		t.Fatalf("unmarshal payload failed: %v", err)
	}
	if decoded["model"] != "gpt-4o" {
		t.Fatalf("expected model 'gpt-4o', got %v", decoded["model"])
	}

	// Case 2: ExtraBody with reasoning_effort, max_completion_tokens, custom metadata
	extraCfg := llmhub.NewConfig(
		llmhub.WithModel("o3-mini"),
		llmhub.WithExtraBody(map[string]json.RawMessage{
			"reasoning_effort":      json.RawMessage(`"high"`),
			"max_completion_tokens": json.RawMessage(`4096`),
			"metadata":              json.RawMessage(`{"environment":"production","team":"ml"}`),
		}),
	)
	payloadWithExtra, err := BuildRequestPayload(prompt, extraCfg, true)
	if err != nil {
		t.Fatalf("BuildRequestPayload with extra body failed: %v", err)
	}

	var decodedExtra map[string]json.RawMessage
	if err := json.Unmarshal(payloadWithExtra, &decodedExtra); err != nil {
		t.Fatalf("unmarshal extra payload failed: %v", err)
	}

	// Verify standard fields preserved
	if string(decodedExtra["model"]) != `"o3-mini"` {
		t.Fatalf("expected model 'o3-mini', got %s", string(decodedExtra["model"]))
	}
	if string(decodedExtra["stream"]) != `true` {
		t.Fatalf("expected stream true, got %s", string(decodedExtra["stream"]))
	}

	// Verify extra fields merged
	if string(decodedExtra["reasoning_effort"]) != `"high"` {
		t.Fatalf("expected reasoning_effort 'high', got %s", string(decodedExtra["reasoning_effort"]))
	}
	if string(decodedExtra["max_completion_tokens"]) != `4096` {
		t.Fatalf("expected max_completion_tokens 4096, got %s", string(decodedExtra["max_completion_tokens"]))
	}
	var meta map[string]string
	if err := json.Unmarshal(decodedExtra["metadata"], &meta); err != nil {
		t.Fatalf("unmarshal metadata failed: %v", err)
	}
	if meta["environment"] != "production" || meta["team"] != "ml" {
		t.Fatalf("unexpected metadata content: %+v", meta)
	}
}

func TestClientGenerate_WithExtraBody(t *testing.T) {
	var receivedBody map[string]json.RawMessage

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		defer r.Body.Close()
		bodyBytes, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(bodyBytes, &receivedBody)

		w.Header().Set("Content-Type", "application/json")
		io.WriteString(w, `{
			"id": "chatcmpl-test",
			"choices": [{
				"message": {
					"role": "assistant",
					"content": "response with extra body"
				}
			}],
			"usage": {
				"prompt_tokens": 10,
				"completion_tokens": 15,
				"total_tokens": 25
			}
		}`)
	}))
	defer server.Close()

	client := NewClient(ClientConfig{
		ProviderName: "test-provider",
		BaseConfig: llmhub.NewConfig(
			llmhub.WithBaseURL(server.URL),
			llmhub.WithHTTPClient(server.Client()),
			llmhub.WithAPIKey("test-key"),
			llmhub.WithModel("test-model"),
			llmhub.WithExtraBody(map[string]json.RawMessage{
				"reasoning_effort": json.RawMessage(`"low"`),
				"custom_flag":      json.RawMessage(`true`),
			}),
		),
	})

	resp, err := client.Generate(context.Background(), []*llmhub.Message{
		llmhub.NewUserMessage(llmhub.Text("Hello")),
	})
	if err != nil {
		t.Fatalf("Generate failed: %v", err)
	}
	if resp.Text() != "response with extra body" {
		t.Fatalf("unexpected response text: %s", resp.Text())
	}

	// Verify server received extra fields
	if string(receivedBody["reasoning_effort"]) != `"low"` {
		t.Fatalf("server did not receive reasoning_effort: %s", string(receivedBody["reasoning_effort"]))
	}
	if string(receivedBody["custom_flag"]) != `true` {
		t.Fatalf("server did not receive custom_flag: %s", string(receivedBody["custom_flag"]))
	}
}

func TestClientStream_TrailingUsageChunk(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, ok := w.(http.Flusher)
		if !ok {
			http.Error(w, "response writer does not flush", http.StatusInternalServerError)
			return
		}

		// Content chunk 1
		fmt.Fprintf(w, "data: {\"id\":\"x\",\"choices\":[{\"delta\":{\"content\":\"Hello\"}}],\"usage\":null}\n\n")
		flusher.Flush()

		// Content chunk 2
		fmt.Fprintf(w, "data: {\"id\":\"x\",\"choices\":[{\"delta\":{\"content\":\" world!\"}}]}\n\n")
		flusher.Flush()

		// Trailing usage chunk (OpenAI style when stream_options: {"include_usage": true})
		fmt.Fprintf(w, "data: {\"id\":\"x\",\"choices\":[],\"usage\":{\"prompt_tokens\":12,\"completion_tokens\":8,\"total_tokens\":20,\"cost\":0.0004}}\n\n")
		flusher.Flush()

		// Final [DONE]
		fmt.Fprintf(w, "data: [DONE]\n\n")
		flusher.Flush()
	}))
	defer server.Close()

	client := NewClient(ClientConfig{
		ProviderName: "test-provider",
		BaseConfig: llmhub.NewConfig(
			llmhub.WithBaseURL(server.URL),
			llmhub.WithHTTPClient(server.Client()),
			llmhub.WithAPIKey("test-key"),
			llmhub.WithModel("test-model"),
		),
	})

	chunksCh, err := client.Stream(context.Background(), []*llmhub.Message{
		llmhub.NewUserMessage(llmhub.Text("Hi")),
	})
	if err != nil {
		t.Fatalf("Stream failed: %v", err)
	}

	var chunks []llmhub.StreamChunk
	for chunk := range chunksCh {
		chunks = append(chunks, chunk)
	}

	// Expect 4 chunks:
	// 1. Delta: "Hello", Usage: nil
	// 2. Delta: " world!", Usage: nil
	// 3. Delta: "", Usage: populated
	// 4. Done: true
	if len(chunks) != 4 {
		t.Fatalf("expected 4 chunks, got %d: %+v", len(chunks), chunks)
	}

	if chunks[0].Delta != "Hello" || chunks[0].Usage != nil {
		t.Fatalf("unexpected chunk 0: %+v", chunks[0])
	}
	if chunks[1].Delta != " world!" || chunks[1].Usage != nil {
		t.Fatalf("unexpected chunk 1: %+v", chunks[1])
	}

	usageChunk := chunks[2]
	if usageChunk.Usage == nil {
		t.Fatalf("expected chunk 2 to have Usage, got nil")
	}
	if usageChunk.Usage.PromptTokens != 12 {
		t.Fatalf("expected PromptTokens 12, got %d", usageChunk.Usage.PromptTokens)
	}
	if usageChunk.Usage.CompletionTokens != 8 {
		t.Fatalf("expected CompletionTokens 8, got %d", usageChunk.Usage.CompletionTokens)
	}
	if usageChunk.Usage.TotalTokens != 20 {
		t.Fatalf("expected TotalTokens 20, got %d", usageChunk.Usage.TotalTokens)
	}
	if usageChunk.Usage.Cost != 0.0004 {
		t.Fatalf("expected Cost 0.0004, got %f", usageChunk.Usage.Cost)
	}

	if !chunks[3].Done {
		t.Fatalf("expected final chunk Done to be true")
	}
}

func TestClientStream_InlineUsageChunk(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, _ := w.(http.Flusher)

		// Single chunk containing both delta and usage (with total_cost)
		fmt.Fprintf(w, "data: {\"id\":\"x\",\"choices\":[{\"delta\":{\"content\":\"Final result\"}}],\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":3,\"total_tokens\":8,\"total_cost\":0.00015}}\n\n")
		flusher.Flush()

		fmt.Fprintf(w, "data: [DONE]\n\n")
		flusher.Flush()
	}))
	defer server.Close()

	client := NewClient(ClientConfig{
		ProviderName: "test-provider",
		BaseConfig: llmhub.NewConfig(
			llmhub.WithBaseURL(server.URL),
			llmhub.WithHTTPClient(server.Client()),
			llmhub.WithAPIKey("test-key"),
			llmhub.WithModel("test-model"),
		),
	})

	chunksCh, err := client.Stream(context.Background(), []*llmhub.Message{
		llmhub.NewUserMessage(llmhub.Text("Hi")),
	})
	if err != nil {
		t.Fatalf("Stream failed: %v", err)
	}

	var chunks []llmhub.StreamChunk
	for chunk := range chunksCh {
		chunks = append(chunks, chunk)
	}

	if len(chunks) != 2 {
		t.Fatalf("expected 2 chunks, got %d: %+v", len(chunks), chunks)
	}

	if chunks[0].Delta != "Final result" {
		t.Fatalf("expected Delta 'Final result', got %q", chunks[0].Delta)
	}
	if chunks[0].Usage == nil {
		t.Fatalf("expected inline Usage, got nil")
	}
	if chunks[0].Usage.TotalTokens != 8 || chunks[0].Usage.Cost != 0.00015 {
		t.Fatalf("unexpected Usage values: %+v", chunks[0].Usage)
	}
	if !chunks[1].Done {
		t.Fatalf("expected final chunk Done to be true")
	}
}

func TestBuildRequestPayload_ExtraBodyOverridesField(t *testing.T) {
	prompt := []*llmhub.Message{
		llmhub.NewUserMessage(llmhub.Text("ping")),
	}

	cfg := llmhub.NewConfig(
		llmhub.WithModel("base-model"),
		llmhub.WithTemperature(0.9),
		llmhub.WithExtraBody(map[string]json.RawMessage{
			"temperature": json.RawMessage(`0.1`),
			"user":        json.RawMessage(`"custom-user-id"`),
		}),
	)

	payload, err := BuildRequestPayload(prompt, cfg, false)
	if err != nil {
		t.Fatalf("BuildRequestPayload failed: %v", err)
	}

	var decoded map[string]interface{}
	if err := json.Unmarshal(payload, &decoded); err != nil {
		t.Fatalf("unmarshal failed: %v", err)
	}

	if decoded["temperature"] != 0.1 {
		t.Fatalf("expected overridden temperature 0.1, got %v", decoded["temperature"])
	}
	if decoded["user"] != "custom-user-id" {
		t.Fatalf("expected user 'custom-user-id', got %v", decoded["user"])
	}
}

func TestClientStream_WithExtraBody(t *testing.T) {
	var receivedBody map[string]json.RawMessage

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		defer r.Body.Close()
		bodyBytes, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(bodyBytes, &receivedBody)

		w.Header().Set("Content-Type", "text/event-stream")
		flusher, _ := w.(http.Flusher)
		fmt.Fprintf(w, "data: {\"id\":\"x\",\"choices\":[{\"delta\":{\"content\":\"streamed\"}}]}\n\n")
		flusher.Flush()
		fmt.Fprintf(w, "data: [DONE]\n\n")
		flusher.Flush()
	}))
	defer server.Close()

	client := NewClient(ClientConfig{
		ProviderName: "test-provider",
		BaseConfig: llmhub.NewConfig(
			llmhub.WithBaseURL(server.URL),
			llmhub.WithHTTPClient(server.Client()),
			llmhub.WithAPIKey("test-key"),
			llmhub.WithModel("test-model"),
		),
	})

	chunksCh, err := client.Stream(context.Background(), []*llmhub.Message{
		llmhub.NewUserMessage(llmhub.Text("Hi")),
	}, llmhub.WithExtraBody(map[string]json.RawMessage{
		"stream_options": json.RawMessage(`{"include_usage":true}`),
	}))
	if err != nil {
		t.Fatalf("Stream failed: %v", err)
	}

	for range chunksCh {
	}

	if string(receivedBody["stream_options"]) != `{"include_usage":true}` {
		t.Fatalf("stream did not receive extra body stream_options: %s", string(receivedBody["stream_options"]))
	}
}

func TestBuildRequestPayload_StreamIncludesUsageOption(t *testing.T) {
	prompt := []*llmhub.Message{
		llmhub.NewUserMessage(llmhub.Text("Hello world")),
	}
	cfg := llmhub.NewConfig(llmhub.WithModel("gpt-4o"))

	// Streaming requests always request usage so the proxy can rely on a
	// trailing usage frame for failover-before-commit accounting.
	payload, err := BuildRequestPayload(prompt, cfg, true)
	if err != nil {
		t.Fatalf("BuildRequestPayload(stream=true) failed: %v", err)
	}
	var decodedStream map[string]interface{}
	if err := json.Unmarshal(payload, &decodedStream); err != nil {
		t.Fatalf("unmarshal payload failed: %v", err)
	}
	streamOptions, ok := decodedStream["stream_options"].(map[string]interface{})
	if !ok {
		t.Fatalf("expected stream_options in stream payload, got %v", decodedStream["stream_options"])
	}
	if streamOptions["include_usage"] != true {
		t.Fatalf("expected include_usage true, got %v", streamOptions["include_usage"])
	}

	// Non-streaming requests must not include stream_options.
	payload, err = BuildRequestPayload(prompt, cfg, false)
	if err != nil {
		t.Fatalf("BuildRequestPayload(stream=false) failed: %v", err)
	}
	var decoded map[string]interface{}
	if err := json.Unmarshal(payload, &decoded); err != nil {
		t.Fatalf("unmarshal payload failed: %v", err)
	}
	if _, present := decoded["stream_options"]; present {
		t.Fatalf("expected no stream_options in non-stream payload, got %v", decoded["stream_options"])
	}
}

func TestBuildRequestPayload_ExtraBodyStreamOptionsStillSupported(t *testing.T) {
	prompt := []*llmhub.Message{
		llmhub.NewUserMessage(llmhub.Text("Hello world")),
	}
	cfg := llmhub.NewConfig(
		llmhub.WithModel("gpt-4o"),
		llmhub.WithExtraBody(map[string]json.RawMessage{
			"stream_options": json.RawMessage(`{"include_usage":false}`),
		}),
	)
	payload, err := BuildRequestPayload(prompt, cfg, true)
	if err != nil {
		t.Fatalf("BuildRequestPayload failed: %v", err)
	}
	var decoded map[string]json.RawMessage
	if err := json.Unmarshal(payload, &decoded); err != nil {
		t.Fatalf("unmarshal payload failed: %v", err)
	}
	if string(decoded["stream_options"]) != `{"include_usage":false}` {
		t.Fatalf("expected ExtraBody stream_options override, got %s", string(decoded["stream_options"]))
	}
}

func TestClientStream_FinishReasonSurvivesSkipFilter(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, _ := w.(http.Flusher)

		fmt.Fprintf(w, "data: {\"id\":\"stream-1\",\"choices\":[{\"delta\":{\"content\":\"hello\"}}]}\n\n")
		flusher.Flush()

		// Final chunk carries only an empty delta plus a finish reason. It must
		// survive the skip filter so callers can distinguish stop vs tool_calls.
		fmt.Fprintf(w, "data: {\"id\":\"stream-1\",\"choices\":[{\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n")
		flusher.Flush()

		fmt.Fprintf(w, "data: [DONE]\n\n")
		flusher.Flush()
	}))
	defer server.Close()

	client := NewClient(ClientConfig{
		ProviderName: "test-provider",
		BaseConfig: llmhub.NewConfig(
			llmhub.WithBaseURL(server.URL),
			llmhub.WithHTTPClient(server.Client()),
			llmhub.WithAPIKey("test-key"),
			llmhub.WithModel("test-model"),
		),
	})

	chunksCh, err := client.Stream(context.Background(), []*llmhub.Message{
		llmhub.NewUserMessage(llmhub.Text("Hi")),
	})
	if err != nil {
		t.Fatalf("Stream failed: %v", err)
	}

	var chunks []llmhub.StreamChunk
	for chunk := range chunksCh {
		chunks = append(chunks, chunk)
	}

	if len(chunks) != 3 {
		t.Fatalf("expected 3 chunks, got %d: %+v", len(chunks), chunks)
	}
	if chunks[0].Delta != "hello" || chunks[0].ID != "stream-1" {
		t.Fatalf("unexpected first chunk: %+v", chunks[0])
	}
	if chunks[1].Delta != "" || chunks[1].FinishReason != "tool_calls" {
		t.Fatalf("expected finish_reason chunk, got %+v", chunks[1])
	}
	if !chunks[2].Done {
		t.Fatalf("expected final Done chunk, got %+v", chunks[2])
	}
}

func TestClientStream_StopFinishReason(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, _ := w.(http.Flusher)

		fmt.Fprintf(w, "data: {\"id\":\"stream-2\",\"choices\":[{\"delta\":{\"content\":\"done\"},\"finish_reason\":\"stop\"}]}\n\n")
		flusher.Flush()

		fmt.Fprintf(w, "data: [DONE]\n\n")
		flusher.Flush()
	}))
	defer server.Close()

	client := NewClient(ClientConfig{
		ProviderName: "test-provider",
		BaseConfig: llmhub.NewConfig(
			llmhub.WithBaseURL(server.URL),
			llmhub.WithHTTPClient(server.Client()),
			llmhub.WithAPIKey("test-key"),
			llmhub.WithModel("test-model"),
		),
	})

	chunksCh, err := client.Stream(context.Background(), []*llmhub.Message{
		llmhub.NewUserMessage(llmhub.Text("Hi")),
	})
	if err != nil {
		t.Fatalf("Stream failed: %v", err)
	}

	var chunks []llmhub.StreamChunk
	for chunk := range chunksCh {
		chunks = append(chunks, chunk)
	}

	if chunks[0].FinishReason != "stop" {
		t.Fatalf("expected stop finish reason, got %+v", chunks[0])
	}
}

func TestClientGenerate_UsageCacheAndReasoningParity(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		io.WriteString(w, `{
			"id": "chatcmpl-usage",
			"choices": [{"message": {"role": "assistant", "content": "hello"}}],
			"usage": {
				"prompt_tokens": 20,
				"completion_tokens": 10,
				"total_tokens": 30,
				"prompt_tokens_details": {"cached_tokens": 7},
				"completion_tokens_details": {"reasoning_tokens": 4}
			}
		}`)
	}))
	defer server.Close()

	client := NewClient(ClientConfig{
		ProviderName: "test-provider",
		BaseConfig: llmhub.NewConfig(
			llmhub.WithBaseURL(server.URL),
			llmhub.WithHTTPClient(server.Client()),
			llmhub.WithAPIKey("test-key"),
			llmhub.WithModel("test-model"),
		),
	})

	resp, err := client.Generate(context.Background(), []*llmhub.Message{
		llmhub.NewUserMessage(llmhub.Text("Hi")),
	})
	if err != nil {
		t.Fatalf("Generate failed: %v", err)
	}
	if resp.Usage.CacheReadTokens != 7 {
		t.Fatalf("expected CacheReadTokens 7, got %d", resp.Usage.CacheReadTokens)
	}
	if resp.Usage.ReasoningTokens != 4 {
		t.Fatalf("expected ReasoningTokens 4, got %d", resp.Usage.ReasoningTokens)
	}
	if resp.Usage.PromptTokens != 20 || resp.Usage.CompletionTokens != 10 || resp.Usage.TotalTokens != 30 {
		t.Fatalf("unexpected usage: %+v", resp.Usage)
	}
}

func TestClientStream_UsageCacheFromTrailingChunk(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, _ := w.(http.Flusher)

		fmt.Fprintf(w, "data: {\"id\":\"stream-3\",\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n")
		flusher.Flush()
		// Trailing usage frame reporting cache_creation and cache_read tokens.
		fmt.Fprintf(w, "data: {\"id\":\"stream-3\",\"choices\":[],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":5,\"total_tokens\":15,\"cache_read_input_tokens\":3,\"cache_creation_input_tokens\":2,\"completion_tokens_details\":{\"reasoning_tokens\":1}}}\n\n")
		flusher.Flush()
		fmt.Fprintf(w, "data: [DONE]\n\n")
		flusher.Flush()
	}))
	defer server.Close()

	client := NewClient(ClientConfig{
		ProviderName: "test-provider",
		BaseConfig: llmhub.NewConfig(
			llmhub.WithBaseURL(server.URL),
			llmhub.WithHTTPClient(server.Client()),
			llmhub.WithAPIKey("test-key"),
			llmhub.WithModel("test-model"),
		),
	})

	chunksCh, err := client.Stream(context.Background(), []*llmhub.Message{
		llmhub.NewUserMessage(llmhub.Text("Hi")),
	})
	if err != nil {
		t.Fatalf("Stream failed: %v", err)
	}

	var chunks []llmhub.StreamChunk
	for chunk := range chunksCh {
		chunks = append(chunks, chunk)
	}

	usageChunk := chunks[1]
	if usageChunk.Usage == nil {
		t.Fatalf("expected usage chunk, got %+v", usageChunk)
	}
	if usageChunk.Usage.CacheReadTokens != 3 {
		t.Fatalf("expected CacheReadTokens 3, got %d", usageChunk.Usage.CacheReadTokens)
	}
	if usageChunk.Usage.CacheCreationTokens != 2 {
		t.Fatalf("expected CacheCreationTokens 2, got %d", usageChunk.Usage.CacheCreationTokens)
	}
	if usageChunk.Usage.ReasoningTokens != 1 {
		t.Fatalf("expected ReasoningTokens 1, got %d", usageChunk.Usage.ReasoningTokens)
	}
}

func TestClientStream_EmptyChoicesNoUsageIgnored(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, _ := w.(http.Flusher)

		// Empty choices chunk with no usage (should be ignored)
		fmt.Fprintf(w, "data: {\"id\":\"x\",\"choices\":[]}\n\n")
		flusher.Flush()

		// Empty choices chunk with null usage (should be ignored)
		fmt.Fprintf(w, "data: {\"id\":\"x\",\"choices\":[],\"usage\":null}\n\n")
		flusher.Flush()

		// Real chunk
		fmt.Fprintf(w, "data: {\"id\":\"x\",\"choices\":[{\"delta\":{\"content\":\"real\"}}]}\n\n")
		flusher.Flush()

		fmt.Fprintf(w, "data: [DONE]\n\n")
		flusher.Flush()
	}))
	defer server.Close()

	client := NewClient(ClientConfig{
		ProviderName: "test-provider",
		BaseConfig: llmhub.NewConfig(
			llmhub.WithBaseURL(server.URL),
			llmhub.WithHTTPClient(server.Client()),
			llmhub.WithAPIKey("test-key"),
			llmhub.WithModel("test-model"),
		),
	})

	chunksCh, err := client.Stream(context.Background(), []*llmhub.Message{
		llmhub.NewUserMessage(llmhub.Text("Hi")),
	})
	if err != nil {
		t.Fatalf("Stream failed: %v", err)
	}

	var chunks []llmhub.StreamChunk
	for chunk := range chunksCh {
		chunks = append(chunks, chunk)
	}

	// Should only receive the real chunk and DONE chunk
	if len(chunks) != 2 {
		t.Fatalf("expected 2 chunks, got %d: %+v", len(chunks), chunks)
	}
	if chunks[0].Delta != "real" {
		t.Fatalf("expected Delta 'real', got %q", chunks[0].Delta)
	}
	if !chunks[1].Done {
		t.Fatalf("expected second chunk Done to be true")
	}
}

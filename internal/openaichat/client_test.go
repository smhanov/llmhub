package openaichat

import (
	"context"
	"encoding/json"
	"errors"
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

func TestBuildRequestPayload_TemperatureOmittedUnlessSet(t *testing.T) {
	prompt := []*llmhub.Message{
		llmhub.NewUserMessage(llmhub.Text("Hello world")),
	}

	// Case 1: unset temperature - the field is absent from the payload.
	payload, err := BuildRequestPayload(prompt, llmhub.NewConfig(llmhub.WithModel("gpt-4o")), false)
	if err != nil {
		t.Fatalf("BuildRequestPayload failed: %v", err)
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
		llmhub.WithModel("gpt-4o"),
		llmhub.WithTemperature(0.5),
	)
	payload, err = BuildRequestPayload(prompt, cfg, false)
	if err != nil {
		t.Fatalf("BuildRequestPayload failed: %v", err)
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
		llmhub.WithModel("gpt-4o"),
		llmhub.WithTemperature(0),
	)
	payload, err = BuildRequestPayload(prompt, zeroCfg, false)
	if err != nil {
		t.Fatalf("BuildRequestPayload failed: %v", err)
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

func TestBuildRequestPayload_ThinkingLevel(t *testing.T) {
	prompt := []*llmhub.Message{
		llmhub.NewUserMessage(llmhub.Text("Hello world")),
	}

	// Level set: reasoning_effort is sent.
	cfg := llmhub.NewConfig(
		llmhub.WithModel("o3-mini"),
		llmhub.WithThinkingLevel(llmhub.ThinkingLevelMedium),
	)
	payload, err := BuildRequestPayload(prompt, cfg, false)
	if err != nil {
		t.Fatalf("BuildRequestPayload failed: %v", err)
	}
	var decoded map[string]json.RawMessage
	if err := json.Unmarshal(payload, &decoded); err != nil {
		t.Fatalf("unmarshal payload failed: %v", err)
	}
	if string(decoded["reasoning_effort"]) != `"medium"` {
		t.Fatalf("expected reasoning_effort 'medium', got %s", string(decoded["reasoning_effort"]))
	}

	// Unset: key is absent so the provider applies its default.
	plainCfg := llmhub.NewConfig(llmhub.WithModel("gpt-4o"))
	plainPayload, err := BuildRequestPayload(prompt, plainCfg, false)
	if err != nil {
		t.Fatalf("BuildRequestPayload failed: %v", err)
	}
	var plainDecoded map[string]json.RawMessage
	if err := json.Unmarshal(plainPayload, &plainDecoded); err != nil {
		t.Fatalf("unmarshal payload failed: %v", err)
	}
	if _, ok := plainDecoded["reasoning_effort"]; ok {
		t.Fatalf("reasoning_effort must be absent when ThinkingLevel is unset, got %s", string(plainDecoded["reasoning_effort"]))
	}

	// Explicit ExtraBody wins on collision with the mapped level.
	overrideCfg := llmhub.NewConfig(
		llmhub.WithModel("o3-mini"),
		llmhub.WithThinkingLevel(llmhub.ThinkingLevelLow),
		llmhub.WithExtraBody(map[string]json.RawMessage{
			"reasoning_effort": json.RawMessage(`"high"`),
		}),
	)
	overridePayload, err := BuildRequestPayload(prompt, overrideCfg, false)
	if err != nil {
		t.Fatalf("BuildRequestPayload failed: %v", err)
	}
	var overrideDecoded map[string]json.RawMessage
	if err := json.Unmarshal(overridePayload, &overrideDecoded); err != nil {
		t.Fatalf("unmarshal payload failed: %v", err)
	}
	if string(overrideDecoded["reasoning_effort"]) != `"high"` {
		t.Fatalf("expected ExtraBody reasoning_effort 'high' to win, got %s", string(overrideDecoded["reasoning_effort"]))
	}
}

func TestBuildRequestPayload_PromptCacheKey(t *testing.T) {
	prompt := []*llmhub.Message{
		llmhub.NewUserMessage(llmhub.Text("Hello world")),
	}

	// Case 1: default config with no cache key - the field is absent so
	// providers fall back to automatic caching.
	defaultCfg := llmhub.NewConfig(llmhub.WithModel("gpt-4o"))
	payload, err := BuildRequestPayload(prompt, defaultCfg, false)
	if err != nil {
		t.Fatalf("BuildRequestPayload failed: %v", err)
	}
	var decoded map[string]json.RawMessage
	if err := json.Unmarshal(payload, &decoded); err != nil {
		t.Fatalf("unmarshal payload failed: %v", err)
	}
	if _, ok := decoded["prompt_cache_key"]; ok {
		t.Fatalf("prompt_cache_key must be absent when no cache key is set, got %s", string(decoded["prompt_cache_key"]))
	}

	// Case 2: WithCacheKey renders the top-level prompt_cache_key field.
	keyedCfg := llmhub.NewConfig(
		llmhub.WithModel("gpt-4o"),
		llmhub.WithCacheKey("conv-1"),
	)
	payload, err = BuildRequestPayload(prompt, keyedCfg, false)
	if err != nil {
		t.Fatalf("BuildRequestPayload failed: %v", err)
	}
	decoded = nil
	if err := json.Unmarshal(payload, &decoded); err != nil {
		t.Fatalf("unmarshal payload failed: %v", err)
	}
	if string(decoded["prompt_cache_key"]) != `"conv-1"` {
		t.Fatalf("expected prompt_cache_key 'conv-1', got %s", string(decoded["prompt_cache_key"]))
	}

	// Case 3: WithPromptCache(false) suppresses the field even when a
	// cache key is also set.
	optOutCfg := llmhub.NewConfig(
		llmhub.WithModel("gpt-4o"),
		llmhub.WithCacheKey("conv-1"),
		llmhub.WithPromptCache(false),
	)
	payload, err = BuildRequestPayload(prompt, optOutCfg, false)
	if err != nil {
		t.Fatalf("BuildRequestPayload failed: %v", err)
	}
	decoded = nil
	if err := json.Unmarshal(payload, &decoded); err != nil {
		t.Fatalf("unmarshal payload failed: %v", err)
	}
	if _, ok := decoded["prompt_cache_key"]; ok {
		t.Fatalf("prompt_cache_key must be absent when prompt caching is disabled, got %s", string(decoded["prompt_cache_key"]))
	}
}

func TestClientGenerate_CacheKeyHeader(t *testing.T) {
	prompt := []*llmhub.Message{
		llmhub.NewUserMessage(llmhub.Text("Hi")),
	}

	newServer := func(header *http.Header) *httptest.Server {
		return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			*header = r.Header.Clone()
			w.Header().Set("Content-Type", "application/json")
			io.WriteString(w, `{"id":"chatcmpl-cache","choices":[{"message":{"role":"assistant","content":"ok"}}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`)
		}))
	}

	// With a cache key: the provider-declared header carries it.
	var keyedHeader http.Header
	keyedServer := newServer(&keyedHeader)
	defer keyedServer.Close()
	keyedClient := NewClient(ClientConfig{
		ProviderName:   "test-provider",
		CacheKeyHeader: "x-grok-conv-id",
		BaseConfig: llmhub.NewConfig(
			llmhub.WithBaseURL(keyedServer.URL),
			llmhub.WithHTTPClient(keyedServer.Client()),
			llmhub.WithAPIKey("test-key"),
			llmhub.WithModel("test-model"),
			llmhub.WithCacheKey("conv-1"),
		),
	})
	if _, err := keyedClient.Generate(context.Background(), prompt); err != nil {
		t.Fatalf("Generate failed: %v", err)
	}
	if got := keyedHeader.Get("x-grok-conv-id"); got != "conv-1" {
		t.Fatalf("expected x-grok-conv-id 'conv-1', got %q", got)
	}

	// Without a cache key: the header is absent.
	var emptyHeader http.Header
	emptyServer := newServer(&emptyHeader)
	defer emptyServer.Close()
	emptyClient := NewClient(ClientConfig{
		ProviderName:   "test-provider",
		CacheKeyHeader: "x-grok-conv-id",
		BaseConfig: llmhub.NewConfig(
			llmhub.WithBaseURL(emptyServer.URL),
			llmhub.WithHTTPClient(emptyServer.Client()),
			llmhub.WithAPIKey("test-key"),
			llmhub.WithModel("test-model"),
		),
	})
	if _, err := emptyClient.Generate(context.Background(), prompt); err != nil {
		t.Fatalf("Generate failed: %v", err)
	}
	if got := emptyHeader.Get("x-grok-conv-id"); got != "" {
		t.Fatalf("expected no x-grok-conv-id without a cache key, got %q", got)
	}

	// With prompt caching disabled: the header is absent even when a
	// cache key is set.
	var optOutHeader http.Header
	optOutServer := newServer(&optOutHeader)
	defer optOutServer.Close()
	optOutClient := NewClient(ClientConfig{
		ProviderName:   "test-provider",
		CacheKeyHeader: "x-grok-conv-id",
		BaseConfig: llmhub.NewConfig(
			llmhub.WithBaseURL(optOutServer.URL),
			llmhub.WithHTTPClient(optOutServer.Client()),
			llmhub.WithAPIKey("test-key"),
			llmhub.WithModel("test-model"),
			llmhub.WithCacheKey("conv-1"),
			llmhub.WithPromptCache(false),
		),
	})
	if _, err := optOutClient.Generate(context.Background(), prompt); err != nil {
		t.Fatalf("Generate failed: %v", err)
	}
	if got := optOutHeader.Get("x-grok-conv-id"); got != "" {
		t.Fatalf("expected no x-grok-conv-id when caching is disabled, got %q", got)
	}
}

func TestClientGenerate_PromptCacheDisable(t *testing.T) {
	prompt := []*llmhub.Message{
		llmhub.NewUserMessage(llmhub.Text("Hi")),
	}

	newClient := func(supportsDisable bool, extraOpts ...llmhub.Option) (*Client, *map[string]json.RawMessage, *httptest.Server) {
		var received map[string]json.RawMessage
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			data, _ := io.ReadAll(r.Body)
			_ = json.Unmarshal(data, &received)
			w.Header().Set("Content-Type", "application/json")
			io.WriteString(w, `{"id":"chatcmpl-cache-off","choices":[{"message":{"role":"assistant","content":"ok"}}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`)
		}))
		opts := append([]llmhub.Option{
			llmhub.WithBaseURL(server.URL),
			llmhub.WithHTTPClient(server.Client()),
			llmhub.WithAPIKey("test-key"),
			llmhub.WithModel("test-model"),
		}, extraOpts...)
		client := NewClient(ClientConfig{
			ProviderName:         "test-provider",
			SupportsCacheDisable: supportsDisable,
			BaseConfig:           llmhub.NewConfig(opts...),
		})
		return client, &received, server
	}

	// Case 1: capability + WithPromptCache(false) -> the explicit mode is
	// rendered and no cache key is sent.
	client, received, server := newClient(true, llmhub.WithPromptCache(false))
	defer server.Close()
	if _, err := client.Generate(context.Background(), prompt); err != nil {
		t.Fatalf("Generate failed: %v", err)
	}
	if string((*received)["prompt_cache_options"]) != `{"mode":"explicit"}` {
		t.Fatalf("expected prompt_cache_options explicit mode, got %s", string((*received)["prompt_cache_options"]))
	}
	if _, ok := (*received)["prompt_cache_key"]; ok {
		t.Fatalf("prompt_cache_key must be absent when caching is disabled")
	}

	// Case 2: no capability + WithPromptCache(false) -> nothing rendered
	// (xai/zai/openrouter lanes unchanged).
	client, received, server = newClient(false, llmhub.WithPromptCache(false))
	defer server.Close()
	if _, err := client.Generate(context.Background(), prompt); err != nil {
		t.Fatalf("Generate failed: %v", err)
	}
	if _, ok := (*received)["prompt_cache_options"]; ok {
		t.Fatalf("prompt_cache_options must be absent without the capability")
	}

	// Case 3: capability + caching enabled (default) -> nothing rendered.
	client, received, server = newClient(true)
	defer server.Close()
	if _, err := client.Generate(context.Background(), prompt); err != nil {
		t.Fatalf("Generate failed: %v", err)
	}
	if _, ok := (*received)["prompt_cache_options"]; ok {
		t.Fatalf("prompt_cache_options must be absent when caching is enabled")
	}

	// Case 4: capability + opt-out + explicit ExtraBody override ->
	// the caller-supplied value wins on collision.
	client, received, server = newClient(true,
		llmhub.WithPromptCache(false),
		llmhub.WithExtraBody(map[string]json.RawMessage{
			"prompt_cache_options": json.RawMessage(`{"mode":"custom"}`),
		}),
	)
	defer server.Close()
	if _, err := client.Generate(context.Background(), prompt); err != nil {
		t.Fatalf("Generate failed: %v", err)
	}
	if string((*received)["prompt_cache_options"]) != `{"mode":"custom"}` {
		t.Fatalf("expected ExtraBody prompt_cache_options to win, got %s", string((*received)["prompt_cache_options"]))
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

func TestBuildRequestPayload_ToolChoiceOmittedWithoutTools(t *testing.T) {
	prompt := []*llmhub.Message{
		llmhub.NewUserMessage(llmhub.Text("hi")),
	}

	// Case 1: tool choice without tools - the control is absent from the
	// payload, because strict OpenAI-compatible servers reject tool_choice
	// when no tools are supplied.
	cfg := llmhub.NewConfig(
		llmhub.WithModel("gpt-4o"),
		llmhub.WithToolChoice(llmhub.AutoToolChoice()),
	)
	payload, err := BuildRequestPayload(prompt, cfg, false)
	if err != nil {
		t.Fatalf("BuildRequestPayload failed: %v", err)
	}
	var decoded map[string]json.RawMessage
	if err := json.Unmarshal(payload, &decoded); err != nil {
		t.Fatalf("unmarshal payload failed: %v", err)
	}
	if _, ok := decoded["tool_choice"]; ok {
		t.Fatalf("tool_choice must be absent without tools, got %s", string(decoded["tool_choice"]))
	}

	// Case 2: tool choice with tools - rendered as before.
	tool := llmhub.NewTool("weather", "Get weather", map[string]interface{}{"type": "object"})
	cfg = llmhub.NewConfig(
		llmhub.WithModel("gpt-4o"),
		llmhub.WithTools(tool),
		llmhub.WithToolChoice(llmhub.AutoToolChoice()),
	)
	payload, err = BuildRequestPayload(prompt, cfg, false)
	if err != nil {
		t.Fatalf("BuildRequestPayload failed: %v", err)
	}
	decoded = nil
	if err := json.Unmarshal(payload, &decoded); err != nil {
		t.Fatalf("unmarshal payload failed: %v", err)
	}
	if string(decoded["tool_choice"]) != `"auto"` {
		t.Fatalf("expected tool_choice 'auto' with tools, got %s", string(decoded["tool_choice"]))
	}
}

func TestBuildRequestPayload_ToolChoiceSurvivesSplitOptions(t *testing.T) {
	prompt := []*llmhub.Message{
		llmhub.NewUserMessage(llmhub.Text("hi")),
	}
	tool := llmhub.NewTool("weather", "Get weather", map[string]interface{}{"type": "object"})

	// Client defaults carry the tool choice; the per-request options supply
	// the tools. The merged config must still render the control.
	cfg := llmhub.NewConfig(
		llmhub.WithModel("gpt-4o"),
		llmhub.WithToolChoice(llmhub.AutoToolChoice()),
	)
	llmhub.ApplyOptions(&cfg, llmhub.WithTools(tool))

	payload, err := BuildRequestPayload(prompt, cfg, false)
	if err != nil {
		t.Fatalf("BuildRequestPayload failed: %v", err)
	}
	var decoded map[string]json.RawMessage
	if err := json.Unmarshal(payload, &decoded); err != nil {
		t.Fatalf("unmarshal payload failed: %v", err)
	}
	if string(decoded["tool_choice"]) != `"auto"` {
		t.Fatalf("expected tool_choice 'auto' after per-request tools, got %s", string(decoded["tool_choice"]))
	}
}

// newStreamTestClient builds a client pointed at an httptest server for a
// given lane name. The shared OpenAI-compatible client is reused by openai,
// openrouter, xai, and zai, so the name must survive into the reported error.
func newStreamTestClient(server *httptest.Server, providerName string) *Client {
	return NewClient(ClientConfig{
		ProviderName: providerName,
		BaseConfig: llmhub.NewConfig(
			llmhub.WithBaseURL(server.URL),
			llmhub.WithHTTPClient(server.Client()),
			llmhub.WithAPIKey("test-key"),
			llmhub.WithModel("test-model"),
		),
	})
}

func collectStream(t *testing.T, ch <-chan llmhub.StreamChunk) []llmhub.StreamChunk {
	t.Helper()
	var chunks []llmhub.StreamChunk
	for chunk := range ch {
		chunks = append(chunks, chunk)
	}
	return chunks
}

func TestClientStream_MidStreamErrorFrame(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, ok := w.(http.Flusher)
		if !ok {
			http.Error(w, "response writer does not flush", http.StatusInternalServerError)
			return
		}

		fmt.Fprintf(w, "data: {\"id\":\"x\",\"choices\":[{\"delta\":{\"content\":\"partial\"}}]}\n\n")
		flusher.Flush()

		// Error-only frame, followed by no [DONE]: the historical failure
		// mode closed the stream as a clean empty finish.
		fmt.Fprintf(w, "data: {\"error\":{\"message\":\"upstream overloaded\",\"type\":\"server_error\",\"code\":\"overloaded\",\"param\":\"model\"}}\n\n")
		flusher.Flush()
	}))
	defer server.Close()

	chunksCh, err := newStreamTestClient(server, "test-provider").Stream(context.Background(), []*llmhub.Message{
		llmhub.NewUserMessage(llmhub.Text("Hi")),
	})
	if err != nil {
		t.Fatalf("Stream failed: %v", err)
	}

	chunks := collectStream(t, chunksCh)
	if len(chunks) != 2 {
		t.Fatalf("expected 2 chunks (partial content + terminal error), got %d: %+v", len(chunks), chunks)
	}
	if chunks[0].Delta != "partial" || chunks[0].Err != nil || chunks[0].Done {
		t.Fatalf("expected partial content before the error, got %+v", chunks[0])
	}

	terminal := chunks[1]
	if !terminal.Done {
		t.Fatalf("expected terminal chunk Done=true, got %+v", terminal)
	}
	if terminal.FinishReason != "" {
		t.Fatalf("expected error chunk to carry no finish reason, got %q", terminal.FinishReason)
	}
	if terminal.Err == nil {
		t.Fatal("expected terminal chunk to carry the provider error, got nil")
	}

	var streamErr *llmhub.StreamError
	if !errors.As(terminal.Err, &streamErr) {
		t.Fatalf("expected *llmhub.StreamError, got %T: %v", terminal.Err, terminal.Err)
	}
	if streamErr.Provider != "test-provider" {
		t.Fatalf("expected provider test-provider, got %q", streamErr.Provider)
	}
	if streamErr.Message != "upstream overloaded" {
		t.Fatalf("expected message upstream overloaded, got %q", streamErr.Message)
	}
	if streamErr.Type != "server_error" {
		t.Fatalf("expected type server_error, got %q", streamErr.Type)
	}
	if streamErr.Code != "overloaded" {
		t.Fatalf("expected code overloaded, got %q", streamErr.Code)
	}
	if streamErr.Param != "model" {
		t.Fatalf("expected param model, got %q", streamErr.Param)
	}
	wantText := "test-provider: upstream overloaded (type=server_error, code=overloaded)"
	if streamErr.Error() != wantText {
		t.Fatalf("expected error text %q, got %q", wantText, streamErr.Error())
	}
}

func TestClientStream_ErrorEventName(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, ok := w.(http.Flusher)
		if !ok {
			http.Error(w, "response writer does not flush", http.StatusInternalServerError)
			return
		}

		// Some routers send an `event: error` frame whose data is the error
		// object itself, with no top-level `error` wrapper.
		fmt.Fprintf(w, "event: error\ndata: {\"message\":\"bad gateway\",\"type\":\"server_error\",\"code\":\"bad_gateway\"}\n\n")
		flusher.Flush()
	}))
	defer server.Close()

	chunksCh, err := newStreamTestClient(server, "test-provider").Stream(context.Background(), []*llmhub.Message{
		llmhub.NewUserMessage(llmhub.Text("Hi")),
	})
	if err != nil {
		t.Fatalf("Stream failed: %v", err)
	}

	chunks := collectStream(t, chunksCh)
	if len(chunks) != 1 {
		t.Fatalf("expected 1 terminal chunk, got %d: %+v", len(chunks), chunks)
	}
	if !chunks[0].Done || chunks[0].Err == nil {
		t.Fatalf("expected Done=true with Err set, got %+v", chunks[0])
	}
	var streamErr *llmhub.StreamError
	if !errors.As(chunks[0].Err, &streamErr) {
		t.Fatalf("expected *llmhub.StreamError, got %T: %v", chunks[0].Err, chunks[0].Err)
	}
	if streamErr.Provider != "test-provider" || streamErr.Message != "bad gateway" {
		t.Fatalf("unexpected error payload: %+v", streamErr)
	}
	if streamErr.Type != "server_error" || streamErr.Code != "bad_gateway" {
		t.Fatalf("expected type/code to survive, got %+v", streamErr)
	}
}

func TestClientStream_ErrorEventNameFallbackToRawData(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, _ := w.(http.Flusher)
		fmt.Fprintf(w, "event: error\ndata: upstream unavailable\n\n")
		flusher.Flush()
	}))
	defer server.Close()

	chunksCh, err := newStreamTestClient(server, "test-provider").Stream(context.Background(), []*llmhub.Message{
		llmhub.NewUserMessage(llmhub.Text("Hi")),
	})
	if err != nil {
		t.Fatalf("Stream failed: %v", err)
	}

	chunks := collectStream(t, chunksCh)
	if len(chunks) != 1 || !chunks[0].Done || chunks[0].Err == nil {
		t.Fatalf("expected one terminal error chunk, got %+v", chunks)
	}
	var streamErr *llmhub.StreamError
	if !errors.As(chunks[0].Err, &streamErr) {
		t.Fatalf("expected *llmhub.StreamError, got %T: %v", chunks[0].Err, chunks[0].Err)
	}
	if streamErr.Message != "upstream unavailable" {
		t.Fatalf("expected raw frame data as message, got %q", streamErr.Message)
	}
}

func TestClientStream_ErrorFrameWithUsage(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, _ := w.(http.Flusher)
		fmt.Fprintf(w, "data: {\"error\":{\"message\":\"model failed\",\"type\":\"server_error\"},\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":0,\"total_tokens\":5}}\n\n")
		flusher.Flush()
		// An error still wins even if the body later completes normally.
		fmt.Fprintf(w, "data: [DONE]\n\n")
		flusher.Flush()
	}))
	defer server.Close()

	chunksCh, err := newStreamTestClient(server, "test-provider").Stream(context.Background(), []*llmhub.Message{
		llmhub.NewUserMessage(llmhub.Text("Hi")),
	})
	if err != nil {
		t.Fatalf("Stream failed: %v", err)
	}

	chunks := collectStream(t, chunksCh)
	if len(chunks) != 1 {
		t.Fatalf("expected a single terminal error chunk, got %d: %+v", len(chunks), chunks)
	}
	if !chunks[0].Done || chunks[0].Err == nil {
		t.Fatalf("expected Done=true with Err set, got %+v", chunks[0])
	}
	// Usage on the error frame is intentionally not retained: the error wins
	// and the frame is not treated as a completion delta.
	if chunks[0].Usage != nil {
		t.Fatalf("expected no usage on the error chunk, got %+v", chunks[0].Usage)
	}
}

func TestClientStream_ErrorCodeNumeric(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, _ := w.(http.Flusher)
		fmt.Fprintf(w, "data: {\"error\":{\"message\":\"rate limited\",\"code\":429}}\n\n")
		flusher.Flush()
	}))
	defer server.Close()

	chunksCh, err := newStreamTestClient(server, "test-provider").Stream(context.Background(), []*llmhub.Message{
		llmhub.NewUserMessage(llmhub.Text("Hi")),
	})
	if err != nil {
		t.Fatalf("Stream failed: %v", err)
	}

	chunks := collectStream(t, chunksCh)
	if len(chunks) != 1 || chunks[0].Err == nil {
		t.Fatalf("expected one terminal error chunk, got %+v", chunks)
	}
	var streamErr *llmhub.StreamError
	if !errors.As(chunks[0].Err, &streamErr) {
		t.Fatalf("expected *llmhub.StreamError, got %T: %v", chunks[0].Err, chunks[0].Err)
	}
	if streamErr.Code != "429" {
		t.Fatalf("expected numeric code normalized to %q, got %q", "429", streamErr.Code)
	}
}

func TestClientStream_ErrorProviderLaneParity(t *testing.T) {
	// The four OpenAI-compatible lanes share this client; each must report
	// itself as the failing provider.
	for _, lane := range []string{"openai", "openrouter", "xai", "zai"} {
		t.Run(lane, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "text/event-stream")
				flusher, _ := w.(http.Flusher)
				fmt.Fprintf(w, "data: {\"error\":{\"message\":\"upstream failed\",\"type\":\"server_error\"}}\n\n")
				flusher.Flush()
			}))
			defer server.Close()

			chunksCh, err := newStreamTestClient(server, lane).Stream(context.Background(), []*llmhub.Message{
				llmhub.NewUserMessage(llmhub.Text("Hi")),
			})
			if err != nil {
				t.Fatalf("Stream failed: %v", err)
			}
			chunks := collectStream(t, chunksCh)
			if len(chunks) != 1 || chunks[0].Err == nil {
				t.Fatalf("expected one terminal error chunk, got %+v", chunks)
			}
			var streamErr *llmhub.StreamError
			if !errors.As(chunks[0].Err, &streamErr) {
				t.Fatalf("expected *llmhub.StreamError, got %T: %v", chunks[0].Err, chunks[0].Err)
			}
			if streamErr.Provider != lane {
				t.Fatalf("expected Provider %q, got %q", lane, streamErr.Provider)
			}
		})
	}
}

func TestClientStream_CleanStreamUnaffectedByErrorHandling(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, _ := w.(http.Flusher)
		fmt.Fprintf(w, "data: {\"id\":\"x\",\"choices\":[{\"delta\":{\"content\":\"Hello\"}}]}\n\n")
		flusher.Flush()
		fmt.Fprintf(w, "data: {\"id\":\"x\",\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":1,\"completion_tokens\":2,\"total_tokens\":3}}\n\n")
		flusher.Flush()
		fmt.Fprintf(w, "data: [DONE]\n\n")
		flusher.Flush()
	}))
	defer server.Close()

	chunksCh, err := newStreamTestClient(server, "test-provider").Stream(context.Background(), []*llmhub.Message{
		llmhub.NewUserMessage(llmhub.Text("Hi")),
	})
	if err != nil {
		t.Fatalf("Stream failed: %v", err)
	}

	chunks := collectStream(t, chunksCh)
	if len(chunks) != 3 {
		t.Fatalf("expected 3 chunks, got %d: %+v", len(chunks), chunks)
	}
	if chunks[0].Delta != "Hello" {
		t.Fatalf("unexpected first chunk: %+v", chunks[0])
	}
	if chunks[1].FinishReason != "stop" || chunks[1].Usage == nil || chunks[1].Usage.TotalTokens != 3 {
		t.Fatalf("unexpected finish/usage chunk: %+v", chunks[1])
	}
	if !chunks[2].Done || chunks[2].Err != nil {
		t.Fatalf("expected clean terminal chunk, got %+v", chunks[2])
	}
}

// newReasoningDetailsClient points the shared client at an httptest server so
// the reasoning_details fixtures below stay focused on the wire shape.
func newReasoningDetailsClient(server *httptest.Server) *Client {
	return NewClient(ClientConfig{
		ProviderName: "test-provider",
		BaseConfig: llmhub.NewConfig(
			llmhub.WithBaseURL(server.URL),
			llmhub.WithHTTPClient(server.Client()),
			llmhub.WithAPIKey("test-key"),
			llmhub.WithModel("test-model"),
		),
	})
}

func TestClientGenerate_ReasoningDetailsSummary(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		io.WriteString(w, `{
			"id": "chatcmpl-rd",
			"choices": [{
				"message": {
					"role": "assistant",
					"content": "42",
					"reasoning_details": [{
						"type": "reasoning.summary",
						"summary": "The ball costs five cents.",
						"id": "rs_1",
						"format": "openai-responses-v1",
						"vendor_extra": true
					}]
				}
			}],
			"usage": {}
		}`)
	}))
	defer server.Close()

	resp, err := newReasoningDetailsClient(server).Generate(context.Background(), []*llmhub.Message{
		llmhub.NewUserMessage(llmhub.Text("bat and ball")),
	})
	if err != nil {
		t.Fatalf("Generate failed: %v", err)
	}
	if len(resp.ReasoningDetails) != 1 {
		t.Fatalf("expected 1 reasoning detail, got %d: %+v", len(resp.ReasoningDetails), resp.ReasoningDetails)
	}
	detail := resp.ReasoningDetails[0]
	if detail.Type != "reasoning.summary" {
		t.Errorf("expected type reasoning.summary, got %q", detail.Type)
	}
	if detail.Text != "The ball costs five cents." {
		t.Errorf("expected summary text, got %q", detail.Text)
	}
	if detail.ID != "rs_1" || detail.Format != "openai-responses-v1" {
		t.Errorf("expected id/format preserved, got %+v", detail)
	}
	if len(detail.Raw) == 0 {
		t.Fatal("expected Raw to preserve the original detail JSON")
	}
	var raw map[string]any
	if err := json.Unmarshal(detail.Raw, &raw); err != nil {
		t.Fatalf("detail.Raw is not valid JSON: %v", err)
	}
	if raw["vendor_extra"] != true {
		t.Errorf("expected unknown field to survive in Raw, got %v", raw["vendor_extra"])
	}
	if resp.ReasoningText() != "The ball costs five cents." {
		t.Errorf("expected details-only text via ReasoningText(), got %q", resp.ReasoningText())
	}
	if resp.Text() != "42" {
		t.Errorf("expected answer text 42, got %q", resp.Text())
	}
}

func TestClientGenerate_ReasoningDetailsFlatWins(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		io.WriteString(w, `{
			"id": "chatcmpl-rd-flat",
			"choices": [{
				"message": {
					"role": "assistant",
					"content": "42",
					"reasoning_content": "F",
					"reasoning_details": [{"type": "reasoning.summary", "summary": "S"}]
				}
			}],
			"usage": {}
		}`)
	}))
	defer server.Close()

	resp, err := newReasoningDetailsClient(server).Generate(context.Background(), []*llmhub.Message{
		llmhub.NewUserMessage(llmhub.Text("bat and ball")),
	})
	if err != nil {
		t.Fatalf("Generate failed: %v", err)
	}
	if resp.ReasoningText() != "F" {
		t.Fatalf("expected flat reasoning to win without duplication, got %q", resp.ReasoningText())
	}
	if len(resp.ReasoningDetails) != 1 || resp.ReasoningDetails[0].Text != "S" {
		t.Fatalf("expected summary detail to remain available, got %+v", resp.ReasoningDetails)
	}
}

func TestClientGenerate_ReasoningDetailsEncrypted(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		io.WriteString(w, `{
			"id": "chatcmpl-rd-enc",
			"choices": [{
				"message": {
					"role": "assistant",
					"content": "42",
					"reasoning_details": [{
						"type": "reasoning.encrypted",
						"data": "ciphertext-blob",
						"id": "enc_1",
						"format": "openai-responses-v1"
					}]
				}
			}],
			"usage": {}
		}`)
	}))
	defer server.Close()

	resp, err := newReasoningDetailsClient(server).Generate(context.Background(), []*llmhub.Message{
		llmhub.NewUserMessage(llmhub.Text("bat and ball")),
	})
	if err != nil {
		t.Fatalf("Generate failed: %v", err)
	}
	if resp.ReasoningText() != "" {
		t.Fatalf("expected empty flat reasoning for encrypted-only details, got %q", resp.ReasoningText())
	}
	if len(resp.ReasoningDetails) != 1 {
		t.Fatalf("expected 1 reasoning detail, got %+v", resp.ReasoningDetails)
	}
	detail := resp.ReasoningDetails[0]
	if detail.Type != "reasoning.encrypted" {
		t.Errorf("expected type reasoning.encrypted, got %q", detail.Type)
	}
	if detail.Data != "ciphertext-blob" {
		t.Errorf("expected encrypted data preserved, got %q", detail.Data)
	}
	if detail.Text != "" {
		t.Errorf("expected no readable text, got %q", detail.Text)
	}
}

func TestClientGenerate_ReasoningDetailsUnknownTypePreservesRaw(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		io.WriteString(w, `{
			"id": "chatcmpl-rd-unknown",
			"choices": [{
				"message": {
					"role": "assistant",
					"content": "ok",
					"reasoning_details": [{
						"type": "vendor.reasoning",
						"text": "opaque",
						"vendor_nested": {"step": 1}
					}]
				}
			}],
			"usage": {}
		}`)
	}))
	defer server.Close()

	resp, err := newReasoningDetailsClient(server).Generate(context.Background(), []*llmhub.Message{
		llmhub.NewUserMessage(llmhub.Text("hi")),
	})
	if err != nil {
		t.Fatalf("Generate failed: %v", err)
	}
	if len(resp.ReasoningDetails) != 1 {
		t.Fatalf("expected unknown type to yield a detail, got %+v", resp.ReasoningDetails)
	}
	detail := resp.ReasoningDetails[0]
	if detail.Type != "vendor.reasoning" || detail.Text != "opaque" {
		t.Errorf("expected unknown type and text preserved, got %+v", detail)
	}
	var raw map[string]any
	if err := json.Unmarshal(detail.Raw, &raw); err != nil {
		t.Fatalf("detail.Raw is not valid JSON: %v", err)
	}
	if _, ok := raw["vendor_nested"]; !ok {
		t.Errorf("expected unknown nested field in Raw, got %v", raw)
	}
}

func TestClientStream_ReasoningDetails(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, _ := w.(http.Flusher)

		fmt.Fprintf(w, "data: {\"id\":\"s1\",\"choices\":[{\"delta\":{\"reasoning_details\":[{\"type\":\"reasoning.summary\",\"summary\":\"step one\",\"id\":\"rs_1\"}]}}]}\n\n")
		flusher.Flush()
		fmt.Fprintf(w, "data: {\"id\":\"s1\",\"choices\":[{\"delta\":{\"reasoning_details\":[{\"type\":\"reasoning.text\",\"text\":\"step two\",\"signature\":\"sig-1\"}]}}]}\n\n")
		flusher.Flush()
		fmt.Fprintf(w, "data: {\"id\":\"s1\",\"choices\":[{\"delta\":{\"content\":\"answer\"}}]}\n\n")
		flusher.Flush()
		fmt.Fprintf(w, "data: [DONE]\n\n")
		flusher.Flush()
	}))
	defer server.Close()

	chunksCh, err := newStreamTestClient(server, "test-provider").Stream(context.Background(), []*llmhub.Message{
		llmhub.NewUserMessage(llmhub.Text("Hi")),
	})
	if err != nil {
		t.Fatalf("Stream failed: %v", err)
	}

	chunks := collectStream(t, chunksCh)
	if len(chunks) != 4 {
		t.Fatalf("expected 4 chunks, got %d: %+v", len(chunks), chunks)
	}
	if len(chunks[0].ReasoningDetails) != 1 {
		t.Fatalf("expected first chunk details, got %+v", chunks[0])
	}
	first := chunks[0].ReasoningDetails[0]
	if first.Type != "reasoning.summary" || first.Text != "step one" || first.ID != "rs_1" {
		t.Errorf("unexpected first detail: %+v", first)
	}
	if chunks[0].ReasoningDelta != "step one" {
		t.Errorf("expected summary fallback delta, got %q", chunks[0].ReasoningDelta)
	}

	if len(chunks[1].ReasoningDetails) != 1 {
		t.Fatalf("expected only this chunk's detail (no accumulation), got %+v", chunks[1].ReasoningDetails)
	}
	second := chunks[1].ReasoningDetails[0]
	if second.Type != "reasoning.text" || second.Text != "step two" || second.Signature != "sig-1" {
		t.Errorf("unexpected second detail: %+v", second)
	}
	if chunks[1].ReasoningDelta != "step two" {
		t.Errorf("expected text fallback delta, got %q", chunks[1].ReasoningDelta)
	}

	if chunks[2].Delta != "answer" || len(chunks[2].ReasoningDetails) != 0 || chunks[2].ReasoningDelta != "" {
		t.Errorf("unexpected answer chunk: %+v", chunks[2])
	}
	if !chunks[3].Done {
		t.Errorf("expected terminal Done chunk, got %+v", chunks[3])
	}
}

func TestClientStream_ReasoningDetailsFlatWins(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, _ := w.(http.Flusher)

		fmt.Fprintf(w, "data: {\"id\":\"s1\",\"choices\":[{\"delta\":{\"reasoning\":\"F\",\"reasoning_details\":[{\"type\":\"reasoning.summary\",\"summary\":\"S\"}]}}]}\n\n")
		flusher.Flush()
		fmt.Fprintf(w, "data: [DONE]\n\n")
		flusher.Flush()
	}))
	defer server.Close()

	chunksCh, err := newStreamTestClient(server, "test-provider").Stream(context.Background(), []*llmhub.Message{
		llmhub.NewUserMessage(llmhub.Text("Hi")),
	})
	if err != nil {
		t.Fatalf("Stream failed: %v", err)
	}

	chunks := collectStream(t, chunksCh)
	if len(chunks) != 2 {
		t.Fatalf("expected 2 chunks, got %d: %+v", len(chunks), chunks)
	}
	if chunks[0].ReasoningDelta != "F" {
		t.Fatalf("expected flat reasoning delta to win, got %q", chunks[0].ReasoningDelta)
	}
	if len(chunks[0].ReasoningDetails) != 1 || chunks[0].ReasoningDetails[0].Text != "S" {
		t.Fatalf("expected details to remain available, got %+v", chunks[0].ReasoningDetails)
	}
}

func TestClientStream_ReasoningDetailsEncryptedOnly(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, _ := w.(http.Flusher)

		fmt.Fprintf(w, "data: {\"id\":\"s1\",\"choices\":[{\"delta\":{\"reasoning_details\":[{\"type\":\"reasoning.encrypted\",\"data\":\"ciphertext-blob\"}]}}]}\n\n")
		flusher.Flush()
		fmt.Fprintf(w, "data: [DONE]\n\n")
		flusher.Flush()
	}))
	defer server.Close()

	chunksCh, err := newStreamTestClient(server, "test-provider").Stream(context.Background(), []*llmhub.Message{
		llmhub.NewUserMessage(llmhub.Text("Hi")),
	})
	if err != nil {
		t.Fatalf("Stream failed: %v", err)
	}

	chunks := collectStream(t, chunksCh)
	if len(chunks) != 2 {
		t.Fatalf("expected encrypted details chunk plus Done, got %d: %+v", len(chunks), chunks)
	}
	if chunks[0].ReasoningDelta != "" {
		t.Errorf("expected no flat reasoning for encrypted details, got %q", chunks[0].ReasoningDelta)
	}
	if len(chunks[0].ReasoningDetails) != 1 {
		t.Fatalf("expected encrypted detail to survive the skip filter, got %+v", chunks[0])
	}
	detail := chunks[0].ReasoningDetails[0]
	if detail.Type != "reasoning.encrypted" || detail.Data != "ciphertext-blob" {
		t.Errorf("unexpected encrypted detail: %+v", detail)
	}
}

func TestClientStream_ReasoningDetailsMalformed(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, _ := w.(http.Flusher)

		fmt.Fprintf(w, "data: {\"id\":\"s1\",\"choices\":[{\"delta\":{\"reasoning_details\":[123]}}]}\n\n")
		flusher.Flush()
		fmt.Fprintf(w, "data: [DONE]\n\n")
		flusher.Flush()
	}))
	defer server.Close()

	chunksCh, err := newStreamTestClient(server, "test-provider").Stream(context.Background(), []*llmhub.Message{
		llmhub.NewUserMessage(llmhub.Text("Hi")),
	})
	if err != nil {
		t.Fatalf("Stream failed: %v", err)
	}

	chunks := collectStream(t, chunksCh)
	if len(chunks) != 1 {
		t.Fatalf("expected a single terminal error chunk, got %d: %+v", len(chunks), chunks)
	}
	if !chunks[0].Done || chunks[0].Err == nil {
		t.Fatalf("expected Done=true with Err set, got %+v", chunks[0])
	}
	if !strings.Contains(chunks[0].Err.Error(), "reasoning_details") {
		t.Fatalf("expected a reasoning_details decoding error, got %v", chunks[0].Err)
	}
}

func TestClientGenerate_ReasoningDetailsMalformed(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		io.WriteString(w, `{
			"id": "chatcmpl-rd-bad",
			"choices": [{
				"message": {
					"role": "assistant",
					"content": "ok",
					"reasoning_details": [123]
				}
			}],
			"usage": {}
		}`)
	}))
	defer server.Close()

	_, err := newReasoningDetailsClient(server).Generate(context.Background(), []*llmhub.Message{
		llmhub.NewUserMessage(llmhub.Text("Hi")),
	})
	if err == nil {
		t.Fatal("expected Generate to surface malformed reasoning_details")
	}
	if !strings.Contains(err.Error(), "reasoning_details") {
		t.Fatalf("expected a reasoning_details decoding error, got %v", err)
	}
}

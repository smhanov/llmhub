package gemini

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

func TestGeminiGenerateThinkingLevel(t *testing.T) {
	cases := []struct {
		name    string
		level   llmhub.ThinkingLevel
		wantNil bool
		want    string
	}{
		{name: "high", level: llmhub.ThinkingLevelHigh, want: "high"},
		{name: "xhigh clamps to high", level: llmhub.ThinkingLevelXHigh, want: "high"},
		{name: "low", level: llmhub.ThinkingLevelLow, want: "low"},
		{name: "minimal passes through", level: llmhub.ThinkingLevelMinimal, want: "minimal"},
		{name: "unset omits thinkingConfig", wantNil: true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				defer r.Body.Close()
				body, _ := io.ReadAll(r.Body)
				var req geminiRequest
				if err := json.Unmarshal(body, &req); err != nil {
					t.Fatalf("decode request: %v", err)
				}
				if tc.wantNil {
					if req.GenerationConfig != nil && req.GenerationConfig.ThinkingConfig != nil {
						t.Fatalf("thinkingConfig must be absent when ThinkingLevel is unset, got %+v", req.GenerationConfig)
					}
				} else if req.GenerationConfig == nil || req.GenerationConfig.ThinkingConfig == nil ||
					req.GenerationConfig.ThinkingConfig.ThinkingLevel != tc.want {
					t.Fatalf("expected thinkingLevel %q, got %+v", tc.want, req.GenerationConfig)
				}
				io.WriteString(w, `{"candidates":[{"content":{"parts":[{"text":"pong"}]}}],"usageMetadata":{}}`)
			}))
			defer server.Close()

			var opts []llmhub.Option
			opts = append(opts, llmhub.WithBaseURL(server.URL), llmhub.WithModel("flash-test"))
			if tc.level != "" {
				opts = append(opts, llmhub.WithThinkingLevel(tc.level))
			}
			provider, err := New("secret", opts...)
			if err != nil {
				t.Fatalf("provider new: %v", err)
			}
			if _, err := provider.Generate(context.Background(), []*llmhub.Message{llmhub.NewUserMessage(llmhub.Text("ping"))}); err != nil {
				t.Fatalf("generate: %v", err)
			}
		})
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

func TestGeminiStreamHTTPErrorIsSynchronous(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		io.WriteString(w, `{"error":{"message":"bad request"}}`)
	}))
	defer server.Close()

	provider, err := New("secret", llmhub.WithBaseURL(server.URL))
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

func TestGeminiStreamFinishReason(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprintln(w, `{"candidates":[{"content":{"parts":[{"text":"done"}]},"finishReason":"MAX_TOKENS"}]}`)
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
	var chunks []llmhub.StreamChunk
	for chunk := range stream {
		chunks = append(chunks, chunk)
	}
	if len(chunks) == 0 || chunks[0].FinishReason != "length" {
		t.Fatalf("expected length finish reason (MAX_TOKENS), got %+v", chunks)
	}
	if chunks[0].Delta != "done" {
		t.Fatalf("expected delta 'done', got %+v", chunks[0])
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

func TestGeminiStreamErrorFrame(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprintln(w, `{"candidates":[{"content":{"parts":[{"text":"partial"}]}}]}`)
		// Error-only frame: before the fix this decoded as an empty
		// geminiResponse and was dropped, closing the stream cleanly.
		fmt.Fprintln(w, `{"error":{"code":429,"message":"quota exceeded","status":"RESOURCE_EXHAUSTED"}}`)
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

	var chunks []llmhub.StreamChunk
	for chunk := range stream {
		chunks = append(chunks, chunk)
	}
	if len(chunks) != 2 {
		t.Fatalf("expected partial content plus terminal error, got %d: %+v", len(chunks), chunks)
	}
	if chunks[0].Delta != "partial" || chunks[0].Err != nil {
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
	if streamErr.Provider != "gemini" {
		t.Fatalf("expected provider gemini, got %q", streamErr.Provider)
	}
	if streamErr.Message != "quota exceeded" {
		t.Fatalf("expected message quota exceeded, got %q", streamErr.Message)
	}
	if streamErr.Type != "RESOURCE_EXHAUSTED" {
		t.Fatalf("expected status mapped to type, got %q", streamErr.Type)
	}
	if streamErr.Code != "429" {
		t.Fatalf("expected numeric code normalized to %q, got %q", "429", streamErr.Code)
	}
}

func TestBuildRequestBody_ToolChoiceOmittedWithoutTools(t *testing.T) {
	prompt := []*llmhub.Message{
		llmhub.NewUserMessage(llmhub.Text("hi")),
	}

	// Case 1: tool choice without tools - no toolConfig key on the wire.
	cfg := llmhub.NewConfig(
		llmhub.WithModel("flash-test"),
		llmhub.WithToolChoice(llmhub.AutoToolChoice()),
	)
	body, err := buildRequestBody(prompt, cfg)
	if err != nil {
		t.Fatalf("buildRequestBody failed: %v", err)
	}
	var decoded map[string]json.RawMessage
	if err := json.Unmarshal(body, &decoded); err != nil {
		t.Fatalf("unmarshal body failed: %v", err)
	}
	if _, ok := decoded["toolConfig"]; ok {
		t.Fatalf("toolConfig must be absent without tools, got %s", string(decoded["toolConfig"]))
	}

	// Case 2: with tools - rendered with the same value as before.
	tool := llmhub.NewTool("weather", "Get weather", map[string]interface{}{"type": "object"})
	cfg = llmhub.NewConfig(
		llmhub.WithModel("flash-test"),
		llmhub.WithTools(tool),
		llmhub.WithToolChoice(llmhub.RequiredToolChoice()),
	)
	body, err = buildRequestBody(prompt, cfg)
	if err != nil {
		t.Fatalf("buildRequestBody failed: %v", err)
	}
	decoded = nil
	if err := json.Unmarshal(body, &decoded); err != nil {
		t.Fatalf("unmarshal body failed: %v", err)
	}
	var config struct {
		FunctionCallingConfig struct {
			Mode string `json:"mode"`
		} `json:"functionCallingConfig"`
	}
	if err := json.Unmarshal(decoded["toolConfig"], &config); err != nil {
		t.Fatalf("unmarshal toolConfig failed: %v", err)
	}
	if config.FunctionCallingConfig.Mode != "ANY" {
		t.Fatalf("expected toolConfig mode 'ANY', got %q", config.FunctionCallingConfig.Mode)
	}

	// Case 3: web search alone adds a googleSearch tool but no
	// functionDeclarations, so the control must still be omitted.
	cfg = llmhub.NewConfig(
		llmhub.WithModel("flash-test"),
		llmhub.WithWebSearch(true),
		llmhub.WithToolChoice(llmhub.AutoToolChoice()),
	)
	body, err = buildRequestBody(prompt, cfg)
	if err != nil {
		t.Fatalf("buildRequestBody failed: %v", err)
	}
	decoded = nil
	if err := json.Unmarshal(body, &decoded); err != nil {
		t.Fatalf("unmarshal body failed: %v", err)
	}
	if _, ok := decoded["tools"]; !ok {
		t.Fatal("expected googleSearch tool on the wire")
	}
	if _, ok := decoded["toolConfig"]; ok {
		t.Fatalf("toolConfig must be absent with googleSearch only, got %s", string(decoded["toolConfig"]))
	}
}

func TestGeminiGenerateUpstreamInfo(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("x-goog-request-id", "req_gemini_1")
		w.Header().Set("x-upstream-served-by", "gemini-node-3")
		io.WriteString(w, `{
			"modelVersion": "gemini-resolved",
			"candidates": [{"content": {"parts": [{"text": "pong"}]}}],
			"usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 2, "totalTokenCount": 3}
		}`)
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
		t.Fatalf("Text() = %q, want pong", resp.Text())
	}
	if resp.Upstream == nil {
		t.Fatal("Upstream is nil")
	}
	if resp.Upstream.StatusCode != http.StatusOK {
		t.Fatalf("StatusCode = %d, want 200", resp.Upstream.StatusCode)
	}
	if got := resp.Upstream.Headers.Get("x-goog-request-id"); got != "req_gemini_1" {
		t.Fatalf("x-goog-request-id = %q, want req_gemini_1", got)
	}
	if got := resp.Upstream.Headers.Get("x-upstream-served-by"); got != "gemini-node-3" {
		t.Fatalf("x-upstream-served-by = %q, want gemini-node-3", got)
	}
	if resp.Upstream.Model != "gemini-resolved" {
		t.Fatalf("Model = %q, want gemini-resolved", resp.Upstream.Model)
	}
	if resp.Upstream.Provider != "" || resp.Upstream.SystemFingerprint != "" {
		t.Fatalf("unexpected identity fields: %+v", resp.Upstream)
	}
}

func TestGeminiStreamUpstreamInfo(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("x-goog-request-id", "req_gemini_stream")
		fmt.Fprintln(w, `{"modelVersion":"gemini-resolved","candidates":[{"content":{"parts":[{"text":"foo"}]}}]}`)
		fmt.Fprintln(w, `{"candidates":[{"content":{"parts":[{"text":"bar"}]}}]}`)
	}))
	defer server.Close()

	provider, err := New("secret", llmhub.WithBaseURL(server.URL), llmhub.WithModel("flash-test"))
	if err != nil {
		t.Fatalf("provider new: %v", err)
	}
	stream, err := provider.Stream(context.Background(), []*llmhub.Message{llmhub.NewUserMessage(llmhub.Text("hi"))})
	if err != nil {
		t.Fatalf("stream: %v", err)
	}

	var chunks []llmhub.StreamChunk
	var text strings.Builder
	for chunk := range stream {
		chunks = append(chunks, chunk)
		text.WriteString(chunk.Delta)
	}
	if text.String() != "foobar" {
		t.Fatalf("streamed text = %q, want foobar", text.String())
	}
	if len(chunks) != 3 {
		t.Fatalf("expected 3 chunks, got %d: %+v", len(chunks), chunks)
	}
	for i, chunk := range chunks {
		if chunk.Upstream == nil {
			t.Fatalf("chunk %d has nil Upstream", i)
		}
		if chunk.Upstream.StatusCode != http.StatusOK {
			t.Fatalf("chunk %d StatusCode = %d, want 200", i, chunk.Upstream.StatusCode)
		}
		if got := chunk.Upstream.Headers.Get("x-goog-request-id"); got != "req_gemini_stream" {
			t.Fatalf("chunk %d x-goog-request-id = %q", i, got)
		}
		if chunk.Upstream.Model != "gemini-resolved" {
			t.Fatalf("chunk %d Model = %q, want gemini-resolved", i, chunk.Upstream.Model)
		}
	}
	if !chunks[len(chunks)-1].Done {
		t.Fatal("terminal chunk is not Done")
	}
	if chunks[0].Upstream == chunks[1].Upstream {
		t.Fatal("expected a distinct Upstream snapshot per chunk")
	}
}

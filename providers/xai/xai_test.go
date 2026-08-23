package xai

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
	"github.com/smhanov/llmhub/auth"
	"github.com/smhanov/llmhub/internal/openaichat"
)

type mockInvalidatableTokenSource struct {
	tokens        []string
	tokenIndex    int32
	invalidated   []string
	invalidateErr error
	tokenErr      error
}

func (m *mockInvalidatableTokenSource) Token(ctx context.Context) (*auth.Token, error) {
	if m.tokenErr != nil {
		return nil, m.tokenErr
	}
	idx := atomic.AddInt32(&m.tokenIndex, 1) - 1
	if int(idx) < len(m.tokens) {
		return &auth.Token{
			AccessToken: m.tokens[idx],
			TokenType:   "Bearer",
		}, nil
	}
	return &auth.Token{
		AccessToken: "default-access-token",
		TokenType:   "Bearer",
	}, nil
}

func (m *mockInvalidatableTokenSource) Invalidate(accessToken string) {
	m.invalidated = append(m.invalidated, accessToken)
}

func TestXAIRegistrationAndDefaults(t *testing.T) {
	providers := llmhub.RegisteredProviders()
	found := false
	for _, p := range providers {
		if p == "xai" {
			found = true
			break
		}
	}
	if !found {
		t.Fatal("xai provider is not registered")
	}

	// Missing both credentials
	_, err := llmhub.New("xai", "")
	if err == nil || !errors.Is(err, llmhub.ErrInvalidInput) {
		t.Fatalf("expected ErrInvalidInput when credentials missing, got: %v", err)
	}

	// Positional API key
	client, err := llmhub.New("xai", "test-api-key")
	if err != nil {
		t.Fatalf("new with api key: %v", err)
	}
	if client.ProviderName() != "xai" {
		t.Fatalf("expected provider name 'xai', got %s", client.ProviderName())
	}
}

func TestXAICredentialPrecedence(t *testing.T) {
	var gotAuth string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotAuth = r.Header.Get("Authorization")
		io.WriteString(w, `{"id":"x","choices":[{"message":{"role":"assistant","content":"ok"}}],"usage":{}}`)
	}))
	defer server.Close()

	src := &mockInvalidatableTokenSource{tokens: []string{"oauth-token-val"}}

	// Both API key and TokenSource provided -> TokenSource must win
	client, err := llmhub.New("xai", "api-key-val",
		llmhub.WithBaseURL(server.URL),
		llmhub.WithTokenSource(src),
	)
	if err != nil {
		t.Fatalf("new: %v", err)
	}

	_, err = client.Generate(context.Background(), []*llmhub.Message{llmhub.NewUserMessage(llmhub.Text("hi"))})
	if err != nil {
		t.Fatalf("generate: %v", err)
	}
	if gotAuth != "Bearer oauth-token-val" {
		t.Fatalf("expected TokenSource to take precedence over APIKey, got auth header: %q", gotAuth)
	}
}

func TestXAIGenerateAndStreamWithReasoningAndTools(t *testing.T) {
	tool := llmhub.NewTool("search", "Search web", map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"q": map[string]interface{}{"type": "string"},
		},
	})

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Accept") == "text/event-stream" || strings.Contains(r.Header.Get("Content-Type"), "application/json") {
			// Check if streaming
			body, _ := io.ReadAll(r.Body)
			var req openaichat.CompletionRequest
			_ = json.Unmarshal(body, &req)

			if req.Stream {
				w.Header().Set("Content-Type", "text/event-stream")
				flusher, _ := w.(http.Flusher)
				fmt.Fprintf(w, "data: {\"id\":\"s1\",\"choices\":[{\"delta\":{\"reasoning\":\"thought-1\",\"content\":\"stream-text\"}}]}\n\n")
				flusher.Flush()
				fmt.Fprintf(w, "data: {\"id\":\"s1\",\"choices\":[{\"delta\":{\"tool_calls\":[{\"id\":\"tc-1\",\"type\":\"function\",\"function\":{\"name\":\"search\",\"arguments\":\"{\\\"q\\\":\\\"golang\\\"}\"}}]}}]}\n\n")
				flusher.Flush()
				fmt.Fprintf(w, "data: [DONE]\n\n")
				flusher.Flush()
				return
			}

			// Non-streaming response
			io.WriteString(w, `{
				"id": "gen-1",
				"choices": [{
					"message": {
						"role": "assistant",
						"content": "gen-text",
						"reasoning": "gen-reasoning",
						"tool_calls": [{
							"id": "tc-1",
							"type": "function",
							"function": {"name": "search", "arguments": "{\"q\":\"golang\"}"}
						}]
					}
				}],
				"usage": {
					"prompt_tokens": 10,
					"completion_tokens": 20,
					"total_tokens": 30
				}
			}`)
		}
	}))
	defer server.Close()

	client, err := llmhub.New("xai", "my-key",
		llmhub.WithBaseURL(server.URL),
		llmhub.WithModel("grok-4.6"),
		llmhub.WithTools(tool),
	)
	if err != nil {
		t.Fatalf("new: %v", err)
	}

	// Generate
	resp, err := client.Generate(context.Background(), []*llmhub.Message{llmhub.NewUserMessage(llmhub.Text("search"))})
	if err != nil {
		t.Fatalf("generate: %v", err)
	}
	if resp.Text() != "gen-text" {
		t.Fatalf("text = %q, want gen-text", resp.Text())
	}
	if resp.ReasoningText() != "gen-reasoning" {
		t.Fatalf("reasoning = %q, want gen-reasoning", resp.ReasoningText())
	}
	if len(resp.ToolCalls()) != 1 || resp.ToolCalls()[0].Name != "search" {
		t.Fatalf("tool calls = %+v", resp.ToolCalls())
	}
	if resp.Usage.PromptTokens != 10 || resp.Usage.CompletionTokens != 20 {
		t.Fatalf("usage = %+v", resp.Usage)
	}

	// Stream
	stream, err := client.Stream(context.Background(), []*llmhub.Message{llmhub.NewUserMessage(llmhub.Text("search"))})
	if err != nil {
		t.Fatalf("stream: %v", err)
	}
	var textBuf, reasoningBuf strings.Builder
	var toolCalls []*llmhub.ToolCallContent
	for chunk := range stream {
		if chunk.Err != nil {
			t.Fatalf("stream chunk error: %v", chunk.Err)
		}
		textBuf.WriteString(chunk.Delta)
		reasoningBuf.WriteString(chunk.ReasoningDelta)
		if len(chunk.ToolCalls) > 0 {
			toolCalls = append(toolCalls, chunk.ToolCalls...)
		}
	}
	if textBuf.String() != "stream-text" {
		t.Fatalf("stream text = %q, want stream-text", textBuf.String())
	}
	if reasoningBuf.String() != "thought-1" {
		t.Fatalf("stream reasoning = %q, want thought-1", reasoningBuf.String())
	}
	if len(toolCalls) != 1 || toolCalls[0].Name != "search" {
		t.Fatalf("stream tool calls = %+v", toolCalls)
	}
}

func TestXAI401ReactiveRetry(t *testing.T) {
	var attempts int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		count := atomic.AddInt32(&attempts, 1)
		authHdr := r.Header.Get("Authorization")

		if count == 1 {
			if authHdr != "Bearer token-1" {
				t.Fatalf("attempt 1 auth = %q, want 'Bearer token-1'", authHdr)
			}
			w.WriteHeader(http.StatusUnauthorized)
			io.WriteString(w, `{"error":{"message":"access token expired"}}`)
			return
		}

		if authHdr != "Bearer token-2" {
			t.Fatalf("attempt 2 auth = %q, want 'Bearer token-2'", authHdr)
		}
		io.WriteString(w, `{"id":"x","choices":[{"message":{"role":"assistant","content":"retried ok"}}],"usage":{}}`)
	}))
	defer server.Close()

	src := &mockInvalidatableTokenSource{
		tokens: []string{"token-1", "token-2"},
	}

	client, err := llmhub.New("xai", "",
		llmhub.WithBaseURL(server.URL),
		llmhub.WithTokenSource(src),
	)
	if err != nil {
		t.Fatalf("new: %v", err)
	}

	resp, err := client.Generate(context.Background(), []*llmhub.Message{llmhub.NewUserMessage(llmhub.Text("hi"))})
	if err != nil {
		t.Fatalf("generate: %v", err)
	}
	if resp.Text() != "retried ok" {
		t.Fatalf("expected retried ok, got %s", resp.Text())
	}

	if len(src.invalidated) != 1 || src.invalidated[0] != "token-1" {
		t.Fatalf("expected token-1 to be invalidated, got: %v", src.invalidated)
	}
	if atomic.LoadInt32(&attempts) != 2 {
		t.Fatalf("expected 2 attempts, got %d", atomic.LoadInt32(&attempts))
	}
}

func TestXAISecond401DoesNotLoop(t *testing.T) {
	var attempts int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&attempts, 1)
		w.WriteHeader(http.StatusUnauthorized)
		io.WriteString(w, `{"error":{"message":"unauthorized"}}`)
	}))
	defer server.Close()

	src := &mockInvalidatableTokenSource{
		tokens: []string{"token-1", "token-2", "token-3"},
	}

	client, err := llmhub.New("xai", "",
		llmhub.WithBaseURL(server.URL),
		llmhub.WithTokenSource(src),
	)
	if err != nil {
		t.Fatalf("new: %v", err)
	}

	_, err = client.Generate(context.Background(), []*llmhub.Message{llmhub.NewUserMessage(llmhub.Text("hi"))})
	if err == nil {
		t.Fatal("expected error on persistent 401")
	}
	if atomic.LoadInt32(&attempts) != 2 {
		t.Fatalf("expected exactly 2 attempts before giving up, got %d", atomic.LoadInt32(&attempts))
	}
}

func TestXAIReauthenticationErrorPreserved(t *testing.T) {
	src := &mockInvalidatableTokenSource{
		tokenErr: auth.ErrReauthenticationRequired,
	}

	client, err := llmhub.New("xai", "",
		llmhub.WithTokenSource(src),
	)
	if err != nil {
		t.Fatalf("new: %v", err)
	}

	_, err = client.Generate(context.Background(), []*llmhub.Message{llmhub.NewUserMessage(llmhub.Text("hi"))})
	if err == nil {
		t.Fatal("expected error when token source fails")
	}
	if !errors.Is(err, auth.ErrReauthenticationRequired) {
		t.Fatalf("expected errors.Is(err, auth.ErrReauthenticationRequired) to be true, got %v", err)
	}
}

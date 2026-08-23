package main

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/smhanov/llmhub"
	"github.com/smhanov/llmhub/auth"
)

type exampleTokenStore struct {
	token *auth.Token
}

func (s *exampleTokenStore) Load(context.Context) (*auth.Token, error) {
	return s.token.Clone(), nil
}

func (s *exampleTokenStore) Save(_ context.Context, token *auth.Token) error {
	s.token = token.Clone()
	return nil
}

type exampleRefreshSource struct {
	store       *exampleTokenStore
	calls       int
	invalidated string
}

func (s *exampleRefreshSource) Token(context.Context) (*auth.Token, error) {
	s.calls++
	if s.calls == 1 {
		return s.store.token.Clone(), nil
	}
	s.store.token = &auth.Token{AccessToken: "refreshed", RefreshToken: "rotated"}
	return s.store.token.Clone(), nil
}

func (s *exampleRefreshSource) Invalidate(accessToken string) {
	s.invalidated = accessToken
}

func TestValidateGenerateResponse(t *testing.T) {
	// 1. Exact match
	resp := &llmhub.Response{
		Content: []llmhub.ContentPart{llmhub.Text("LLHUB_XAI_GENERATE_OK\n")},
	}
	if err := validateGenerateResponse(resp, "LLHUB_XAI_GENERATE_OK"); err != nil {
		t.Fatalf("expected valid response, got: %v", err)
	}

	// 2. Mismatch
	badResp := &llmhub.Response{
		Content: []llmhub.ContentPart{llmhub.Text("Different response")},
	}
	if err := validateGenerateResponse(badResp, "LLHUB_XAI_GENERATE_OK"); err == nil {
		t.Fatal("expected error on text mismatch")
	}

	// 3. Nil response
	if err := validateGenerateResponse(nil, "LLHUB_XAI_GENERATE_OK"); err == nil {
		t.Fatal("expected error on nil response")
	}
}

func TestValidateStreamResponse(t *testing.T) {
	// 1. Successful stream
	ch1 := make(chan llmhub.StreamChunk, 4)
	ch1 <- llmhub.StreamChunk{Delta: "LLHUB_"}
	ch1 <- llmhub.StreamChunk{Delta: "XAI_STREAM_"}
	ch1 <- llmhub.StreamChunk{Delta: "OK"}
	ch1 <- llmhub.StreamChunk{Done: true}
	close(ch1)

	if err := validateStreamResponse(ch1, "LLHUB_XAI_STREAM_OK"); err != nil {
		t.Fatalf("expected valid stream response, got: %v", err)
	}

	// 2. Stream chunk error
	ch2 := make(chan llmhub.StreamChunk, 2)
	ch2 <- llmhub.StreamChunk{Delta: "partial"}
	ch2 <- llmhub.StreamChunk{Err: errors.New("network error"), Done: true}
	close(ch2)

	if err := validateStreamResponse(ch2, "LLHUB_XAI_STREAM_OK"); err == nil {
		t.Fatal("expected error on stream chunk error")
	}

	// 3. Stream closed without Done
	ch3 := make(chan llmhub.StreamChunk, 1)
	ch3 <- llmhub.StreamChunk{Delta: "LLHUB_XAI_STREAM_OK"}
	close(ch3)

	if err := validateStreamResponse(ch3, "LLHUB_XAI_STREAM_OK"); err == nil {
		t.Fatal("expected error on stream closed without Done")
	}
}

func TestSanitizeError(t *testing.T) {
	rawErr := errors.New("something went wrong")
	sanitized := sanitizeError(rawErr)
	if sanitized == nil || strings.Contains(sanitized.Error(), "secret") {
		t.Fatalf("unexpected sanitized error: %v", sanitized)
	}
}

func TestPerformVerifyRefreshLoadsThenInvalidates(t *testing.T) {
	store := &exampleTokenStore{token: &auth.Token{AccessToken: "original", RefreshToken: "refresh"}}
	source := &exampleRefreshSource{store: store}

	if err := performVerifyRefresh(context.Background(), store, source); err != nil {
		t.Fatalf("verify refresh: %v", err)
	}
	if source.calls != 2 {
		t.Fatalf("Token calls = %d, want 2", source.calls)
	}
	if source.invalidated != "original" {
		t.Fatalf("invalidated = %q, want original", source.invalidated)
	}
	if store.token.AccessToken != "refreshed" || store.token.RefreshToken != "rotated" {
		t.Fatalf("store token = %+v", store.token)
	}
}

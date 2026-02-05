package llmhub

import (
	"context"
	"errors"
	"strings"
	"testing"
)

func TestClientGenerate(t *testing.T) {
	var captured bool
	stub := &testProvider{
		generateFn: func(ctx context.Context, prompt []*Message, opts ...Option) (*Response, error) {
			captured = true
			if len(prompt) != 1 {
				t.Fatalf("expected prompt length 1")
			}
			cfg := NewConfig(opts...)
			if cfg.Model != "unit-test" {
				t.Fatalf("expected option propagation")
			}
			return &Response{ID: "1", Content: []ContentPart{Text("ok")}}, nil
		},
	}
	client := Wrap(stub, WithModel("unit-test"))
	resp, err := client.Generate(context.Background(), []*Message{NewUserMessage(Text("hi"))})
	if err != nil {
		t.Fatalf("generate failed: %v", err)
	}
	if resp.Text() != "ok" || !captured {
		t.Fatalf("unexpected response: %+v", resp)
	}
}

func TestClientGeneratePropagatesErrors(t *testing.T) {
	stub := &testProvider{
		generateFn: func(ctx context.Context, prompt []*Message, opts ...Option) (*Response, error) {
			return nil, errors.New("boom")
		},
	}
	client := Wrap(stub)
	if _, err := client.Generate(context.Background(), nil); err == nil {
		t.Fatalf("expected error")
	}
}

func TestClientStream(t *testing.T) {
	stub := &testProvider{
		streamFn: func(ctx context.Context, prompt []*Message, opts ...Option) (<-chan StreamChunk, error) {
			ch := make(chan StreamChunk, 2)
			ch <- StreamChunk{Delta: "hello "}
			ch <- StreamChunk{Delta: "world", Done: true}
			close(ch)
			return ch, nil
		},
	}
	client := Wrap(stub)
	stream, err := client.Stream(context.Background(), nil)
	if err != nil {
		t.Fatalf("stream failed: %v", err)
	}
	var combined strings.Builder
	for chunk := range stream {
		if chunk.Err != nil {
			t.Fatalf("chunk error: %v", chunk.Err)
		}
		combined.WriteString(chunk.Delta)
		if chunk.Done {
			break
		}
	}
	if combined.String() != "hello world" {
		t.Fatalf("unexpected stream result: %s", combined.String())
	}
}

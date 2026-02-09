package llmhub

import (
	"context"
	"errors"
	"math"
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

func TestClientGenerateCostCalculation(t *testing.T) {
	stub := &testProvider{
		generateFn: func(ctx context.Context, prompt []*Message, opts ...Option) (*Response, error) {
			return &Response{
				ID:      "cost-test",
				Content: []ContentPart{Text("ok")},
				Usage: UsageMetadata{
					PromptTokens:     1000,
					CompletionTokens: 500,
					TotalTokens:      1500,
				},
			}, nil
		},
	}
	client := Wrap(stub, WithModel("unit-test"), WithCost(2.50, 10.00))
	resp, err := client.Generate(context.Background(), []*Message{NewUserMessage(Text("hi"))})
	if err != nil {
		t.Fatalf("generate failed: %v", err)
	}
	// Expected: (1000 * 2.50 / 1_000_000) + (500 * 10.00 / 1_000_000)
	//         = 0.0025 + 0.005 = 0.0075
	expected := 0.0075
	if math.Abs(resp.Usage.Cost-expected) > 1e-9 {
		t.Fatalf("expected cost %f, got %f", expected, resp.Usage.Cost)
	}
}

func TestClientGenerateCostZeroWhenNotConfigured(t *testing.T) {
	stub := &testProvider{
		generateFn: func(ctx context.Context, prompt []*Message, opts ...Option) (*Response, error) {
			return &Response{
				ID:      "no-cost",
				Content: []ContentPart{Text("ok")},
				Usage: UsageMetadata{
					PromptTokens:     1000,
					CompletionTokens: 500,
					TotalTokens:      1500,
				},
			}, nil
		},
	}
	client := Wrap(stub)
	resp, err := client.Generate(context.Background(), []*Message{NewUserMessage(Text("hi"))})
	if err != nil {
		t.Fatalf("generate failed: %v", err)
	}
	if resp.Usage.Cost != 0 {
		t.Fatalf("expected zero cost when not configured, got %f", resp.Usage.Cost)
	}
}

func TestClientGenerateCostPerRequestOverride(t *testing.T) {
	stub := &testProvider{
		generateFn: func(ctx context.Context, prompt []*Message, opts ...Option) (*Response, error) {
			return &Response{
				ID:      "override",
				Content: []ContentPart{Text("ok")},
				Usage: UsageMetadata{
					PromptTokens:     2000,
					CompletionTokens: 1000,
					TotalTokens:      3000,
				},
			}, nil
		},
	}
	// Client defaults: $5.00/$15.00 per 1M tokens
	client := Wrap(stub, WithCost(5.00, 15.00))
	// Per-request override: $1.00/$3.00 per 1M tokens
	resp, err := client.Generate(context.Background(),
		[]*Message{NewUserMessage(Text("hi"))},
		WithCost(1.00, 3.00),
	)
	if err != nil {
		t.Fatalf("generate failed: %v", err)
	}
	// Expected: (2000 * 1.00 / 1_000_000) + (1000 * 3.00 / 1_000_000)
	//         = 0.002 + 0.003 = 0.005
	expected := 0.005
	if math.Abs(resp.Usage.Cost-expected) > 1e-9 {
		t.Fatalf("expected cost %f, got %f", expected, resp.Usage.Cost)
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

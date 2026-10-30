package llmhub

import (
	"context"
	"errors"
	"net/http"
	"testing"
	"time"
)

// TestClientStreamTiming covers AC1: every chunk carries a Timing snapshot,
// the terminal chunk reports FirstToken, Total >= FirstToken, and Chunks
// counts content chunks.
func TestClientStreamTiming(t *testing.T) {
	stub := &testProvider{
		streamFn: func(ctx context.Context, prompt []*Message, opts ...Option) (<-chan StreamChunk, error) {
			ch := make(chan StreamChunk, 3)
			go func() {
				defer close(ch)
				time.Sleep(5 * time.Millisecond)
				ch <- StreamChunk{Delta: "hello "}
				time.Sleep(10 * time.Millisecond)
				ch <- StreamChunk{Delta: "world"}
				ch <- StreamChunk{Done: true}
			}()
			return ch, nil
		},
	}
	client := Wrap(stub)
	stream, err := client.Stream(context.Background(), nil)
	if err != nil {
		t.Fatalf("stream failed: %v", err)
	}
	var chunks []StreamChunk
	for chunk := range stream {
		if chunk.Err != nil {
			t.Fatalf("chunk error: %v", chunk.Err)
		}
		if chunk.Timing == nil {
			t.Fatalf("chunk %q missing Timing", chunk.Delta)
		}
		chunks = append(chunks, chunk)
		if chunk.Done {
			break
		}
	}
	if len(chunks) != 3 {
		t.Fatalf("got %d chunks, want 3", len(chunks))
	}
	if chunks[0].Timing.Total != 0 {
		t.Errorf("non-terminal chunk carries Total = %v, want 0", chunks[0].Timing.Total)
	}
	if chunks[0].Timing.Chunks != 1 || chunks[1].Timing.Chunks != 2 {
		t.Errorf("chunk counters = %d, %d, want 1, 2", chunks[0].Timing.Chunks, chunks[1].Timing.Chunks)
	}
	final := chunks[2].Timing
	if final.FirstToken <= 0 {
		t.Errorf("FirstToken = %v, want > 0", final.FirstToken)
	}
	if final.Total < final.FirstToken {
		t.Errorf("Total = %v < FirstToken = %v", final.Total, final.FirstToken)
	}
	if final.Chunks != 2 {
		t.Errorf("final Chunks = %d, want 2 (terminal frame is not content)", final.Chunks)
	}
	if final.StartedAt.After(time.Now()) {
		t.Errorf("StartedAt = %v is in the future", final.StartedAt)
	}
}

// TestClientStreamTimingUsageFrameNotAToken covers AC2: a usage-only frame
// does not count as the first token.
func TestClientStreamTimingUsageFrameNotAToken(t *testing.T) {
	stub := &testProvider{
		streamFn: func(ctx context.Context, prompt []*Message, opts ...Option) (<-chan StreamChunk, error) {
			ch := make(chan StreamChunk, 3)
			ch <- StreamChunk{Usage: &UsageMetadata{TotalTokens: 1}}
			ch <- StreamChunk{Delta: "hi"}
			ch <- StreamChunk{Done: true}
			close(ch)
			return ch, nil
		},
	}
	client := Wrap(stub)
	stream, err := client.Stream(context.Background(), nil)
	if err != nil {
		t.Fatalf("stream failed: %v", err)
	}
	var chunks []StreamChunk
	for chunk := range stream {
		chunks = append(chunks, chunk)
		if chunk.Done {
			break
		}
	}
	if len(chunks) != 3 {
		t.Fatalf("got %d chunks, want 3", len(chunks))
	}
	if chunks[0].Timing.FirstToken != 0 || chunks[0].Timing.Chunks != 0 {
		t.Errorf("usage frame counted as content: FirstToken=%v Chunks=%d, want 0, 0",
			chunks[0].Timing.FirstToken, chunks[0].Timing.Chunks)
	}
	if chunks[1].Timing.FirstToken <= 0 || chunks[1].Timing.Chunks != 1 {
		t.Errorf("first content chunk: FirstToken=%v Chunks=%d, want >0, 1",
			chunks[1].Timing.FirstToken, chunks[1].Timing.Chunks)
	}
}

// TestClientStreamTimingToolCallChunkIsContent verifies tool-call deltas
// count as first token (streamed tool calls are content, not accounting).
func TestClientStreamTimingToolCallChunkIsContent(t *testing.T) {
	stub := &testProvider{
		streamFn: func(ctx context.Context, prompt []*Message, opts ...Option) (<-chan StreamChunk, error) {
			ch := make(chan StreamChunk, 2)
			ch <- StreamChunk{ToolCalls: []*ToolCallContent{ToolCall("id1", "get_weather", "{}")}}
			ch <- StreamChunk{Done: true}
			close(ch)
			return ch, nil
		},
	}
	client := Wrap(stub)
	stream, err := client.Stream(context.Background(), nil)
	if err != nil {
		t.Fatalf("stream failed: %v", err)
	}
	var final StreamChunk
	for chunk := range stream {
		final = chunk
	}
	if final.Timing.FirstToken <= 0 || final.Timing.Chunks != 1 {
		t.Errorf("tool-call chunk not counted: FirstToken=%v Chunks=%d", final.Timing.FirstToken, final.Timing.Chunks)
	}
}

// TestClientStreamTimingErrorTerminal covers AC3: a stream ending in an
// error chunk still carries Total on the terminal chunk.
func TestClientStreamTimingErrorTerminal(t *testing.T) {
	stub := &testProvider{
		streamFn: func(ctx context.Context, prompt []*Message, opts ...Option) (<-chan StreamChunk, error) {
			ch := make(chan StreamChunk, 2)
			ch <- StreamChunk{Delta: "partial"}
			ch <- StreamChunk{Err: errors.New("boom"), Done: true}
			close(ch)
			return ch, nil
		},
	}
	client := Wrap(stub)
	stream, err := client.Stream(context.Background(), nil)
	if err != nil {
		t.Fatalf("stream failed: %v", err)
	}
	var terminal StreamChunk
	for chunk := range stream {
		if chunk.Err != nil {
			terminal = chunk
			break
		}
	}
	if terminal.Timing == nil {
		t.Fatal("terminal error chunk missing Timing")
	}
	if terminal.Timing.Total <= 0 {
		t.Errorf("terminal error chunk Total = %v, want > 0", terminal.Timing.Total)
	}
	if terminal.Timing.FirstToken <= 0 || terminal.Timing.Chunks != 1 {
		t.Errorf("FirstToken=%v Chunks=%d, want >0, 1", terminal.Timing.FirstToken, terminal.Timing.Chunks)
	}
}

// TestClientStreamTimingSynthesizesDoneOnBareClose covers the defensive
// path: an upstream that closes without a terminal frame still yields one
// final Done chunk carrying Total.
func TestClientStreamTimingSynthesizesDoneOnBareClose(t *testing.T) {
	stub := &testProvider{
		streamFn: func(ctx context.Context, prompt []*Message, opts ...Option) (<-chan StreamChunk, error) {
			ch := make(chan StreamChunk, 1)
			ch <- StreamChunk{
				Delta: "only",
				Upstream: &UpstreamInfo{
					StatusCode: http.StatusOK,
					Headers:    http.Header{"X-Request-Id": []string{"req-1"}},
				},
			}
			close(ch)
			return ch, nil
		},
	}
	client := Wrap(stub)
	stream, err := client.Stream(context.Background(), nil)
	if err != nil {
		t.Fatalf("stream failed: %v", err)
	}
	var chunks []StreamChunk
	for chunk := range stream {
		chunks = append(chunks, chunk)
	}
	if len(chunks) != 2 {
		t.Fatalf("got %d chunks, want 2 (content + synthesized Done)", len(chunks))
	}
	if !chunks[1].Done || chunks[1].Err != nil {
		t.Fatalf("expected synthesized Done chunk, got %+v", chunks[1])
	}
	if chunks[1].Timing == nil || chunks[1].Timing.Total <= 0 {
		t.Fatalf("synthesized chunk Timing = %+v, want Total > 0", chunks[1].Timing)
	}
	if chunks[1].Timing.Chunks != 1 {
		t.Errorf("synthesized chunk Chunks = %d, want 1 (it carries no content)", chunks[1].Timing.Chunks)
	}
	if chunks[1].Upstream == nil || chunks[1].Upstream.StatusCode != http.StatusOK {
		t.Fatalf("synthesized chunk Upstream = %+v, want last observed upstream snapshot", chunks[1].Upstream)
	}
	if got := chunks[1].Upstream.Headers.Get("X-Request-Id"); got != "req-1" {
		t.Fatalf("synthesized chunk X-Request-Id = %q, want req-1", got)
	}
	if chunks[0].Upstream == chunks[1].Upstream {
		t.Fatal("synthesized chunk should carry its own Upstream snapshot")
	}
}

// TestClientGenerateTiming covers AC5: non-streaming Generate returns
// Response.Timing with Total > 0.
func TestClientGenerateTiming(t *testing.T) {
	stub := &testProvider{
		generateFn: func(ctx context.Context, prompt []*Message, opts ...Option) (*Response, error) {
			time.Sleep(5 * time.Millisecond)
			return &Response{ID: "r1", Content: []ContentPart{Text("ok")}}, nil
		},
	}
	client := Wrap(stub)
	resp, err := client.Generate(context.Background(), nil)
	if err != nil {
		t.Fatalf("generate failed: %v", err)
	}
	if resp.Timing == nil {
		t.Fatal("response missing Timing")
	}
	if resp.Timing.Total <= 0 {
		t.Errorf("Total = %v, want > 0", resp.Timing.Total)
	}
	if resp.Timing.FirstToken != 0 {
		t.Errorf("FirstToken = %v on a non-streaming response, want 0", resp.Timing.FirstToken)
	}
	if resp.Timing.StartedAt.IsZero() {
		t.Error("StartedAt is zero")
	}
}

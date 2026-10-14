package llmhub

import "context"

// Client is the main entry point for interacting with LLM providers via the unified API.
type Client struct {
	provider    Provider
	defaultOpts []Option
}

// Wrap binds an already-constructed provider to a Client.
func Wrap(provider Provider, opts ...Option) *Client {
	return &Client{provider: provider, defaultOpts: append([]Option(nil), opts...)}
}

// ProviderName returns the underlying provider identifier.
func (c *Client) ProviderName() string {
	if c == nil || c.provider == nil {
		return ""
	}
	return c.provider.Name()
}

// Generate performs a single request/response interaction with the provider.
// The returned response carries built-in latency telemetry in Response.Timing.
func (c *Client) Generate(ctx context.Context, prompt []*Message, opts ...Option) (*Response, error) {
	if c == nil || c.provider == nil {
		return nil, &Error{Provider: "", Op: "Generate", Err: ErrInvalidInput}
	}
	timer := newRequestTimer()
	ctx = timer.trace(ctx)
	merged := mergeOptions(c.defaultOpts, opts)
	resp, err := c.provider.Generate(ctx, prompt, merged...)
	if err != nil {
		return nil, &Error{Provider: c.provider.Name(), Op: "Generate", Err: err}
	}
	computeCost(resp, merged)
	resp.Timing = timer.snapshot(true)
	return resp, nil
}

// computeCost populates Response.Usage.Cost based on token counts and configured rates.
// If the provider already reported a cost in the response, that cost is preserved
// and overrides any rates configured via WithCost.
func computeCost(resp *Response, opts []Option) {
	if resp == nil {
		return
	}
	if resp.Usage.Cost != 0 {
		return
	}
	cfg := NewConfig(opts...)
	inputCost := float64(resp.Usage.PromptTokens) * cfg.InputCostPerMillionTokens / 1_000_000
	outputCost := float64(resp.Usage.CompletionTokens) * cfg.OutputCostPerMillionTokens / 1_000_000
	resp.Usage.Cost = inputCost + outputCost
}

// Stream initiates a streaming interaction and returns a read-only channel of
// chunks. Every chunk carries a Timing snapshot; the terminal chunk (Done or
// Err) carries the complete metrics including Total stream duration.
func (c *Client) Stream(ctx context.Context, prompt []*Message, opts ...Option) (<-chan StreamChunk, error) {
	if c == nil || c.provider == nil {
		return nil, &Error{Provider: "", Op: "Stream", Err: ErrInvalidInput}
	}
	timer := newRequestTimer()
	ctx = timer.trace(ctx)
	merged := mergeOptions(c.defaultOpts, opts)
	ch, err := c.provider.Stream(ctx, prompt, merged...)
	if err != nil {
		return nil, &Error{Provider: c.provider.Name(), Op: "Stream", Err: err}
	}
	return instrumentStream(timer, ch), nil
}

// instrumentStream forwards provider chunks to a new channel, attaching a
// TimingMetrics snapshot to each. It owns the timer's plain fields, so no
// locking is required. Content-bearing chunks (text, reasoning, tool-call
// deltas) mark TTFT and the chunk counter; usage-only and terminal-only
// frames do not count as content.
func instrumentStream(timer *requestTimer, ch <-chan StreamChunk) <-chan StreamChunk {
	out := make(chan StreamChunk)
	go func() {
		defer close(out)
		for chunk := range ch {
			if chunk.Delta != "" || chunk.ReasoningDelta != "" || len(chunk.ToolCalls) > 0 {
				timer.markContent()
			}
			terminal := chunk.Done || chunk.Err != nil
			chunk.Timing = timer.snapshot(terminal)
			out <- chunk
			if terminal {
				return
			}
		}
		// Defensive: the upstream closed without a terminal frame. No
		// built-in lane does this, but third-party providers might; emit
		// one so consumers can always read Total from a final chunk.
		out <- StreamChunk{Done: true, Timing: timer.snapshot(true)}
	}()
	return out
}

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
func (c *Client) Generate(ctx context.Context, prompt []*Message, opts ...Option) (*Response, error) {
	if c == nil || c.provider == nil {
		return nil, &Error{Provider: "", Op: "Generate", Err: ErrInvalidInput}
	}
	merged := mergeOptions(c.defaultOpts, opts)
	resp, err := c.provider.Generate(ctx, prompt, merged...)
	if err != nil {
		return nil, &Error{Provider: c.provider.Name(), Op: "Generate", Err: err}
	}
	return resp, nil
}

// Stream initiates a streaming interaction and returns a read-only channel of chunks.
func (c *Client) Stream(ctx context.Context, prompt []*Message, opts ...Option) (<-chan StreamChunk, error) {
	if c == nil || c.provider == nil {
		return nil, &Error{Provider: "", Op: "Stream", Err: ErrInvalidInput}
	}
	merged := mergeOptions(c.defaultOpts, opts)
	ch, err := c.provider.Stream(ctx, prompt, merged...)
	if err != nil {
		return nil, &Error{Provider: c.provider.Name(), Op: "Stream", Err: err}
	}
	return ch, nil
}

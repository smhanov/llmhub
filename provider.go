package llmhub

import "context"

// Provider describes a backend capable of generating responses from LLMs.
type Provider interface {
	Name() string
	Generate(ctx context.Context, prompt []*Message, opts ...Option) (*Response, error)
	Stream(ctx context.Context, prompt []*Message, opts ...Option) (<-chan StreamChunk, error)
}

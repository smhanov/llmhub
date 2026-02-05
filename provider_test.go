package llmhub

import "context"

type testProvider struct {
	name       string
	generateFn func(ctx context.Context, prompt []*Message, opts ...Option) (*Response, error)
	streamFn   func(ctx context.Context, prompt []*Message, opts ...Option) (<-chan StreamChunk, error)
}

func (p *testProvider) Name() string {
	if p.name != "" {
		return p.name
	}
	return "test"
}

func (p *testProvider) Generate(ctx context.Context, prompt []*Message, opts ...Option) (*Response, error) {
	if p.generateFn != nil {
		return p.generateFn(ctx, prompt, opts...)
	}
	return &Response{Content: []ContentPart{Text("default")}}, nil
}

func (p *testProvider) Stream(ctx context.Context, prompt []*Message, opts ...Option) (<-chan StreamChunk, error) {
	if p.streamFn != nil {
		return p.streamFn(ctx, prompt, opts...)
	}
	ch := make(chan StreamChunk, 1)
	ch <- StreamChunk{Delta: "default", Done: true}
	close(ch)
	return ch, nil
}

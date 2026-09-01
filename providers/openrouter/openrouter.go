package openrouter

import (
	"context"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/smhanov/llmhub"
	"github.com/smhanov/llmhub/internal/openaichat"
)

const (
	providerName   = "openrouter"
	defaultBaseURL = "https://openrouter.ai/api/v1"
	defaultModel   = "openai/gpt-4o-mini"
)

// Client implements the llmhub.Provider interface for OpenRouter's API.
type Client struct {
	chat *openaichat.Client
}

func init() {
	llmhub.MustRegisterProvider(providerName, New)
}

// New instantiates a new OpenRouter provider.
//
// If the configured base URL does not end with "/v1", the suffix is appended
// automatically so that callers can pass either "https://openrouter.ai/api" or
// "https://openrouter.ai/api/v1".
//
// When the model is set to "default" (case-insensitive), the provider queries
// the /v1/models endpoint and selects the first available model.
func New(apiKey string, opts ...llmhub.Option) (llmhub.Provider, error) {
	cfg := llmhub.NewConfig(opts...)
	if cfg.APIKey == "" {
		cfg.APIKey = apiKey
	}
	if cfg.APIKey == "" {
		return nil, fmt.Errorf("openrouter: %w: api key is required", llmhub.ErrInvalidInput)
	}
	if cfg.BaseURL == "" {
		cfg.BaseURL = defaultBaseURL
	}
	cfg.BaseURL = ensureV1Suffix(cfg.BaseURL)
	if cfg.Model == "" {
		cfg.Model = defaultModel
	}
	if cfg.HTTPClient == nil {
		cfg.HTTPClient = &http.Client{Timeout: 2 * time.Minute}
	}
	if strings.EqualFold(cfg.Model, "default") {
		model, err := openaichat.FetchFirstModel(context.Background(), cfg)
		if err != nil {
			return nil, fmt.Errorf("openrouter: resolve default model: %w", err)
		}
		cfg.Model = model
	}
	return &Client{
		chat: openaichat.NewClient(openaichat.ClientConfig{
			ProviderName:          providerName,
			BaseConfig:            cfg,
			AuthHeaderAfterCustom: false,
		}),
	}, nil
}

func (c *Client) Name() string { return c.chat.Name() }

func (c *Client) Generate(ctx context.Context, prompt []*llmhub.Message, opts ...llmhub.Option) (*llmhub.Response, error) {
	return c.chat.Generate(ctx, prompt, opts...)
}

func (c *Client) Stream(ctx context.Context, prompt []*llmhub.Message, opts ...llmhub.Option) (<-chan llmhub.StreamChunk, error) {
	return c.chat.Stream(ctx, prompt, opts...)
}

func ensureV1Suffix(base string) string {
	return openaichat.EnsureV1Suffix(base)
}

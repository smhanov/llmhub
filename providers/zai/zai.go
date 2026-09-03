package zai

import (
	"context"
	"fmt"
	"net/http"
	"time"

	"github.com/smhanov/llmhub"
	"github.com/smhanov/llmhub/internal/openaichat"
)

const (
	providerName = "zai"

	// GeneralBaseURL is Z.AI's general Chat Completions endpoint root.
	GeneralBaseURL = "https://api.z.ai/api/paas/v4"
	// CodingPlanBaseURL is Z.AI's Coding Plan Chat Completions endpoint root.
	CodingPlanBaseURL = "https://api.z.ai/api/coding/paas/v4"
	// DefaultModel is the default GLM model identifier.
	DefaultModel = "glm-5.3"
)

func init() {
	llmhub.MustRegisterProvider(providerName, New)
}

// Client implements the llmhub.Provider interface for Z.AI's OpenAI-compatible API.
type Client struct {
	chat *openaichat.Client
}

// New instantiates a Z.AI provider using the general API endpoint by default.
//
// Z.AI already includes a version segment in its base URL, so unlike the OpenAI
// provider this constructor does not append /v1. Override the endpoint with
// llmhub.WithBaseURL, or use NewCodingPlan for Coding Plan accounts.
func New(apiKey string, opts ...llmhub.Option) (llmhub.Provider, error) {
	return newClient(apiKey, GeneralBaseURL, opts...)
}

// NewCodingPlan instantiates a Z.AI provider using the Coding Plan endpoint.
func NewCodingPlan(apiKey string, opts ...llmhub.Option) (llmhub.Provider, error) {
	return newClient(apiKey, CodingPlanBaseURL, opts...)
}

func newClient(apiKey, defaultBaseURL string, opts ...llmhub.Option) (llmhub.Provider, error) {
	cfg := llmhub.NewConfig(opts...)
	if cfg.APIKey == "" {
		cfg.APIKey = apiKey
	}
	if cfg.APIKey == "" {
		return nil, fmt.Errorf("zai: %w: api key is required", llmhub.ErrInvalidInput)
	}
	if cfg.BaseURL == "" {
		cfg.BaseURL = defaultBaseURL
	}
	cfg.BaseURL = openaichat.ExactBaseURL(cfg.BaseURL)
	if cfg.Model == "" {
		cfg.Model = DefaultModel
	}
	if cfg.HTTPClient == nil {
		cfg.HTTPClient = &http.Client{Timeout: 2 * time.Minute}
	}
	return &Client{
		chat: openaichat.NewClient(openaichat.ClientConfig{
			ProviderName:          providerName,
			BaseConfig:            cfg,
			NormalizeBaseURL:      openaichat.ExactBaseURL,
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

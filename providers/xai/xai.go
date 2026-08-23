package xai

import (
	"context"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/smhanov/llmhub"
	"github.com/smhanov/llmhub/auth"
	"github.com/smhanov/llmhub/auth/oauth2"
	"github.com/smhanov/llmhub/internal/openaichat"
)

const (
	providerName   = "xai"
	DefaultBaseURL = "https://api.x.ai/v1"
	DefaultModel   = "grok-4.6"

	// DefaultOAuthIssuer is the standard xAI OAuth 2.0 authorization server issuer URL.
	DefaultOAuthIssuer = "https://auth.x.ai"

	// DefaultDeviceAuthURL is the default device authorization endpoint for the xAI device flow.
	DefaultDeviceAuthURL = "https://auth.x.ai/oauth2/device/code"

	// DefaultTokenURL is the default token endpoint for xAI device flow and token refreshing.
	DefaultTokenURL = "https://auth.x.ai/oauth2/token"

	// DefaultClientID is the public OAuth device client identifier used by the official xAI/Grok CLI ecosystem.
	// This is a public identifier, not a confidential secret.
	DefaultClientID = "b1a00492-073a-47ea-816f-4c329264a828"

	// DefaultRefreshLeadTime is the default lead time for proactive token refreshing (90 minutes).
	DefaultRefreshLeadTime = 90 * time.Minute
)

// DefaultScopes contains the standard scopes requested during the xAI device authorization flow.
var DefaultScopes = []string{
	"openid",
	"profile",
	"email",
	"offline_access",
	"grok-cli:access",
	"api:access",
	"conversations:read",
	"conversations:write",
	"workspaces:read",
	"workspaces:write",
}

func init() {
	llmhub.MustRegisterProvider(providerName, New)
}

// Client implements the llmhub.Provider interface for xAI (Grok) models.
type Client struct {
	chat *openaichat.Client
}

// New instantiates a new xAI provider.
//
// Credential precedence:
// 1. A non-nil TokenSource (configured via llmhub.WithTokenSource) takes precedence.
// 2. Otherwise Config.APIKey is used (from options or positional apiKey).
// 3. If neither is provided, New returns an error wrapping llmhub.ErrInvalidInput.
func New(apiKey string, opts ...llmhub.Option) (llmhub.Provider, error) {
	cfg := llmhub.NewConfig(opts...)
	if cfg.APIKey == "" {
		cfg.APIKey = apiKey
	}
	if cfg.TokenSource == nil && cfg.APIKey == "" {
		return nil, fmt.Errorf("xai: %w: either an API key or TokenSource is required", llmhub.ErrInvalidInput)
	}
	if cfg.BaseURL == "" {
		cfg.BaseURL = DefaultBaseURL
	}
	cfg.BaseURL = openaichat.EnsureV1Suffix(cfg.BaseURL)
	if cfg.Model == "" {
		cfg.Model = DefaultModel
	}
	if cfg.HTTPClient == nil {
		cfg.HTTPClient = &http.Client{Timeout: 2 * time.Minute}
	}
	if strings.EqualFold(cfg.Model, "default") {
		cfg.Model = DefaultModel
	}
	return &Client{
		chat: openaichat.NewClient(openaichat.ClientConfig{
			ProviderName:          providerName,
			BaseConfig:            cfg,
			AuthHeaderAfterCustom: true,
		}),
	}, nil
}

// Name returns the provider identifier "xai".
func (c *Client) Name() string {
	return providerName
}

// Generate executes a non-streaming completion request.
func (c *Client) Generate(ctx context.Context, prompt []*llmhub.Message, opts ...llmhub.Option) (*llmhub.Response, error) {
	return c.chat.Generate(ctx, prompt, opts...)
}

// Stream executes a streaming completion request and returns a channel of chunks.
func (c *Client) Stream(ctx context.Context, prompt []*llmhub.Message, opts ...llmhub.Option) (<-chan llmhub.StreamChunk, error) {
	return c.chat.Stream(ctx, prompt, opts...)
}

// DeviceFlowOption configures an xAI DeviceFlow instance.
type DeviceFlowOption func(*oauth2.DeviceFlowConfig)

// WithDeviceAuthURL overrides the device authorization URL.
func WithDeviceAuthURL(url string) DeviceFlowOption {
	return func(c *oauth2.DeviceFlowConfig) {
		if url != "" {
			c.DeviceAuthURL = url
		}
	}
}

// WithTokenURL overrides the token endpoint URL.
func WithTokenURL(url string) DeviceFlowOption {
	return func(c *oauth2.DeviceFlowConfig) {
		if url != "" {
			c.TokenURL = url
		}
	}
}

// WithDeviceClientID overrides the client ID for device flow.
func WithDeviceClientID(id string) DeviceFlowOption {
	return func(c *oauth2.DeviceFlowConfig) {
		if id != "" {
			c.ClientID = id
		}
	}
}

// WithScopes sets the scopes requested during device authorization.
func WithScopes(scopes ...string) DeviceFlowOption {
	return func(c *oauth2.DeviceFlowConfig) {
		c.Scopes = scopes
	}
}

// WithHTTPClient sets the HTTP client used for device flow and token requests.
func WithHTTPClient(client *http.Client) DeviceFlowOption {
	return func(c *oauth2.DeviceFlowConfig) {
		c.HTTPClient = client
	}
}

// NewDeviceFlow creates an OAuth 2.0 DeviceFlow pre-configured with xAI defaults.
func NewDeviceFlow(opts ...DeviceFlowOption) *oauth2.DeviceFlow {
	scopes := make([]string, len(DefaultScopes))
	copy(scopes, DefaultScopes)

	cfg := oauth2.DeviceFlowConfig{
		DeviceAuthURL: DefaultDeviceAuthURL,
		TokenURL:      DefaultTokenURL,
		ClientID:      DefaultClientID,
		Scopes:        scopes,
	}
	for _, opt := range opts {
		if opt != nil {
			opt(&cfg)
		}
	}
	return oauth2.NewDeviceFlow(cfg)
}

// TokenSourceOption configures an xAI refreshing token source.
type TokenSourceOption func(*tokenSourceConfig)

type tokenSourceConfig struct {
	flowOptions []DeviceFlowOption
	leadTime    time.Duration
}

// WithTokenSourceLeadTime sets the proactive refresh lead time.
func WithTokenSourceLeadTime(d time.Duration) TokenSourceOption {
	return func(c *tokenSourceConfig) {
		if d > 0 {
			c.leadTime = d
		}
	}
}

// WithTokenSourceFlowOption passes device flow options to the underlying refresh client.
func WithTokenSourceFlowOption(opt DeviceFlowOption) TokenSourceOption {
	return func(c *tokenSourceConfig) {
		if opt != nil {
			c.flowOptions = append(c.flowOptions, opt)
		}
	}
}

// NewTokenSource creates an auth.InvalidatableTokenSource backed by store and the xAI OAuth token refresher.
func NewTokenSource(store auth.TokenStore, opts ...TokenSourceOption) *oauth2.RefreshingTokenSource {
	cfg := tokenSourceConfig{
		leadTime: DefaultRefreshLeadTime,
	}
	for _, opt := range opts {
		if opt != nil {
			opt(&cfg)
		}
	}

	flow := NewDeviceFlow(cfg.flowOptions...)
	return oauth2.NewRefreshingTokenSource(
		store,
		flow,
		oauth2.WithRefreshLeadTime(cfg.leadTime),
	)
}

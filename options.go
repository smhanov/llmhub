package llmhub

import (
	"encoding/json"
	"net/http"

	"github.com/smhanov/llmhub/auth"
)

// Config captures all tunable request options shared across providers.
type Config struct {
	Model       string
	Temperature float64
	// MaxTokens applies a hard cap to generated output tokens.
	// Leave this unset unless you specifically need that cap, because values
	// that are too low can cause the model to return truncated output.
	MaxTokens       int
	APIKey          string
	TokenSource     auth.TokenSource
	BaseURL         string
	HTTPClient      *http.Client
	Headers         map[string]string
	ExtraBody       map[string]json.RawMessage
	EnableWebSearch bool // Enables web search/grounding (Gemini: google_search, Perplexity: always on)
	Tools           []Tool
	ToolChoice      *ToolChoice

	// RetryOnStatus overrides whether HTTP-backed providers retry a given
	// status code. true means retry with backoff; false means return the
	// response immediately. Statuses not present keep the default (retry
	// 429 only). Set via WithRetryOnStatus.
	RetryOnStatus map[int]bool

	// ResponseModalities controls the output modalities the model should
	// produce. For example, setting this to []string{"IMAGE"} tells the
	// Gemini image-generation models to return an image instead of text.
	// Leave nil for the provider default (text).
	ResponseModalities []string

	// Cost accounting: prices expressed per 1 million tokens.
	InputCostPerMillionTokens  float64
	OutputCostPerMillionTokens float64
}

// Option mutates a Config in a functional-options friendly way.
type Option func(*Config)

const defaultTemperature = 0.7

// NewConfig produces a Config populated by the provided options.
func NewConfig(opts ...Option) Config {
	cfg := Config{
		Temperature: defaultTemperature,
		Headers:     map[string]string{},
		ExtraBody:   map[string]json.RawMessage{},
	}
	ApplyOptions(&cfg, opts...)
	return cfg
}

// ApplyOptions mutates a Config in-place with the provided options.
func ApplyOptions(cfg *Config, opts ...Option) {
	if cfg.Headers == nil {
		cfg.Headers = map[string]string{}
	}
	if cfg.ExtraBody == nil {
		cfg.ExtraBody = map[string]json.RawMessage{}
	}
	for _, opt := range opts {
		if opt != nil {
			opt(cfg)
		}
	}
}

// Clone produces a copy of the config suitable for per-request overrides.
func (c Config) Clone() Config {
	clone := c
	if c.Headers != nil {
		clone.Headers = make(map[string]string, len(c.Headers))
		for k, v := range c.Headers {
			clone.Headers[k] = v
		}
	}
	if c.ExtraBody != nil {
		clone.ExtraBody = make(map[string]json.RawMessage, len(c.ExtraBody))
		for k, v := range c.ExtraBody {
			clone.ExtraBody[k] = append(json.RawMessage(nil), v...)
		}
	}
	if c.ResponseModalities != nil {
		clone.ResponseModalities = make([]string, len(c.ResponseModalities))
		copy(clone.ResponseModalities, c.ResponseModalities)
	}
	if c.Tools != nil {
		clone.Tools = make([]Tool, len(c.Tools))
		for i, tool := range c.Tools {
			clone.Tools[i] = cloneTool(tool)
		}
	}
	if c.ToolChoice != nil {
		choice := *c.ToolChoice
		clone.ToolChoice = &choice
	}
	if c.RetryOnStatus != nil {
		clone.RetryOnStatus = make(map[int]bool, len(c.RetryOnStatus))
		for k, v := range c.RetryOnStatus {
			clone.RetryOnStatus[k] = v
		}
	}
	return clone
}

// WithModel selects the target model identifier.
func WithModel(model string) Option {
	return func(c *Config) {
		c.Model = model
	}
}

// WithTemperature sets the sampling temperature.
func WithTemperature(temp float64) Option {
	return func(c *Config) {
		c.Temperature = temp
	}
}

// WithMaxTokens applies a hard cap to the number of generated tokens.
//
// Prefer leaving this unset unless you specifically need a strict output
// limit, because setting it too low often causes truncated responses.
func WithMaxTokens(max int) Option {
	return func(c *Config) {
		c.MaxTokens = max
	}
}

// WithAPIKey stores the credential used by the provider.
func WithAPIKey(key string) Option {
	return func(c *Config) {
		c.APIKey = key
	}
}

// WithTokenSource configures an abstract token source for providers that support OAuth or dynamic token acquisition.
//
// For providers supporting both API keys and OAuth (e.g. xAI), a non-nil TokenSource takes precedence over APIKey.
func WithTokenSource(source auth.TokenSource) Option {
	return func(c *Config) {
		c.TokenSource = source
	}
}

// WithBaseURL overrides the provider base URL (useful for proxies and on-prem).
func WithBaseURL(url string) Option {
	return func(c *Config) {
		c.BaseURL = url
	}
}

// WithHTTPClient swaps the HTTP client used by HTTP-backed providers.
func WithHTTPClient(client *http.Client) Option {
	return func(c *Config) {
		c.HTTPClient = client
	}
}

// WithRetryOnStatus overrides whether HTTP-backed providers retry a given
// status code. By default, providers retry 429 with backoff. Pass
// WithRetryOnStatus(429, false) to surface rate limits immediately so the
// caller can apply its own backoff or failover. Pass true to opt into retry
// for a status that is not retried by default (for example 500).
func WithRetryOnStatus(status int, retry bool) Option {
	return func(c *Config) {
		if c.RetryOnStatus == nil {
			c.RetryOnStatus = map[int]bool{}
		}
		c.RetryOnStatus[status] = retry
	}
}

// WithHeader injects a custom header for every request.
func WithHeader(key, value string) Option {
	return func(c *Config) {
		if c.Headers == nil {
			c.Headers = map[string]string{}
		}
		c.Headers[key] = value
	}
}

// WithExtraBody adds arbitrary additional fields to the outbound JSON request
// body. On key collision, these fields override the standard generated fields.
// Applies only to OpenAI-compatible providers (OpenAI, OpenRouter, xAI);
// other providers ignore it. Values must be valid JSON.
func WithExtraBody(extra map[string]json.RawMessage) Option {
	return func(c *Config) {
		if c.ExtraBody == nil {
			c.ExtraBody = make(map[string]json.RawMessage, len(extra))
		}
		for k, v := range extra {
			c.ExtraBody[k] = append(json.RawMessage(nil), v...)
		}
	}
}

// WithWebSearch enables web search/grounding capabilities.
// For Gemini, this enables google_search tool.
// For Perplexity models, web search is always enabled.
func WithWebSearch(enabled bool) Option {
	return func(c *Config) {
		c.EnableWebSearch = enabled
	}
}

// WithTools supplies callable tools the model may request.
func WithTools(tools ...Tool) Option {
	return func(c *Config) {
		c.Tools = make([]Tool, len(tools))
		for i, tool := range tools {
			c.Tools[i] = cloneTool(tool)
		}
	}
}

// WithToolChoice controls whether supplied tools may, must, or must not be used.
func WithToolChoice(choice ToolChoice) Option {
	return func(c *Config) {
		c.ToolChoice = &choice
	}
}

// WithResponseModalities specifies the output modalities the model should produce.
// For Gemini image-generation models (e.g. gemini-2.5-flash-image), pass "IMAGE"
// to receive image output. Pass "TEXT" and "IMAGE" together to allow mixed output.
// Leave unset for the provider default (text only).
func WithResponseModalities(modalities ...string) Option {
	return func(c *Config) {
		c.ResponseModalities = modalities
	}
}

// WithCost sets the cost per 1 million tokens (input and output) in US dollars.
// This is used to compute the estimated cost of each request based on token usage.
func WithCost(inputCostPerMillionTokens, outputCostPerMillionTokens float64) Option {
	return func(c *Config) {
		c.InputCostPerMillionTokens = inputCostPerMillionTokens
		c.OutputCostPerMillionTokens = outputCostPerMillionTokens
	}
}

// mergeOptions concatenates the default and per-request options while preserving order.
func mergeOptions(defaults []Option, overrides []Option) []Option {
	combined := make([]Option, 0, len(defaults)+len(overrides))
	combined = append(combined, defaults...)
	combined = append(combined, overrides...)
	return combined
}

func cloneTool(tool Tool) Tool {
	clone := tool
	if tool.Parameters != nil {
		clone.Parameters = cloneMap(tool.Parameters)
	}
	return clone
}

func cloneMap(in map[string]interface{}) map[string]interface{} {
	out := make(map[string]interface{}, len(in))
	for k, v := range in {
		out[k] = cloneValue(v)
	}
	return out
}

func cloneValue(v interface{}) interface{} {
	switch typed := v.(type) {
	case map[string]interface{}:
		return cloneMap(typed)
	case []interface{}:
		out := make([]interface{}, len(typed))
		for i, item := range typed {
			out[i] = cloneValue(item)
		}
		return out
	case []string:
		out := make([]string, len(typed))
		copy(out, typed)
		return out
	default:
		return typed
	}
}

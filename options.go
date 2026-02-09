package llmhub

import "net/http"

// Config captures all tunable request options shared across providers.
type Config struct {
	Model             string
	Temperature       float64
	MaxTokens         int
	APIKey            string
	BaseURL           string
	HTTPClient        *http.Client
	Headers           map[string]string
	EnableWebSearch   bool // Enables web search/grounding (Gemini: google_search, Perplexity: always on)

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
	}
	ApplyOptions(&cfg, opts...)
	return cfg
}

// ApplyOptions mutates a Config in-place with the provided options.
func ApplyOptions(cfg *Config, opts ...Option) {
	if cfg.Headers == nil {
		cfg.Headers = map[string]string{}
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

// WithMaxTokens constrains the number of generated tokens.
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

// WithHeader injects a custom header for every request.
func WithHeader(key, value string) Option {
	return func(c *Config) {
		if c.Headers == nil {
			c.Headers = map[string]string{}
		}
		c.Headers[key] = value
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

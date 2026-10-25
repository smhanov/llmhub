package llmhub

import (
	"encoding/json"
	"net/http"

	"github.com/smhanov/llmhub/auth"
)

// Config captures all tunable request options shared across providers.
type Config struct {
	Model string
	// Temperature is the sampling temperature sent to the provider. nil
	// (the zero value) means unset: nothing is sent on the wire and the
	// provider's own default applies. When set, including to 0, that exact
	// value is sent. Leave nil unless you specifically need a custom value;
	// some models (e.g. recent Anthropic ones) reject the field entirely.
	Temperature *float64
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

	// ThinkingLevel controls reasoning depth across providers (see the
	// ThinkingLevel type for the per-provider mapping). Leave empty for the
	// model default. Providers without a native thinking control ignore it.
	// On Gemini this requires a 3.x+ model (see ThinkingLevel).
	ThinkingLevel ThinkingLevel

	// DisablePromptCache turns off prompt caching, which is enabled by
	// default: llmhub emits whatever top-level cache-enabling fields the
	// destination lane understands, and on the native OpenAI provider it
	// additionally renders prompt_cache_options {"mode":"explicit"} so
	// OpenAI's server-side automatic caching (which bills cache writes on
	// GPT-5.6+) is truly off. Set via WithPromptCache(false).
	DisablePromptCache bool

	// CacheKey supplies a stable per-conversation prompt cache key. On the
	// OpenAI-compatible lane it is rendered as the top-level
	// prompt_cache_key routing field (OpenAI natively; OpenRouter uses it
	// as its sticky-routing key fallback), and on xAI it is additionally
	// sent as the x-grok-conv-id request header. Leave empty to rely on
	// provider-automatic caching.
	CacheKey string

	// Cost accounting: prices expressed per 1 million tokens.
	InputCostPerMillionTokens  float64
	OutputCostPerMillionTokens float64
}

// Option mutates a Config in a functional-options friendly way.
type Option func(*Config)

// NewConfig produces a Config populated by the provided options.
func NewConfig(opts ...Option) Config {
	cfg := Config{
		Headers:   map[string]string{},
		ExtraBody: map[string]json.RawMessage{},
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
	if c.Temperature != nil {
		t := *c.Temperature
		clone.Temperature = &t
	}
	if c.RetryOnStatus != nil {
		clone.RetryOnStatus = make(map[int]bool, len(c.RetryOnStatus))
		for k, v := range c.RetryOnStatus {
			clone.RetryOnStatus[k] = v
		}
	}
	return clone
}

// PromptCacheEnabled reports whether prompt caching is on for this config.
// Prompt caching is enabled by default: DisablePromptCache is false (the
// zero value). Only WithPromptCache(false) turns it off.
func (c Config) PromptCacheEnabled() bool {
	return !c.DisablePromptCache
}

// WithModel selects the target model identifier.
func WithModel(model string) Option {
	return func(c *Config) {
		c.Model = model
	}
}

// WithTemperature sets the sampling temperature sent to the provider.
// The value is sent verbatim, including 0. Leave it unset to omit the
// field entirely and use the provider's default; some models (e.g.
// recent Anthropic ones) reject the field, so omit for those.
func WithTemperature(temp float64) Option {
	return func(c *Config) {
		t := temp
		c.Temperature = &t
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
// The control is only sent when tools are configured: if no tools are supplied
// (WithTools with no arguments, or never called), the tool choice is omitted
// from the request rather than sent alone, since providers reject that shape.
// Tools supplied only through WithExtraBody do not count.
// Ollama supports tool calling but has no tool-choice control, so this setting
// is ignored there.
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

// WithThinkingLevel sets the reasoning depth ("minimal", "low", "medium",
// "high", "xhigh"). Leave unset to use the model default. Each provider
// maps it to its native thinking/effort control (see ThinkingLevel);
// providers without one ignore it. On Gemini this requires a 3.x+ model
// (see ThinkingLevel).
func WithThinkingLevel(level ThinkingLevel) Option {
	return func(c *Config) {
		c.ThinkingLevel = level
	}
}

// WithPromptCache turns prompt caching on or off for this config. Caching is
// enabled by default: llmhub emits the cache-enabling fields each provider
// understands (Anthropic automatic-caching cache_control marker; the OpenAI
// prompt_cache_key routing field when a cache key is set via WithCacheKey).
// Pass false to opt out everywhere: marker lanes (Anthropic) get no marker,
// and the native OpenAI provider renders prompt_cache_options
// {"mode":"explicit"} so OpenAI stops caching (and stops billing cache
// writes) without any extra caller action.
func WithPromptCache(enabled bool) Option {
	return func(c *Config) {
		c.DisablePromptCache = !enabled
	}
}

// WithCacheKey sets a stable per-conversation prompt cache key. On the
// OpenAI-compatible lane it is rendered as the top-level prompt_cache_key
// routing field (OpenAI natively; OpenRouter pins sticky provider routing
// to it from the first request), and on xAI it is additionally sent as the
// x-grok-conv-id request header. Leave unset for provider-automatic caching.
func WithCacheKey(key string) Option {
	return func(c *Config) {
		c.CacheKey = key
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

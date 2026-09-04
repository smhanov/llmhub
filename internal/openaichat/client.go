package openaichat

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"regexp"
	"strings"

	"github.com/smhanov/llmhub"
	"github.com/smhanov/llmhub/auth"
	"github.com/smhanov/llmhub/internal/httpretry"
	"github.com/smhanov/llmhub/internal/sse"
)

const (
	ChatEndpointPath   = "/chat/completions"
	ModelsEndpointPath = "/models"
)

// ClientConfig configures the shared OpenAI-compatible chat client.
type ClientConfig struct {
	ProviderName          string
	BaseConfig            llmhub.Config
	NormalizeBaseURL      func(string) string
	AuthHeaderAfterCustom bool
}

// Client implements the OpenAI chat completions wire protocol for both streaming and non-streaming.
type Client struct {
	providerName          string
	baseCfg               llmhub.Config
	normalizeBaseURL      func(string) string
	authHeaderAfterCustom bool
}

// NewClient creates a new shared OpenAI-compatible client.
func NewClient(cfg ClientConfig) *Client {
	normalize := cfg.NormalizeBaseURL
	if normalize == nil {
		normalize = ExactBaseURL
	}
	return &Client{
		providerName:          cfg.ProviderName,
		baseCfg:               cfg.BaseConfig,
		normalizeBaseURL:      normalize,
		authHeaderAfterCustom: cfg.AuthHeaderAfterCustom,
	}
}

// Name returns the configured provider name.
func (c *Client) Name() string {
	return c.providerName
}

// BaseConfig returns the base configuration.
func (c *Client) BaseConfig() llmhub.Config {
	return c.baseCfg
}

// Generate executes a non-streaming chat completion request.
func (c *Client) Generate(ctx context.Context, prompt []*llmhub.Message, opts ...llmhub.Option) (*llmhub.Response, error) {
	cfg := c.MergeConfig(opts...)
	payload, err := BuildRequestPayload(prompt, cfg, false)
	if err != nil {
		return nil, err
	}

	httpResp, err := c.doRequest(ctx, cfg, payload)
	if err != nil {
		return nil, err
	}
	defer httpResp.Body.Close()

	if httpResp.StatusCode >= 400 {
		body, _ := io.ReadAll(httpResp.Body)
		return nil, fmt.Errorf("%s: http %d: %s", c.providerName, httpResp.StatusCode, strings.TrimSpace(string(body)))
	}

	var decoded completionResponse
	if err := json.NewDecoder(httpResp.Body).Decode(&decoded); err != nil {
		return nil, err
	}
	if len(decoded.Choices) == 0 {
		return nil, fmt.Errorf("%s: no choices returned", c.providerName)
	}

	parts, err := ConvertFromAPIContent(decoded.Choices[0].Message.Content)
	if err != nil {
		return nil, err
	}
	parts = AppendReasoningParts(parts, decoded.Choices[0].Message.ReasoningContent, decoded.Choices[0].Message.Reasoning)
	parts = appendToolCallParts(parts, decoded.Choices[0].Message.ToolCalls)

	resp := &llmhub.Response{
		ID:      decoded.ID,
		Content: parts,
		Usage:   usageFromBlock(decoded.Usage),
		Raw:     decoded,
	}
	return resp, nil
}

func usageFromBlock(u usageBlock) llmhub.UsageMetadata {
	var cost float64
	if u.Cost != nil {
		cost = *u.Cost
	} else if u.TotalCost != nil {
		cost = *u.TotalCost
	}
	var cached, reasoning int
	if u.PromptTokensDetails != nil {
		cached = u.PromptTokensDetails.CachedTokens
	}
	if u.CompletionTokensDetails != nil {
		reasoning = u.CompletionTokensDetails.ReasoningTokens
	}
	return llmhub.UsageMetadata{
		PromptTokens:        u.PromptTokens,
		CompletionTokens:    u.CompletionTokens,
		TotalTokens:         u.TotalTokens,
		CacheReadTokens:     cached + u.CacheReadInputTokens,
		CacheCreationTokens: u.CacheCreationInputTokens,
		ReasoningTokens:     reasoning,
		Cost:                cost,
	}
}

func httpStatusError(provider string, resp *http.Response) error {
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	return fmt.Errorf("%s: http %d: %s", provider, resp.StatusCode, strings.TrimSpace(string(body)))
}

// Stream executes a streaming chat completion request and yields chunks over a channel.
func (c *Client) Stream(ctx context.Context, prompt []*llmhub.Message, opts ...llmhub.Option) (<-chan llmhub.StreamChunk, error) {
	cfg := c.MergeConfig(opts...)
	payload, err := BuildRequestPayload(prompt, cfg, true)
	if err != nil {
		return nil, err
	}

	httpResp, err := c.doRequest(ctx, cfg, payload)
	if err != nil {
		return nil, err
	}
	if httpResp.StatusCode >= 400 {
		return nil, httpStatusError(c.providerName, httpResp)
	}

	chunks := make(chan llmhub.StreamChunk)
	go func() {
		defer httpResp.Body.Close()
		defer close(chunks)

		decoder := sse.NewDecoder(httpResp.Body)
		for {
			event, err := decoder.Next()
			if err != nil {
				if errors.Is(err, io.EOF) {
					return
				}
				chunks <- llmhub.StreamChunk{Err: err, Done: true}
				return
			}
			data := strings.TrimSpace(event.Data)
			if data == "" {
				continue
			}
			if data == "[DONE]" {
				chunks <- llmhub.StreamChunk{Done: true}
				return
			}
			var streamResp streamResponse
			if err := json.Unmarshal([]byte(data), &streamResp); err != nil {
				chunks <- llmhub.StreamChunk{Err: err, Done: true}
				return
			}

			var usage *llmhub.UsageMetadata
			if streamResp.Usage != nil {
				u := usageFromBlock(*streamResp.Usage)
				usage = &u
			}

			var (
				deltaText      string
				reasoningDelta string
				toolCalls      []*llmhub.ToolCallContent
				finishReason   string
			)
			if len(streamResp.Choices) > 0 {
				var err error
				deltaText, reasoningDelta, err = ExtractDeltaContent(streamResp.Choices[0].Delta.Content)
				if err != nil {
					chunks <- llmhub.StreamChunk{Err: err, Done: true}
					return
				}
				reasoningDelta = FirstNonEmpty(reasoningDelta, streamResp.Choices[0].Delta.ReasoningContent, streamResp.Choices[0].Delta.Reasoning)
				toolCalls = ToolCallsFromAPI(streamResp.Choices[0].Delta.ToolCalls)
				finishReason = streamResp.Choices[0].FinishReason
			}

			if deltaText == "" && reasoningDelta == "" && len(toolCalls) == 0 && usage == nil && finishReason == "" {
				continue
			}
			select {
			case <-ctx.Done():
				chunks <- llmhub.StreamChunk{Err: ctx.Err(), Done: true}
				return
			case chunks <- llmhub.StreamChunk{
				ID:             streamResp.ID,
				Delta:          deltaText,
				ReasoningDelta: reasoningDelta,
				ToolCalls:      toolCalls,
				Usage:          usage,
				FinishReason:   finishReason,
			}:
			}
		}
	}()

	return chunks, nil
}

// MergeConfig combines base configuration with per-request option overrides.
func (c *Client) MergeConfig(opts ...llmhub.Option) llmhub.Config {
	cfg := c.baseCfg.Clone()
	llmhub.ApplyOptions(&cfg, opts...)
	if cfg.HTTPClient == nil {
		cfg.HTTPClient = c.baseCfg.HTTPClient
	}
	if cfg.Model == "" {
		cfg.Model = c.baseCfg.Model
	}
	if cfg.APIKey == "" {
		cfg.APIKey = c.baseCfg.APIKey
	}
	if cfg.TokenSource == nil {
		cfg.TokenSource = c.baseCfg.TokenSource
	}
	if cfg.BaseURL == "" {
		cfg.BaseURL = c.baseCfg.BaseURL
	}
	cfg.BaseURL = c.normalizeBaseURL(cfg.BaseURL)
	if strings.EqualFold(cfg.Model, "default") {
		cfg.Model = c.baseCfg.Model
	}
	return cfg
}

func (c *Client) doRequest(ctx context.Context, cfg llmhub.Config, payload []byte) (*http.Response, error) {
	var usedToken *auth.Token

	retryCfg := httpretry.FromLLMHub(cfg)
	httpResp, err := httpretry.Do(ctx, cfg.HTTPClient, func() (*http.Request, error) {
		req, token, err := c.buildHTTPRequest(ctx, cfg, payload)
		if err != nil {
			return nil, err
		}
		usedToken = token
		return req, nil
	}, retryCfg)
	if err != nil {
		return nil, err
	}

	// If 401 and TokenSource implements auth.InvalidatableTokenSource, retry once
	if httpResp.StatusCode == http.StatusUnauthorized && cfg.TokenSource != nil && usedToken != nil {
		if inv, ok := cfg.TokenSource.(auth.InvalidatableTokenSource); ok {
			_ = httpResp.Body.Close()
			inv.Invalidate(usedToken.AccessToken)

			retryResp, err := httpretry.Do(ctx, cfg.HTTPClient, func() (*http.Request, error) {
				req, _, err := c.buildHTTPRequest(ctx, cfg, payload)
				if err != nil {
					return nil, err
				}
				return req, nil
			}, retryCfg)
			if err != nil {
				return nil, err
			}
			return retryResp, nil
		}
	}

	return httpResp, nil
}

func (c *Client) buildHTTPRequest(ctx context.Context, cfg llmhub.Config, payload []byte) (*http.Request, *auth.Token, error) {
	url := cfg.BaseURL + ChatEndpointPath
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(payload))
	if err != nil {
		return nil, nil, err
	}

	var usedToken *auth.Token
	var authHeader string

	if cfg.TokenSource != nil {
		token, err := cfg.TokenSource.Token(ctx)
		if err != nil {
			return nil, nil, err
		}
		usedToken = token
		authHeader = token.TypeOrDefault() + " " + token.AccessToken
	} else if cfg.APIKey != "" {
		authHeader = "Bearer " + cfg.APIKey
	}

	req.Header.Set("Content-Type", "application/json")

	if c.authHeaderAfterCustom {
		// xAI style: custom headers first, then Authorization credential
		for k, v := range cfg.Headers {
			req.Header.Set(k, v)
		}
		if authHeader != "" {
			req.Header.Set("Authorization", authHeader)
		}
	} else {
		// OpenAI style: Authorization first, then custom headers
		if authHeader != "" {
			req.Header.Set("Authorization", authHeader)
		}
		for k, v := range cfg.Headers {
			req.Header.Set(k, v)
		}
	}

	return req, usedToken, nil
}

// ExactBaseURL trims trailing slashes without adding a version suffix.
func ExactBaseURL(base string) string {
	return strings.TrimRight(base, "/")
}

// versionSegmentSuffix matches a trailing version path segment such as "/v1",
// "/v2", "/v4" or "/v10" at the end of a URL path.
var versionSegmentSuffix = regexp.MustCompile(`/v[0-9]+$`)

// hasTrailingVersionSegment reports whether the URL path already ends with a
// version segment (/v1, /v2, /v4, ...). Vendors that pin a version other than
// v1 (z.ai serves https://api.z.ai/api/paas/v4/chat/completions) must not get
// "/v1" appended, otherwise requests land on a nonexistent /vN/v1/... path.
func hasTrailingVersionSegment(base string) bool {
	if u, err := url.Parse(base); err == nil && u.Path != "" {
		return versionSegmentSuffix.MatchString(u.Path)
	}
	// Unparseable input (empty path, relative junk): fall back to the raw
	// string so the historical suffix behaviour still applies.
	return versionSegmentSuffix.MatchString(base)
}

// EnsureV1Suffix appends "/v1" to the base URL when it does not already end
// with a version segment. Base URLs that already carry a version segment
// (including a non-v1 one such as "/v4") are returned unchanged.
func EnsureV1Suffix(base string) string {
	trimmed := ExactBaseURL(base)
	if hasTrailingVersionSegment(trimmed) {
		return trimmed
	}
	return trimmed + "/v1"
}

// FetchFirstModel queries the /models endpoint and returns the ID of the first available model.
func FetchFirstModel(ctx context.Context, cfg llmhub.Config) (string, error) {
	url := cfg.BaseURL + ModelsEndpointPath
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return "", err
	}
	if cfg.TokenSource != nil {
		tok, err := cfg.TokenSource.Token(ctx)
		if err != nil {
			return "", err
		}
		req.Header.Set("Authorization", tok.TypeOrDefault()+" "+tok.AccessToken)
	} else if cfg.APIKey != "" {
		req.Header.Set("Authorization", "Bearer "+cfg.APIKey)
	}
	for k, v := range cfg.Headers {
		req.Header.Set(k, v)
	}

	resp, err := cfg.HTTPClient.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 400 {
		body, _ := io.ReadAll(resp.Body)
		return "", fmt.Errorf("http %d: %s", resp.StatusCode, strings.TrimSpace(string(body)))
	}

	var models modelsResponse
	if err := json.NewDecoder(resp.Body).Decode(&models); err != nil {
		return "", err
	}
	if len(models.Data) == 0 {
		return "", errors.New("no models available")
	}
	return models.Data[0].ID, nil
}

// BuildRequestPayload converts normalized messages and config into OpenAI completion JSON payload.
func BuildRequestPayload(prompt []*llmhub.Message, cfg llmhub.Config, stream bool) ([]byte, error) {
	messages := make([]ChatMessage, 0, len(prompt))
	for _, msg := range prompt {
		converted, err := ConvertToAPIMessage(msg)
		if err != nil {
			return nil, err
		}
		messages = append(messages, converted)
	}
	req := CompletionRequest{
		Model:       cfg.Model,
		Messages:    messages,
		Temperature: cfg.Temperature,
		MaxTokens:   cfg.MaxTokens,
		Stream:      stream,
	}
	if stream {
		req.StreamOptions = &streamOptions{IncludeUsage: true}
	}
	if len(cfg.Tools) > 0 {
		req.Tools = ConvertTools(cfg.Tools)
	}
	if cfg.ToolChoice != nil {
		req.ToolChoice = ConvertToolChoice(*cfg.ToolChoice)
	}

	payload, err := json.Marshal(req)
	if err != nil {
		return nil, err
	}
	if len(cfg.ExtraBody) == 0 {
		return payload, nil
	}

	var merged map[string]json.RawMessage
	if err := json.Unmarshal(payload, &merged); err != nil {
		return nil, err
	}
	for k, v := range cfg.ExtraBody {
		merged[k] = v
	}
	return json.Marshal(merged)
}

// ConvertToAPIMessage converts a normalized llmhub.Message into an OpenAI chat message.
func ConvertToAPIMessage(msg *llmhub.Message) (ChatMessage, error) {
	if msg == nil {
		return ChatMessage{}, fmt.Errorf("openaichat: %w: message is nil", llmhub.ErrInvalidInput)
	}
	if msg.Role == llmhub.RoleTool {
		content, err := flattenTextContent(msg.Content)
		if err != nil {
			return ChatMessage{}, err
		}
		return ChatMessage{
			Role:       string(msg.Role),
			Content:    content,
			Name:       metaValue(msg, "name"),
			ToolCallID: metaValue(msg, "tool_call_id"),
		}, nil
	}
	if len(msg.Content) == 0 {
		return ChatMessage{Role: string(msg.Role), Content: ""}, nil
	}
	if len(msg.Content) == 1 {
		if text, ok := msg.Content[0].(*llmhub.TextContent); ok {
			return ChatMessage{Role: string(msg.Role), Content: text.Text}, nil
		}
		if reasoning, ok := msg.Content[0].(*llmhub.ReasoningContent); ok {
			// Reasoning-only assistant turns replay via the dedicated
			// reasoning_content field, never as plain content.
			return ChatMessage{Role: string(msg.Role), Content: "", ReasoningContent: reasoning.Text}, nil
		}
	}
	var textBuilder strings.Builder
	var reasoningBuilder strings.Builder
	var toolCalls []OpenAIToolCall
	content := make([]messageContent, 0, len(msg.Content))
	for _, part := range msg.Content {
		switch v := part.(type) {
		case *llmhub.TextContent:
			textBuilder.WriteString(v.Text)
			content = append(content, messageContent{Type: "text", Text: v.Text})
		case *llmhub.ReasoningContent:
			// Replayed assistant reasoning goes in the dedicated
			// reasoning_content field, never into plain content.
			reasoningBuilder.WriteString(v.Text)
		case *llmhub.ImageContent:
			content = append(content, messageContent{Type: "image_url", ImageURL: &imageURL{URL: v.URL, Detail: v.Detail}})
		case *llmhub.ToolCallContent:
			toolCalls = append(toolCalls, OpenAIToolCall{
				Index: v.Index,
				ID:    v.ID,
				Type:  "function",
				Function: OpenAIFunctionCall{
					Name:      v.Name,
					Arguments: v.Arguments,
				},
			})
		default:
			return ChatMessage{}, fmt.Errorf("openaichat: unsupported content type %T", v)
		}
	}
	if len(toolCalls) > 0 {
		var contentValue interface{}
		if textBuilder.Len() > 0 {
			contentValue = textBuilder.String()
		}
		return ChatMessage{
			Role:             string(msg.Role),
			Content:          contentValue,
			ToolCalls:        toolCalls,
			ReasoningContent: reasoningBuilder.String(),
		}, nil
	}
	// Text-only turns (reasoning extracted into ReasoningContent) use the
	// plain string wire form; mixed multimodal content keeps the part array.
	if len(content) == 1 {
		if content[0].Type == "text" {
			return ChatMessage{
				Role:             string(msg.Role),
				Content:          content[0].Text,
				ReasoningContent: reasoningBuilder.String(),
			}, nil
		}
	}
	return ChatMessage{
		Role:             string(msg.Role),
		Content:          content,
		ReasoningContent: reasoningBuilder.String(),
	}, nil
}

// ConvertFromAPIContent parses JSON response content from an OpenAI chat completion.
func ConvertFromAPIContent(raw json.RawMessage) ([]llmhub.ContentPart, error) {
	if len(raw) == 0 {
		return nil, nil
	}
	if string(raw) == "null" {
		return nil, nil
	}
	if raw[0] == '"' {
		var text string
		if err := json.Unmarshal(raw, &text); err != nil {
			return nil, err
		}
		return []llmhub.ContentPart{llmhub.Text(text)}, nil
	}
	var blocks []messageContent
	if err := json.Unmarshal(raw, &blocks); err == nil {
		parts := make([]llmhub.ContentPart, 0, len(blocks))
		for _, block := range blocks {
			switch block.Type {
			case "text":
				parts = append(parts, llmhub.Text(block.Text))
			case "reasoning", "thinking", "reasoning_content", "redacted_thinking":
				reasoning := FirstNonEmpty(block.ReasoningContent, block.Reasoning, block.Thinking, block.Text)
				parts = AppendReasoningParts(parts, reasoning)
			case "image_url":
				if block.ImageURL != nil {
					parts = append(parts, &llmhub.ImageContent{URL: block.ImageURL.URL, Detail: block.ImageURL.Detail})
				}
			}
		}
		return parts, nil
	}
	var fallback string
	if err := json.Unmarshal(raw, &fallback); err == nil {
		return []llmhub.ContentPart{llmhub.Text(fallback)}, nil
	}
	return nil, fmt.Errorf("openaichat: unable to decode content payload")
}

// ConvertTools transforms normalized llmhub.Tool objects into OpenAI tool definitions.
func ConvertTools(tools []llmhub.Tool) []OpenAITool {
	converted := make([]OpenAITool, 0, len(tools))
	for _, tool := range tools {
		converted = append(converted, OpenAITool{
			Type: "function",
			Function: OpenAIToolFunction{
				Name:        tool.Name,
				Description: tool.Description,
				Parameters:  tool.Parameters,
			},
		})
	}
	return converted
}

// ConvertToolChoice converts a ToolChoice into an OpenAI tool_choice parameter.
func ConvertToolChoice(choice llmhub.ToolChoice) interface{} {
	switch choice.Mode {
	case llmhub.ToolChoiceNone:
		return "none"
	case llmhub.ToolChoiceRequired:
		return "required"
	case llmhub.ToolChoiceNamed:
		return map[string]interface{}{
			"type": "function",
			"function": map[string]string{
				"name": choice.Name,
			},
		}
	default:
		return "auto"
	}
}

func appendToolCallParts(parts []llmhub.ContentPart, calls []OpenAIToolCall) []llmhub.ContentPart {
	for _, call := range ToolCallsFromAPI(calls) {
		parts = append(parts, call)
	}
	return parts
}

// ToolCallsFromAPI converts OpenAI API tool call structures into normalized llmhub.ToolCallContent.
func ToolCallsFromAPI(calls []OpenAIToolCall) []*llmhub.ToolCallContent {
	if len(calls) == 0 {
		return nil
	}
	converted := make([]*llmhub.ToolCallContent, 0, len(calls))
	for _, call := range calls {
		if call.Function.Name == "" && call.Function.Arguments == "" && call.ID == "" {
			continue
		}
		converted = append(converted, llmhub.ToolCallWithIndex(call.Index, call.ID, call.Function.Name, call.Function.Arguments))
	}
	return converted
}

func flattenTextContent(parts []llmhub.ContentPart) (string, error) {
	var b strings.Builder
	for _, part := range parts {
		switch v := part.(type) {
		case *llmhub.TextContent:
			b.WriteString(v.Text)
		case *llmhub.ReasoningContent:
			b.WriteString(v.Text)
		default:
			return "", fmt.Errorf("openaichat: tool messages must be text content")
		}
	}
	return b.String(), nil
}

func metaValue(msg *llmhub.Message, key string) string {
	if msg == nil || msg.Meta == nil {
		return ""
	}
	return msg.Meta[key]
}

// ExtractDeltaContent extracts text and reasoning strings from an SSE delta content field.
func ExtractDeltaContent(raw json.RawMessage) (string, string, error) {
	if len(raw) == 0 {
		return "", "", nil
	}
	if raw[0] == '"' {
		var text string
		if err := json.Unmarshal(raw, &text); err != nil {
			return "", "", err
		}
		return text, "", nil
	}
	var blocks []messageContent
	if err := json.Unmarshal(raw, &blocks); err == nil {
		var builder strings.Builder
		var reasoningBuilder strings.Builder
		for _, block := range blocks {
			switch block.Type {
			case "text":
				builder.WriteString(block.Text)
			case "reasoning", "thinking", "reasoning_content", "redacted_thinking":
				reasoningBuilder.WriteString(FirstNonEmpty(block.ReasoningContent, block.Reasoning, block.Thinking, block.Text))
			}
		}
		return builder.String(), reasoningBuilder.String(), nil
	}
	var block messageContent
	if err := json.Unmarshal(raw, &block); err == nil {
		switch block.Type {
		case "text":
			return block.Text, "", nil
		case "reasoning", "thinking", "reasoning_content", "redacted_thinking":
			return "", FirstNonEmpty(block.ReasoningContent, block.Reasoning, block.Thinking, block.Text), nil
		}
	}
	return "", "", fmt.Errorf("openaichat: unable to decode stream delta payload")
}

// AppendReasoningParts appends reasoning text candidates to parts as ReasoningContent.
func AppendReasoningParts(parts []llmhub.ContentPart, candidates ...string) []llmhub.ContentPart {
	for _, candidate := range candidates {
		if candidate == "" {
			continue
		}
		parts = append(parts, llmhub.Reasoning(candidate))
	}
	return parts
}

// FirstNonEmpty returns the first non-empty string in candidates.
func FirstNonEmpty(values ...string) string {
	for _, value := range values {
		if value != "" {
			return value
		}
	}
	return ""
}

type CompletionRequest struct {
	Model         string         `json:"model"`
	Messages      []ChatMessage  `json:"messages"`
	Temperature   float64        `json:"temperature,omitempty"`
	MaxTokens     int            `json:"max_tokens,omitempty"`
	Stream        bool           `json:"stream,omitempty"`
	StreamOptions *streamOptions `json:"stream_options,omitempty"`
	Tools         []OpenAITool   `json:"tools,omitempty"`
	ToolChoice    interface{}    `json:"tool_choice,omitempty"`
}

// streamOptions mirrors the OpenAI stream_options parameter. llmhub always
// requests include_usage on streaming requests so callers can rely on a
// trailing usage frame for failover-before-commit accounting.
type streamOptions struct {
	IncludeUsage bool `json:"include_usage"`
}

type ChatMessage struct {
	Role       string           `json:"role"`
	Content    interface{}      `json:"content"`
	Name       string           `json:"name,omitempty"`
	ToolCallID string           `json:"tool_call_id,omitempty"`
	ToolCalls  []OpenAIToolCall `json:"tool_calls,omitempty"`
	// ReasoningContent replays a prior assistant turn's reasoning
	// (DeepSeek-style `reasoning_content`). It is request-side only; the
	// response side parses it in chatMessageResponse.
	ReasoningContent string `json:"reasoning_content,omitempty"`
}

type messageContent struct {
	Type             string    `json:"type"`
	Text             string    `json:"text,omitempty"`
	Reasoning        string    `json:"reasoning,omitempty"`
	ReasoningContent string    `json:"reasoning_content,omitempty"`
	Thinking         string    `json:"thinking,omitempty"`
	ImageURL         *imageURL `json:"image_url,omitempty"`
}

type imageURL struct {
	URL    string `json:"url"`
	Detail string `json:"detail,omitempty"`
}

type OpenAITool struct {
	Type     string             `json:"type"`
	Function OpenAIToolFunction `json:"function"`
}

type OpenAIToolFunction struct {
	Name        string                 `json:"name"`
	Description string                 `json:"description,omitempty"`
	Parameters  map[string]interface{} `json:"parameters,omitempty"`
}

type OpenAIToolCall struct {
	Index    int                `json:"index,omitempty"`
	ID       string             `json:"id,omitempty"`
	Type     string             `json:"type,omitempty"`
	Function OpenAIFunctionCall `json:"function"`
}

type OpenAIFunctionCall struct {
	Name      string `json:"name,omitempty"`
	Arguments string `json:"arguments,omitempty"`
}

type completionResponse struct {
	ID      string `json:"id"`
	Choices []struct {
		Message chatMessageResponse `json:"message"`
	} `json:"choices"`
	Usage usageBlock `json:"usage"`
}

type chatMessageResponse struct {
	Role             string           `json:"role"`
	Content          json.RawMessage  `json:"content"`
	Reasoning        string           `json:"reasoning,omitempty"`
	ReasoningContent string           `json:"reasoning_content,omitempty"`
	ToolCalls        []OpenAIToolCall `json:"tool_calls,omitempty"`
}

type usageBlock struct {
	PromptTokens             int                      `json:"prompt_tokens"`
	CompletionTokens         int                      `json:"completion_tokens"`
	TotalTokens              int                      `json:"total_tokens"`
	PromptTokensDetails      *promptTokensDetails     `json:"prompt_tokens_details,omitempty"`
	CompletionTokensDetails  *completionTokensDetails `json:"completion_tokens_details,omitempty"`
	CacheReadInputTokens     int                      `json:"cache_read_input_tokens,omitempty"`
	CacheCreationInputTokens int                      `json:"cache_creation_input_tokens,omitempty"`
	Cost                     *float64                 `json:"cost,omitempty"`
	TotalCost                *float64                 `json:"total_cost,omitempty"`
}

type promptTokensDetails struct {
	CachedTokens int `json:"cached_tokens,omitempty"`
}

type completionTokensDetails struct {
	ReasoningTokens int `json:"reasoning_tokens,omitempty"`
}

type streamResponse struct {
	ID      string `json:"id"`
	Choices []struct {
		Delta struct {
			Content          json.RawMessage  `json:"content"`
			Reasoning        string           `json:"reasoning,omitempty"`
			ReasoningContent string           `json:"reasoning_content,omitempty"`
			ToolCalls        []OpenAIToolCall `json:"tool_calls,omitempty"`
		} `json:"delta"`
		FinishReason string `json:"finish_reason,omitempty"`
	} `json:"choices"`
	Usage *usageBlock `json:"usage,omitempty"`
}

type modelsResponse struct {
	Data []struct {
		ID string `json:"id"`
	} `json:"data"`
}

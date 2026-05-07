package openai

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/smhanov/llmhub"
	"github.com/smhanov/llmhub/internal/httpretry"
	"github.com/smhanov/llmhub/internal/sse"
)

const (
	providerName       = "openai"
	defaultBaseURL     = "https://api.openai.com/v1"
	chatEndpointPath   = "/chat/completions"
	modelsEndpointPath = "/models"
)

// Client implements the llmhub.Provider interface for OpenAI's Chat Completions API.
type Client struct {
	baseCfg llmhub.Config
}

func init() {
	llmhub.MustRegisterProvider(providerName, New)
}

// New instantiates a new OpenAI provider.
//
// If the configured base URL does not end with "/v1", the suffix is appended
// automatically so that callers can pass either "https://api.openai.com" or
// "https://api.openai.com/v1".
//
// When the model is set to "default" (case-insensitive), the provider queries
// the /v1/models endpoint and selects the first available model.
func New(apiKey string, opts ...llmhub.Option) (llmhub.Provider, error) {
	cfg := llmhub.NewConfig(opts...)
	if cfg.APIKey == "" {
		cfg.APIKey = apiKey
	}
	if cfg.APIKey == "" {
		return nil, fmt.Errorf("openai: %w: api key is required", llmhub.ErrInvalidInput)
	}
	if cfg.BaseURL == "" {
		cfg.BaseURL = defaultBaseURL
	}
	cfg.BaseURL = ensureV1Suffix(cfg.BaseURL)
	if cfg.Model == "" {
		cfg.Model = "gpt-4o-mini"
	}
	if cfg.HTTPClient == nil {
		cfg.HTTPClient = &http.Client{Timeout: 2 * time.Minute}
	}
	if strings.EqualFold(cfg.Model, "default") {
		model, err := fetchFirstModel(context.Background(), cfg)
		if err != nil {
			return nil, fmt.Errorf("openai: resolve default model: %w", err)
		}
		cfg.Model = model
	}
	return &Client{baseCfg: cfg}, nil
}

func (c *Client) Name() string { return providerName }

func (c *Client) Generate(ctx context.Context, prompt []*llmhub.Message, opts ...llmhub.Option) (*llmhub.Response, error) {
	cfg := c.mergeConfig(opts...)
	payload, err := buildRequestPayload(prompt, cfg, false)
	if err != nil {
		return nil, err
	}
	httpResp, err := httpretry.Do(ctx, cfg.HTTPClient, func() (*http.Request, error) {
		httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, cfg.BaseURL+chatEndpointPath, bytes.NewReader(payload))
		if err != nil {
			return nil, err
		}
		applyHeaders(httpReq, cfg)
		return httpReq, nil
	}, httpretry.DefaultConfig())
	if err != nil {
		return nil, err
	}
	defer httpResp.Body.Close()
	if httpResp.StatusCode >= 400 {
		body, _ := io.ReadAll(httpResp.Body)
		return nil, fmt.Errorf("openai: http %d: %s", httpResp.StatusCode, strings.TrimSpace(string(body)))
	}

	var decoded completionResponse
	if err := json.NewDecoder(httpResp.Body).Decode(&decoded); err != nil {
		return nil, err
	}
	if len(decoded.Choices) == 0 {
		return nil, errors.New("openai: no choices returned")
	}
	parts, err := convertFromAPIContent(decoded.Choices[0].Message.Content)
	if err != nil {
		return nil, err
	}
	parts = appendReasoningParts(parts, decoded.Choices[0].Message.ReasoningContent, decoded.Choices[0].Message.Reasoning)
	parts = appendToolCallParts(parts, decoded.Choices[0].Message.ToolCalls)
	resp := &llmhub.Response{
		ID:      decoded.ID,
		Content: parts,
		Usage: llmhub.UsageMetadata{
			PromptTokens:     decoded.Usage.PromptTokens,
			CompletionTokens: decoded.Usage.CompletionTokens,
			TotalTokens:      decoded.Usage.TotalTokens,
		},
		Raw: decoded,
	}
	return resp, nil
}

func (c *Client) Stream(ctx context.Context, prompt []*llmhub.Message, opts ...llmhub.Option) (<-chan llmhub.StreamChunk, error) {
	cfg := c.mergeConfig(opts...)
	payload, err := buildRequestPayload(prompt, cfg, true)
	if err != nil {
		return nil, err
	}
	httpResp, err := httpretry.Do(ctx, cfg.HTTPClient, func() (*http.Request, error) {
		httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, cfg.BaseURL+chatEndpointPath, bytes.NewReader(payload))
		if err != nil {
			return nil, err
		}
		applyHeaders(httpReq, cfg)
		return httpReq, nil
	}, httpretry.DefaultConfig())
	if err != nil {
		return nil, err
	}

	chunks := make(chan llmhub.StreamChunk)
	go func() {
		defer httpResp.Body.Close()
		defer close(chunks)
		if httpResp.StatusCode >= 400 {
			body, _ := io.ReadAll(httpResp.Body)
			chunks <- llmhub.StreamChunk{Err: fmt.Errorf("openai: http %d: %s", httpResp.StatusCode, strings.TrimSpace(string(body))), Done: true}
			return
		}

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
			var payload streamResponse
			if err := json.Unmarshal([]byte(data), &payload); err != nil {
				chunks <- llmhub.StreamChunk{Err: err, Done: true}
				return
			}
			if len(payload.Choices) == 0 {
				continue
			}
			deltaText, reasoningDelta, err := extractDeltaContent(payload.Choices[0].Delta.Content)
			if err != nil {
				chunks <- llmhub.StreamChunk{Err: err, Done: true}
				return
			}
			reasoningDelta = firstNonEmpty(reasoningDelta, payload.Choices[0].Delta.ReasoningContent, payload.Choices[0].Delta.Reasoning)
			toolCalls := toolCallsFromAPI(payload.Choices[0].Delta.ToolCalls)
			if deltaText == "" && reasoningDelta == "" && len(toolCalls) == 0 {
				continue
			}
			select {
			case <-ctx.Done():
				chunks <- llmhub.StreamChunk{Err: ctx.Err(), Done: true}
				return
			case chunks <- llmhub.StreamChunk{Delta: deltaText, ReasoningDelta: reasoningDelta, ToolCalls: toolCalls}:
			}
		}
	}()

	return chunks, nil
}

func (c *Client) mergeConfig(opts ...llmhub.Option) llmhub.Config {
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
	if cfg.BaseURL == "" {
		cfg.BaseURL = c.baseCfg.BaseURL
	}
	// Re-apply normalisations that New() performed, because the caller's
	// original options (stored in llmhub.Client.defaultOpts) are re-applied
	// on every call and may overwrite the corrected values.
	cfg.BaseURL = ensureV1Suffix(cfg.BaseURL)
	if strings.EqualFold(cfg.Model, "default") {
		cfg.Model = c.baseCfg.Model
	}
	return cfg
}

func buildRequestPayload(prompt []*llmhub.Message, cfg llmhub.Config, stream bool) ([]byte, error) {
	messages := make([]chatMessage, 0, len(prompt))
	for _, msg := range prompt {
		converted, err := convertToAPIMessage(msg)
		if err != nil {
			return nil, err
		}
		messages = append(messages, converted)
	}
	req := completionRequest{
		Model:       cfg.Model,
		Messages:    messages,
		Temperature: cfg.Temperature,
		MaxTokens:   cfg.MaxTokens,
		Stream:      stream,
	}
	if len(cfg.Tools) > 0 {
		req.Tools = convertTools(cfg.Tools)
	}
	if cfg.ToolChoice != nil {
		req.ToolChoice = convertToolChoice(*cfg.ToolChoice)
	}
	return json.Marshal(req)
}

func applyHeaders(r *http.Request, cfg llmhub.Config) {
	r.Header.Set("Authorization", "Bearer "+cfg.APIKey)
	r.Header.Set("Content-Type", "application/json")
	for k, v := range cfg.Headers {
		r.Header.Set(k, v)
	}
}

func convertToAPIMessage(msg *llmhub.Message) (chatMessage, error) {
	if msg == nil {
		return chatMessage{}, fmt.Errorf("openai: %w: message is nil", llmhub.ErrInvalidInput)
	}
	if msg.Role == llmhub.RoleTool {
		content, err := flattenTextContent(msg.Content)
		if err != nil {
			return chatMessage{}, err
		}
		return chatMessage{
			Role:       string(msg.Role),
			Content:    content,
			Name:       metaValue(msg, "name"),
			ToolCallID: metaValue(msg, "tool_call_id"),
		}, nil
	}
	if len(msg.Content) == 0 {
		return chatMessage{Role: string(msg.Role), Content: ""}, nil
	}
	if len(msg.Content) == 1 {
		if text, ok := msg.Content[0].(*llmhub.TextContent); ok {
			return chatMessage{Role: string(msg.Role), Content: text.Text}, nil
		}
		if reasoning, ok := msg.Content[0].(*llmhub.ReasoningContent); ok {
			return chatMessage{Role: string(msg.Role), Content: reasoning.Text}, nil
		}
	}
	var textBuilder strings.Builder
	var toolCalls []openAIToolCall
	content := make([]messageContent, 0, len(msg.Content))
	for _, part := range msg.Content {
		switch v := part.(type) {
		case *llmhub.TextContent:
			textBuilder.WriteString(v.Text)
			content = append(content, messageContent{Type: "text", Text: v.Text})
		case *llmhub.ReasoningContent:
			textBuilder.WriteString(v.Text)
			content = append(content, messageContent{Type: "text", Text: v.Text})
		case *llmhub.ImageContent:
			content = append(content, messageContent{Type: "image_url", ImageURL: &imageURL{URL: v.URL, Detail: v.Detail}})
		case *llmhub.ToolCallContent:
			toolCalls = append(toolCalls, openAIToolCall{
				ID:   v.ID,
				Type: "function",
				Function: openAIFunctionCall{
					Name:      v.Name,
					Arguments: v.Arguments,
				},
			})
		default:
			return chatMessage{}, fmt.Errorf("openai: unsupported content type %T", v)
		}
	}
	if len(toolCalls) > 0 {
		var contentValue interface{}
		if textBuilder.Len() > 0 {
			contentValue = textBuilder.String()
		}
		return chatMessage{Role: string(msg.Role), Content: contentValue, ToolCalls: toolCalls}, nil
	}
	return chatMessage{Role: string(msg.Role), Content: content}, nil
}

func convertFromAPIContent(raw json.RawMessage) ([]llmhub.ContentPart, error) {
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
				reasoning := firstNonEmpty(block.ReasoningContent, block.Reasoning, block.Thinking, block.Text)
				parts = appendReasoningParts(parts, reasoning)
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
	return nil, fmt.Errorf("openai: unable to decode content payload")
}

func convertTools(tools []llmhub.Tool) []openAITool {
	converted := make([]openAITool, 0, len(tools))
	for _, tool := range tools {
		converted = append(converted, openAITool{
			Type: "function",
			Function: openAIToolFunction{
				Name:        tool.Name,
				Description: tool.Description,
				Parameters:  tool.Parameters,
			},
		})
	}
	return converted
}

func convertToolChoice(choice llmhub.ToolChoice) interface{} {
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

func appendToolCallParts(parts []llmhub.ContentPart, calls []openAIToolCall) []llmhub.ContentPart {
	for _, call := range calls {
		if call.Function.Name == "" {
			continue
		}
		parts = append(parts, llmhub.ToolCall(call.ID, call.Function.Name, call.Function.Arguments))
	}
	return parts
}

func toolCallsFromAPI(calls []openAIToolCall) []*llmhub.ToolCallContent {
	if len(calls) == 0 {
		return nil
	}
	converted := make([]*llmhub.ToolCallContent, 0, len(calls))
	for _, call := range calls {
		if call.Function.Name == "" && call.Function.Arguments == "" {
			continue
		}
		converted = append(converted, llmhub.ToolCall(call.ID, call.Function.Name, call.Function.Arguments))
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
			return "", fmt.Errorf("openai: tool messages must be text content")
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

func extractDeltaContent(raw json.RawMessage) (string, string, error) {
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
				reasoningBuilder.WriteString(firstNonEmpty(block.ReasoningContent, block.Reasoning, block.Thinking, block.Text))
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
			return "", firstNonEmpty(block.ReasoningContent, block.Reasoning, block.Thinking, block.Text), nil
		}
	}
	return "", "", fmt.Errorf("openai: unable to decode stream delta payload")
}

func appendReasoningParts(parts []llmhub.ContentPart, candidates ...string) []llmhub.ContentPart {
	for _, candidate := range candidates {
		if candidate == "" {
			continue
		}
		parts = append(parts, llmhub.Reasoning(candidate))
	}
	return parts
}

func firstNonEmpty(values ...string) string {
	for _, value := range values {
		if value != "" {
			return value
		}
	}
	return ""
}

type completionRequest struct {
	Model       string        `json:"model"`
	Messages    []chatMessage `json:"messages"`
	Temperature float64       `json:"temperature,omitempty"`
	MaxTokens   int           `json:"max_tokens,omitempty"`
	Stream      bool          `json:"stream,omitempty"`
	Tools       []openAITool  `json:"tools,omitempty"`
	ToolChoice  interface{}   `json:"tool_choice,omitempty"`
}

type chatMessage struct {
	Role       string           `json:"role"`
	Content    interface{}      `json:"content"`
	Name       string           `json:"name,omitempty"`
	ToolCallID string           `json:"tool_call_id,omitempty"`
	ToolCalls  []openAIToolCall `json:"tool_calls,omitempty"`
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

type openAITool struct {
	Type     string             `json:"type"`
	Function openAIToolFunction `json:"function"`
}

type openAIToolFunction struct {
	Name        string                 `json:"name"`
	Description string                 `json:"description,omitempty"`
	Parameters  map[string]interface{} `json:"parameters,omitempty"`
}

type openAIToolCall struct {
	ID       string             `json:"id,omitempty"`
	Type     string             `json:"type,omitempty"`
	Function openAIFunctionCall `json:"function"`
}

type openAIFunctionCall struct {
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
	ToolCalls        []openAIToolCall `json:"tool_calls,omitempty"`
}

type usageBlock struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

type streamResponse struct {
	ID      string `json:"id"`
	Choices []struct {
		Delta struct {
			Content          json.RawMessage  `json:"content"`
			Reasoning        string           `json:"reasoning,omitempty"`
			ReasoningContent string           `json:"reasoning_content,omitempty"`
			ToolCalls        []openAIToolCall `json:"tool_calls,omitempty"`
		} `json:"delta"`
	} `json:"choices"`
}

type modelsResponse struct {
	Data []struct {
		ID string `json:"id"`
	} `json:"data"`
}

// ensureV1Suffix appends "/v1" to the base URL when it is not already present.
func ensureV1Suffix(base string) string {
	trimmed := strings.TrimRight(base, "/")
	if strings.HasSuffix(trimmed, "/v1") {
		return trimmed
	}
	return trimmed + "/v1"
}

// fetchFirstModel queries the /models endpoint and returns the ID of the first
// model in the response.
func fetchFirstModel(ctx context.Context, cfg llmhub.Config) (string, error) {
	url := cfg.BaseURL + modelsEndpointPath
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return "", err
	}
	req.Header.Set("Authorization", "Bearer "+cfg.APIKey)
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

package ollama

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/smhanov/llmhub"
	"github.com/smhanov/llmhub/internal/httpretry"
)

const (
	providerName   = "ollama"
	defaultBaseURL = "http://localhost:11434"
	chatEndpoint   = "/api/chat"
)

type Client struct {
	baseCfg llmhub.Config
}

func init() { llmhub.MustRegisterProvider(providerName, New) }

func New(apiKey string, opts ...llmhub.Option) (llmhub.Provider, error) {
	cfg := llmhub.NewConfig(opts...)
	if cfg.BaseURL == "" {
		cfg.BaseURL = defaultBaseURL
	}
	if cfg.Model == "" {
		cfg.Model = "llama3"
	}
	if cfg.HTTPClient == nil {
		cfg.HTTPClient = &http.Client{Timeout: 2 * time.Minute}
	}
	if cfg.APIKey == "" {
		cfg.APIKey = apiKey
	}
	return &Client{baseCfg: cfg}, nil
}

func (c *Client) Name() string { return providerName }

func (c *Client) Generate(ctx context.Context, prompt []*llmhub.Message, opts ...llmhub.Option) (*llmhub.Response, error) {
	cfg := c.mergeConfig(opts...)
	reqBody, err := buildChatRequest(prompt, cfg, false)
	if err != nil {
		return nil, err
	}
	resp, err := httpretry.Do(ctx, cfg.HTTPClient, func() (*http.Request, error) {
		req, err := http.NewRequestWithContext(ctx, http.MethodPost, cfg.BaseURL+chatEndpoint, bytes.NewReader(reqBody))
		if err != nil {
			return nil, err
		}
		applyHeaders(req, cfg)
		return req, nil
	}, httpretry.FromLLMHub(cfg))
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 400 {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("ollama: http %d: %s", resp.StatusCode, strings.TrimSpace(string(body)))
	}
	var decoded chatResponse
	if err := json.NewDecoder(resp.Body).Decode(&decoded); err != nil {
		return nil, err
	}
	if decoded.Error != "" {
		return nil, fmt.Errorf("ollama: %s", decoded.Error)
	}
	text := decoded.Message.Content
	if text == "" {
		text = decoded.Response
	}
	parts := make([]llmhub.ContentPart, 0, 2)
	if decoded.Message.Thinking != "" {
		parts = append(parts, llmhub.Reasoning(decoded.Message.Thinking))
	}
	parts = append(parts, llmhub.Text(text))
	parts = appendToolCallParts(parts, decoded.Message.ToolCalls)
	var cost float64
	if decoded.Cost != nil {
		cost = *decoded.Cost
	} else if decoded.TotalCost != nil {
		cost = *decoded.TotalCost
	}
	return &llmhub.Response{
		Content: parts,
		Usage: llmhub.UsageMetadata{
			PromptTokens:     decoded.PromptEvalCount,
			CompletionTokens: decoded.EvalCount,
			TotalTokens:      decoded.PromptEvalCount + decoded.EvalCount,
			Cost:             cost,
		},
		Raw: decoded,
	}, nil
}

func (c *Client) Stream(ctx context.Context, prompt []*llmhub.Message, opts ...llmhub.Option) (<-chan llmhub.StreamChunk, error) {
	cfg := c.mergeConfig(opts...)
	reqBody, err := buildChatRequest(prompt, cfg, true)
	if err != nil {
		return nil, err
	}
	resp, err := httpretry.Do(ctx, cfg.HTTPClient, func() (*http.Request, error) {
		req, err := http.NewRequestWithContext(ctx, http.MethodPost, cfg.BaseURL+chatEndpoint, bytes.NewReader(reqBody))
		if err != nil {
			return nil, err
		}
		applyHeaders(req, cfg)
		return req, nil
	}, httpretry.FromLLMHub(cfg))
	if err != nil {
		return nil, err
	}
	if resp.StatusCode >= 400 {
		defer resp.Body.Close()
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("ollama: http %d: %s", resp.StatusCode, strings.TrimSpace(string(body)))
	}

	ch := make(chan llmhub.StreamChunk)
	go func() {
		defer resp.Body.Close()
		defer close(ch)
		scanner := bufio.NewScanner(resp.Body)
		buf := make([]byte, 0, 64*1024)
		scanner.Buffer(buf, 4*1024*1024)
		for scanner.Scan() {
			line := strings.TrimSpace(scanner.Text())
			if line == "" {
				continue
			}
			var chunk chatResponse
			if err := json.Unmarshal([]byte(line), &chunk); err != nil {
				ch <- llmhub.StreamChunk{Err: err, Done: true}
				return
			}
			if chunk.Error != "" {
				ch <- llmhub.StreamChunk{Err: fmt.Errorf("ollama: %s", chunk.Error), Done: true}
				return
			}
			text := chunk.Message.Content
			if text == "" {
				text = chunk.Response
			}
			if text != "" {
				select {
				case <-ctx.Done():
					ch <- llmhub.StreamChunk{Err: ctx.Err(), Done: true}
					return
				case ch <- llmhub.StreamChunk{
					Delta:          text,
					ReasoningDelta: chunk.Message.Thinking,
					ToolCalls:      toolCallsFromAPI(chunk.Message.ToolCalls),
				}:
				}
			} else if chunk.Message.Thinking != "" {
				select {
				case <-ctx.Done():
					ch <- llmhub.StreamChunk{Err: ctx.Err(), Done: true}
					return
				case ch <- llmhub.StreamChunk{ReasoningDelta: chunk.Message.Thinking}:
				}
			} else if len(chunk.Message.ToolCalls) > 0 {
				select {
				case <-ctx.Done():
					ch <- llmhub.StreamChunk{Err: ctx.Err(), Done: true}
					return
				case ch <- llmhub.StreamChunk{ToolCalls: toolCallsFromAPI(chunk.Message.ToolCalls)}:
				}
			}
			if chunk.Done {
				ch <- llmhub.StreamChunk{Done: true, FinishReason: mapDoneReason(chunk.DoneReason)}
				return
			}
		}
		if err := scanner.Err(); err != nil {
			ch <- llmhub.StreamChunk{Err: err, Done: true}
			return
		}
		ch <- llmhub.StreamChunk{Done: true}
	}()

	return ch, nil
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
	if cfg.BaseURL == "" {
		cfg.BaseURL = c.baseCfg.BaseURL
	}
	if cfg.APIKey == "" {
		cfg.APIKey = c.baseCfg.APIKey
	}
	return cfg
}

func buildChatRequest(prompt []*llmhub.Message, cfg llmhub.Config, stream bool) ([]byte, error) {
	msgs := make([]ollamaMessage, 0, len(prompt))
	for _, msg := range prompt {
		if msg == nil {
			continue
		}
		converted, err := convertMessage(msg)
		if err != nil {
			return nil, err
		}
		msgs = append(msgs, converted)
	}
	req := chatRequest{
		Model:    cfg.Model,
		Messages: msgs,
		Stream:   stream,
	}
	if len(cfg.Tools) > 0 {
		req.Tools = convertTools(cfg.Tools)
	}
	if cfg.Temperature != 0 || cfg.MaxTokens != 0 {
		req.Options = map[string]interface{}{}
		if cfg.Temperature != 0 {
			req.Options["temperature"] = cfg.Temperature
		}
		if cfg.MaxTokens != 0 {
			req.Options["num_predict"] = cfg.MaxTokens
		}
	}
	return json.Marshal(req)
}

func convertMessage(msg *llmhub.Message) (ollamaMessage, error) {
	if msg.Role == llmhub.RoleTool {
		text, err := flattenText(msg.Content)
		if err != nil {
			return ollamaMessage{}, err
		}
		return ollamaMessage{
			Role:       string(msg.Role),
			Content:    text,
			Name:       metaValue(msg, "name"),
			ToolCallID: metaValue(msg, "tool_call_id"),
		}, nil
	}
	var textBuilder strings.Builder
	var thinkingBuilder strings.Builder
	var toolCalls []ollamaToolCall
	for _, part := range msg.Content {
		switch v := part.(type) {
		case *llmhub.TextContent:
			textBuilder.WriteString(v.Text)
		case *llmhub.ReasoningContent:
			thinkingBuilder.WriteString(v.Text)
		case *llmhub.ToolCallContent:
			args, err := parseArguments(v.Arguments)
			if err != nil {
				return ollamaMessage{}, err
			}
			toolCalls = append(toolCalls, ollamaToolCall{
				ID:   v.ID,
				Type: "function",
				Function: ollamaFunctionCall{
					Name:      v.Name,
					Arguments: args,
				},
			})
		default:
			return ollamaMessage{}, fmt.Errorf("ollama: only text, reasoning, and tool call content are supported")
		}
	}
	return ollamaMessage{Role: string(msg.Role), Content: textBuilder.String(), Thinking: thinkingBuilder.String(), ToolCalls: toolCalls}, nil
}

func flattenText(parts []llmhub.ContentPart) (string, error) {
	var b strings.Builder
	for _, part := range parts {
		text, ok := part.(*llmhub.TextContent)
		if !ok {
			return "", fmt.Errorf("ollama: only text content is supported")
		}
		b.WriteString(text.Text)
	}
	return b.String(), nil
}

func convertTools(tools []llmhub.Tool) []ollamaTool {
	converted := make([]ollamaTool, 0, len(tools))
	for _, tool := range tools {
		converted = append(converted, ollamaTool{
			Type: "function",
			Function: ollamaToolFunction{
				Name:        tool.Name,
				Description: tool.Description,
				Parameters:  tool.Parameters,
			},
		})
	}
	return converted
}

func appendToolCallParts(parts []llmhub.ContentPart, calls []ollamaToolCall) []llmhub.ContentPart {
	for _, call := range toolCallsFromAPI(calls) {
		parts = append(parts, call)
	}
	return parts
}

func toolCallsFromAPI(calls []ollamaToolCall) []*llmhub.ToolCallContent {
	if len(calls) == 0 {
		return nil
	}
	converted := make([]*llmhub.ToolCallContent, 0, len(calls))
	for _, call := range calls {
		args, _ := json.Marshal(call.Function.Arguments)
		converted = append(converted, llmhub.ToolCall(call.ID, call.Function.Name, string(args)))
	}
	return converted
}

func parseArguments(arguments string) (map[string]interface{}, error) {
	if strings.TrimSpace(arguments) == "" {
		return map[string]interface{}{}, nil
	}
	var parsed map[string]interface{}
	if err := json.Unmarshal([]byte(arguments), &parsed); err != nil {
		return nil, fmt.Errorf("ollama: tool call arguments must be a JSON object: %w", err)
	}
	return parsed, nil
}

func metaValue(msg *llmhub.Message, key string) string {
	if msg == nil || msg.Meta == nil {
		return ""
	}
	return msg.Meta[key]
}

func applyHeaders(r *http.Request, cfg llmhub.Config) {
	r.Header.Set("content-type", "application/json")
	if cfg.APIKey != "" {
		r.Header.Set("authorization", "Bearer "+cfg.APIKey)
	}
	for k, v := range cfg.Headers {
		r.Header.Set(k, v)
	}
}

type chatRequest struct {
	Model    string                 `json:"model"`
	Messages []ollamaMessage        `json:"messages"`
	Stream   bool                   `json:"stream"`
	Options  map[string]interface{} `json:"options,omitempty"`
	Tools    []ollamaTool           `json:"tools,omitempty"`
}

type ollamaMessage struct {
	Role       string           `json:"role"`
	Content    string           `json:"content"`
	Thinking   string           `json:"thinking,omitempty"`
	Name       string           `json:"name,omitempty"`
	ToolCallID string           `json:"tool_call_id,omitempty"`
	ToolCalls  []ollamaToolCall `json:"tool_calls,omitempty"`
}

type ollamaTool struct {
	Type     string             `json:"type"`
	Function ollamaToolFunction `json:"function"`
}

type ollamaToolFunction struct {
	Name        string                 `json:"name"`
	Description string                 `json:"description,omitempty"`
	Parameters  map[string]interface{} `json:"parameters,omitempty"`
}

type ollamaToolCall struct {
	ID       string             `json:"id,omitempty"`
	Type     string             `json:"type,omitempty"`
	Function ollamaFunctionCall `json:"function"`
}

type ollamaFunctionCall struct {
	Name      string                 `json:"name,omitempty"`
	Arguments map[string]interface{} `json:"arguments,omitempty"`
}

type chatResponse struct {
	Message         ollamaMessage `json:"message"`
	Response        string        `json:"response"`
	Done            bool          `json:"done"`
	DoneReason      string        `json:"done_reason"`
	Error           string        `json:"error"`
	PromptEvalCount int           `json:"prompt_eval_count"`
	EvalCount       int           `json:"eval_count"`
	Cost            *float64      `json:"cost,omitempty"`
	TotalCost       *float64      `json:"total_cost,omitempty"`
}

// mapDoneReason translates an Ollama done_reason into the OpenAI finish
// reason vocabulary used by StreamChunk.FinishReason. Unmappable reasons map
// to the empty string, matching the contract that absent reasons stay empty.
func mapDoneReason(reason string) string {
	switch reason {
	case "stop":
		return "stop"
	case "length":
		return "length"
	default:
		return ""
	}
}

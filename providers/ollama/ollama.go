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
	}, httpretry.DefaultConfig())
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
	return &llmhub.Response{
		Content: parts,
		Usage: llmhub.UsageMetadata{
			PromptTokens:     decoded.PromptEvalCount,
			CompletionTokens: decoded.EvalCount,
			TotalTokens:      decoded.PromptEvalCount + decoded.EvalCount,
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
	}, httpretry.DefaultConfig())
	if err != nil {
		return nil, err
	}

	ch := make(chan llmhub.StreamChunk)
	go func() {
		defer resp.Body.Close()
		defer close(ch)
		if resp.StatusCode >= 400 {
			body, _ := io.ReadAll(resp.Body)
			ch <- llmhub.StreamChunk{Err: fmt.Errorf("ollama: http %d: %s", resp.StatusCode, strings.TrimSpace(string(body))), Done: true}
			return
		}
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
				case ch <- llmhub.StreamChunk{Delta: text, ReasoningDelta: chunk.Message.Thinking}:
				}
			} else if chunk.Message.Thinking != "" {
				select {
				case <-ctx.Done():
					ch <- llmhub.StreamChunk{Err: ctx.Err(), Done: true}
					return
				case ch <- llmhub.StreamChunk{ReasoningDelta: chunk.Message.Thinking}:
				}
			}
			if chunk.Done {
				ch <- llmhub.StreamChunk{Done: true}
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
		text, err := flattenText(msg.Content)
		if err != nil {
			return nil, err
		}
		msgs = append(msgs, ollamaMessage{Role: string(msg.Role), Content: text})
	}
	req := chatRequest{
		Model:    cfg.Model,
		Messages: msgs,
		Stream:   stream,
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
}

type ollamaMessage struct {
	Role     string `json:"role"`
	Content  string `json:"content"`
	Thinking string `json:"thinking,omitempty"`
}

type chatResponse struct {
	Message         ollamaMessage `json:"message"`
	Response        string        `json:"response"`
	Done            bool          `json:"done"`
	Error           string        `json:"error"`
	PromptEvalCount int           `json:"prompt_eval_count"`
	EvalCount       int           `json:"eval_count"`
}

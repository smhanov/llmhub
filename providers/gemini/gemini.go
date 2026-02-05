package gemini

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"path"
	"strings"
	"time"

	"github.com/smhanov/llmhub"
	"github.com/smhanov/llmhub/internal/httpretry"
)

const (
	providerName      = "gemini"
	defaultBaseURL    = "https://generativelanguage.googleapis.com"
	defaultModel      = "gemini-1.5-flash"
	generateSuffix    = ":generateContent"
	streamSuffix      = ":streamGenerateContent"
	apiVersionSegment = "v1beta"
)

type Client struct {
	baseCfg llmhub.Config
}

func init() { llmhub.MustRegisterProvider(providerName, New) }

func New(apiKey string, opts ...llmhub.Option) (llmhub.Provider, error) {
	cfg := llmhub.NewConfig(opts...)
	if cfg.APIKey == "" {
		cfg.APIKey = apiKey
	}
	if cfg.APIKey == "" {
		return nil, fmt.Errorf("gemini: %w: api key is required", llmhub.ErrInvalidInput)
	}
	if cfg.BaseURL == "" {
		cfg.BaseURL = defaultBaseURL
	}
	if cfg.Model == "" {
		cfg.Model = defaultModel
	}
	if cfg.HTTPClient == nil {
		cfg.HTTPClient = &http.Client{Timeout: 2 * time.Minute}
	}
	return &Client{baseCfg: cfg}, nil
}

func (c *Client) Name() string { return providerName }

func (c *Client) Generate(ctx context.Context, prompt []*llmhub.Message, opts ...llmhub.Option) (*llmhub.Response, error) {
	cfg := c.mergeConfig(opts...)
	reqBody, err := buildRequestBody(prompt, cfg)
	if err != nil {
		return nil, err
	}
	endpoint, err := buildEndpoint(cfg, false)
	if err != nil {
		return nil, err
	}
	resp, err := httpretry.Do(ctx, cfg.HTTPClient, func() (*http.Request, error) {
		req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(reqBody))
		if err != nil {
			return nil, err
		}
		req.Header.Set("content-type", "application/json")
		for k, v := range cfg.Headers {
			req.Header.Set(k, v)
		}
		return req, nil
	}, httpretry.DefaultConfig())
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 400 {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("gemini: http %d: %s", resp.StatusCode, strings.TrimSpace(string(body)))
	}

	var decoded geminiResponse
	if err := json.NewDecoder(resp.Body).Decode(&decoded); err != nil {
		return nil, err
	}
	parts, err := convertCandidate(decoded.Candidates)
	if err != nil {
		return nil, err
	}
	usage := decoded.Usage
	return &llmhub.Response{
		Content: parts,
		Usage: llmhub.UsageMetadata{
			PromptTokens:     usage.PromptTokenCount,
			CompletionTokens: usage.CandidatesTokenCount,
			TotalTokens:      usage.TotalTokenCount,
		},
		Raw: decoded,
	}, nil
}

func (c *Client) Stream(ctx context.Context, prompt []*llmhub.Message, opts ...llmhub.Option) (<-chan llmhub.StreamChunk, error) {
	cfg := c.mergeConfig(opts...)
	reqBody, err := buildRequestBody(prompt, cfg)
	if err != nil {
		return nil, err
	}
	endpoint, err := buildEndpoint(cfg, true)
	if err != nil {
		return nil, err
	}
	resp, err := httpretry.Do(ctx, cfg.HTTPClient, func() (*http.Request, error) {
		req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(reqBody))
		if err != nil {
			return nil, err
		}
		req.Header.Set("content-type", "application/json")
		for k, v := range cfg.Headers {
			req.Header.Set(k, v)
		}
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
			ch <- llmhub.StreamChunk{Err: fmt.Errorf("gemini: http %d: %s", resp.StatusCode, strings.TrimSpace(string(body))), Done: true}
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
			var chunk geminiResponse
			if err := json.Unmarshal([]byte(line), &chunk); err != nil {
				ch <- llmhub.StreamChunk{Err: err, Done: true}
				return
			}
			text, err := extractText(chunk.Candidates)
			if err != nil {
				ch <- llmhub.StreamChunk{Err: err, Done: true}
				return
			}
			if text == "" {
				continue
			}
			select {
			case <-ctx.Done():
				ch <- llmhub.StreamChunk{Err: ctx.Err(), Done: true}
				return
			case ch <- llmhub.StreamChunk{Delta: text}:
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
	if cfg.APIKey == "" {
		cfg.APIKey = c.baseCfg.APIKey
	}
	if cfg.BaseURL == "" {
		cfg.BaseURL = c.baseCfg.BaseURL
	}
	return cfg
}

func buildEndpoint(cfg llmhub.Config, stream bool) (string, error) {
	base, err := url.Parse(cfg.BaseURL)
	if err != nil {
		return "", err
	}
	suffix := generateSuffix
	if stream {
		suffix = streamSuffix
	}
	base.Path = path.Join(strings.TrimSuffix(base.Path, "/"), apiVersionSegment, "models", cfg.Model) + suffix
	q := base.Query()
	q.Set("key", cfg.APIKey)
	base.RawQuery = q.Encode()
	return base.String(), nil
}

func buildRequestBody(prompt []*llmhub.Message, cfg llmhub.Config) ([]byte, error) {
	contents, system, err := convertMessages(prompt)
	if err != nil {
		return nil, err
	}
	req := geminiRequest{
		Contents: contents,
	}
	if system != nil {
		req.SystemInstruction = system
	}
	if cfg.Temperature != 0 || cfg.MaxTokens != 0 {
		req.GenerationConfig = &generationConfig{}
		if cfg.Temperature != 0 {
			req.GenerationConfig.Temperature = cfg.Temperature
		}
		if cfg.MaxTokens != 0 {
			req.GenerationConfig.MaxOutputTokens = cfg.MaxTokens
		}
	}
	return json.Marshal(req)
}

func convertMessages(prompt []*llmhub.Message) ([]geminiContent, *geminiContent, error) {
	contents := make([]geminiContent, 0, len(prompt))
	var systemParts []geminiPart
	for _, msg := range prompt {
		if msg == nil {
			continue
		}
		switch msg.Role {
		case llmhub.RoleSystem:
			parts, err := convertParts(msg.Content)
			if err != nil {
				return nil, nil, err
			}
			systemParts = append(systemParts, parts...)
		default:
			parts, err := convertParts(msg.Content)
			if err != nil {
				return nil, nil, err
			}
			if len(parts) == 0 {
				continue
			}
			contents = append(contents, geminiContent{Role: string(msg.Role), Parts: parts})
		}
	}
	var system *geminiContent
	if len(systemParts) > 0 {
		system = &geminiContent{Parts: systemParts}
	}
	return contents, system, nil
}

func convertParts(parts []llmhub.ContentPart) ([]geminiPart, error) {
	converted := make([]geminiPart, 0, len(parts))
	for _, part := range parts {
		switch v := part.(type) {
		case *llmhub.TextContent:
			converted = append(converted, geminiPart{Text: v.Text})
		case *llmhub.ImageContent:
			inline, err := toInlineData(v.URL)
			if err != nil {
				// Fall back to file reference if not inline.
				converted = append(converted, geminiPart{FileData: &fileData{FileURI: v.URL}})
			} else {
				converted = append(converted, geminiPart{InlineData: inline})
			}
		default:
			return nil, fmt.Errorf("gemini: unsupported content type %T", v)
		}
	}
	return converted, nil
}

func toInlineData(urlStr string) (*inlineData, error) {
	if !strings.HasPrefix(urlStr, "data:") {
		return nil, fmt.Errorf("not inline data")
	}
	payload := strings.TrimPrefix(urlStr, "data:")
	parts := strings.SplitN(payload, ",", 2)
	if len(parts) != 2 {
		return nil, fmt.Errorf("gemini: malformed data url")
	}
	meta, data := parts[0], parts[1]
	mime := "application/octet-stream"
	if meta != "" {
		fields := strings.Split(meta, ";")
		mime = fields[0]
		encoded := false
		for _, f := range fields[1:] {
			if f == "base64" {
				encoded = true
				break
			}
		}
		if !encoded {
			return nil, fmt.Errorf("gemini: data urls must be base64 encoded")
		}
	}
	if data == "" {
		return nil, fmt.Errorf("gemini: data url empty")
	}
	return &inlineData{MimeType: mime, Data: data}, nil
}

func convertCandidate(candidates []candidate) ([]llmhub.ContentPart, error) {
	if len(candidates) == 0 {
		return nil, errors.New("gemini: no candidates returned")
	}
	return convertGeminiParts(candidates[0].Content.Parts)
}

func convertGeminiParts(parts []geminiPart) ([]llmhub.ContentPart, error) {
	out := make([]llmhub.ContentPart, 0, len(parts))
	for _, part := range parts {
		switch {
		case part.Text != "":
			out = append(out, llmhub.Text(part.Text))
		case part.InlineData != nil:
			dataURL := fmt.Sprintf("data:%s;base64,%s", part.InlineData.MimeType, part.InlineData.Data)
			out = append(out, &llmhub.ImageContent{URL: dataURL})
		case part.FileData != nil:
			out = append(out, &llmhub.ImageContent{URL: part.FileData.FileURI})
		}
	}
	return out, nil
}

func extractText(candidates []candidate) (string, error) {
	parts, err := convertCandidate(candidates)
	if err != nil {
		return "", err
	}
	var b strings.Builder
	for _, part := range parts {
		if text, ok := part.(*llmhub.TextContent); ok {
			b.WriteString(text.Text)
		}
	}
	return b.String(), nil
}

type geminiRequest struct {
	Contents          []geminiContent   `json:"contents"`
	SystemInstruction *geminiContent    `json:"system_instruction,omitempty"`
	GenerationConfig  *generationConfig `json:"generation_config,omitempty"`
}

type geminiContent struct {
	Role  string       `json:"role,omitempty"`
	Parts []geminiPart `json:"parts"`
}

type geminiPart struct {
	Text       string      `json:"text,omitempty"`
	InlineData *inlineData `json:"inline_data,omitempty"`
	FileData   *fileData   `json:"file_data,omitempty"`
}

type inlineData struct {
	MimeType string `json:"mime_type"`
	Data     string `json:"data"`
}

type fileData struct {
	FileURI string `json:"file_uri"`
}

type generationConfig struct {
	Temperature     float64 `json:"temperature,omitempty"`
	MaxOutputTokens int     `json:"max_output_tokens,omitempty"`
}

type geminiResponse struct {
	Candidates []candidate   `json:"candidates"`
	Usage      usageMetadata `json:"usage_metadata"`
}

type candidate struct {
	Content geminiContent `json:"content"`
}

type usageMetadata struct {
	PromptTokenCount     int `json:"prompt_token_count"`
	CandidatesTokenCount int `json:"candidates_token_count"`
	TotalTokenCount      int `json:"total_token_count"`
}

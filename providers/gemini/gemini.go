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
	}, httpretry.FromLLMHub(cfg))
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
	var cost float64
	if usage.Cost != nil {
		cost = *usage.Cost
	} else if usage.TotalCost != nil {
		cost = *usage.TotalCost
	}
	return &llmhub.Response{
		Content: parts,
		Usage: llmhub.UsageMetadata{
			PromptTokens:     usage.PromptTokenCount,
			CompletionTokens: usage.CandidatesTokenCount,
			TotalTokens:      usage.TotalTokenCount,
			Cost:             cost,
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
	}, httpretry.FromLLMHub(cfg))
	if err != nil {
		return nil, err
	}
	if resp.StatusCode >= 400 {
		defer resp.Body.Close()
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("gemini: http %d: %s", resp.StatusCode, strings.TrimSpace(string(body)))
	}

	ch := make(chan llmhub.StreamChunk)
	go func() {
		defer resp.Body.Close()
		defer close(ch)
		scanner := bufio.NewScanner(resp.Body)
		buf := make([]byte, 0, 64*1024)
		scanner.Buffer(buf, 4*1024*1024)
		var pending strings.Builder
		for scanner.Scan() {
			line := strings.TrimSpace(scanner.Text())
			if line == "" {
				continue
			}
			if strings.HasPrefix(line, "data:") {
				line = strings.TrimSpace(strings.TrimPrefix(line, "data:"))
			}
			if line == "" || line == "[DONE]" {
				continue
			}
			pending.WriteString(line)
			payload := pending.String()
			var parsed []geminiResponse
			var chunk geminiResponse
			if err := json.Unmarshal([]byte(payload), &chunk); err != nil {
				if strings.Contains(err.Error(), "unexpected end of JSON input") {
					continue
				}
				if err := json.Unmarshal([]byte(payload), &parsed); err != nil {
					ch <- llmhub.StreamChunk{Err: err, Done: true}
					return
				}
			} else {
				parsed = []geminiResponse{chunk}
			}
			pending.Reset()
			for _, current := range parsed {
				text, reasoning, toolCalls, finishReason, err := extractContent(current.Candidates)
				if err != nil {
					ch <- llmhub.StreamChunk{Err: err, Done: true}
					return
				}
				if text == "" && reasoning == "" && len(toolCalls) == 0 && finishReason == "" {
					continue
				}
				select {
				case <-ctx.Done():
					ch <- llmhub.StreamChunk{Err: ctx.Err(), Done: true}
					return
				case ch <- llmhub.StreamChunk{Delta: text, ReasoningDelta: reasoning, ToolCalls: toolCalls, FinishReason: finishReason}:
				}
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
	if cfg.Temperature != 0 || cfg.MaxTokens != 0 || len(cfg.ResponseModalities) > 0 {
		req.GenerationConfig = &generationConfig{}
		if cfg.Temperature != 0 {
			req.GenerationConfig.Temperature = cfg.Temperature
		}
		if cfg.MaxTokens != 0 {
			req.GenerationConfig.MaxOutputTokens = cfg.MaxTokens
		}
		if len(cfg.ResponseModalities) > 0 {
			req.GenerationConfig.ResponseModalities = cfg.ResponseModalities
		}
	}
	if cfg.EnableWebSearch {
		req.Tools = append(req.Tools, geminiTool{GoogleSearch: &googleSearchTool{}})
	}
	if len(cfg.Tools) > 0 {
		req.Tools = append(req.Tools, geminiTool{FunctionDeclarations: convertTools(cfg.Tools)})
	}
	if cfg.ToolChoice != nil {
		req.ToolConfig = convertToolChoice(*cfg.ToolChoice)
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
		case llmhub.RoleTool:
			part, err := convertToolResult(msg)
			if err != nil {
				return nil, nil, err
			}
			contents = append(contents, geminiContent{Role: "user", Parts: []geminiPart{part}})
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
		case *llmhub.ReasoningContent:
			converted = append(converted, geminiPart{Text: v.Text, Thought: true})
		case *llmhub.ImageContent:
			inline, err := toInlineData(v.URL)
			if err != nil {
				// Fall back to file reference if not inline.
				converted = append(converted, geminiPart{FileData: &fileData{FileURI: v.URL}})
			} else {
				converted = append(converted, geminiPart{InlineData: inline})
			}
		case *llmhub.ToolCallContent:
			args, err := rawJSON(v.Arguments)
			if err != nil {
				return nil, err
			}
			converted = append(converted, geminiPart{FunctionCall: &functionCall{Name: v.Name, Args: args}})
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
		case part.Text != "" && part.Thought:
			out = append(out, llmhub.Reasoning(part.Text))
		case part.Text != "":
			out = append(out, llmhub.Text(part.Text))
		case part.InlineData != nil:
			dataURL := fmt.Sprintf("data:%s;base64,%s", part.InlineData.MimeType, part.InlineData.Data)
			out = append(out, &llmhub.ImageContent{URL: dataURL})
		case part.FileData != nil:
			out = append(out, &llmhub.ImageContent{URL: part.FileData.FileURI})
		case part.FunctionCall != nil:
			out = append(out, llmhub.ToolCall("", part.FunctionCall.Name, string(part.FunctionCall.Args)))
		}
	}
	return out, nil
}

func extractContent(candidates []candidate) (string, string, []*llmhub.ToolCallContent, string, error) {
	parts, err := convertCandidate(candidates)
	if err != nil {
		return "", "", nil, "", err
	}
	var textBuilder strings.Builder
	var reasoningBuilder strings.Builder
	var toolCalls []*llmhub.ToolCallContent
	for _, part := range parts {
		switch value := part.(type) {
		case *llmhub.TextContent:
			textBuilder.WriteString(value.Text)
		case *llmhub.ReasoningContent:
			reasoningBuilder.WriteString(value.Text)
		case *llmhub.ToolCallContent:
			toolCalls = append(toolCalls, value)
		}
	}
	finishReason := ""
	if len(candidates) > 0 {
		finishReason = mapFinishReason(candidates[0].FinishReason)
	}
	return textBuilder.String(), reasoningBuilder.String(), toolCalls, finishReason, nil
}

// mapFinishReason translates a Gemini finish reason into the OpenAI finish
// reason vocabulary used by StreamChunk.FinishReason. Unmappable reasons map
// to the empty string, matching the contract that absent reasons stay empty.
func mapFinishReason(reason string) string {
	switch reason {
	case "STOP":
		return "stop"
	case "MAX_TOKENS":
		return "length"
	case "MALFORMED_FUNCTION_CALL":
		return "tool_calls"
	default:
		return ""
	}
}

func convertTools(tools []llmhub.Tool) []functionDeclaration {
	converted := make([]functionDeclaration, 0, len(tools))
	for _, tool := range tools {
		converted = append(converted, functionDeclaration{
			Name:        tool.Name,
			Description: tool.Description,
			Parameters:  tool.Parameters,
		})
	}
	return converted
}

func convertToolChoice(choice llmhub.ToolChoice) *geminiToolConfig {
	config := &geminiToolConfig{FunctionCallingConfig: &functionCallingConfig{}}
	switch choice.Mode {
	case llmhub.ToolChoiceNone:
		config.FunctionCallingConfig.Mode = "NONE"
	case llmhub.ToolChoiceRequired:
		config.FunctionCallingConfig.Mode = "ANY"
	case llmhub.ToolChoiceNamed:
		config.FunctionCallingConfig.Mode = "ANY"
		config.FunctionCallingConfig.AllowedFunctionNames = []string{choice.Name}
	default:
		config.FunctionCallingConfig.Mode = "AUTO"
	}
	return config
}

func convertToolResult(msg *llmhub.Message) (geminiPart, error) {
	text, err := flattenText(msg.Content)
	if err != nil {
		return geminiPart{}, err
	}
	response := map[string]interface{}{"result": text}
	if json.Valid([]byte(text)) {
		var parsed map[string]interface{}
		if err := json.Unmarshal([]byte(text), &parsed); err == nil && parsed != nil {
			response = parsed
		}
	}
	return geminiPart{
		FunctionResponse: &functionResponse{
			Name:     metaValue(msg, "name"),
			Response: response,
		},
	}, nil
}

func flattenText(parts []llmhub.ContentPart) (string, error) {
	var b strings.Builder
	for _, part := range parts {
		text, ok := part.(*llmhub.TextContent)
		if !ok {
			return "", fmt.Errorf("gemini: tool results must be text content")
		}
		b.WriteString(text.Text)
	}
	return b.String(), nil
}

func rawJSON(arguments string) (json.RawMessage, error) {
	if strings.TrimSpace(arguments) == "" {
		return json.RawMessage(`{}`), nil
	}
	if !json.Valid([]byte(arguments)) {
		return nil, fmt.Errorf("gemini: tool call arguments must be valid JSON")
	}
	return json.RawMessage(arguments), nil
}

func metaValue(msg *llmhub.Message, key string) string {
	if msg == nil || msg.Meta == nil {
		return ""
	}
	return msg.Meta[key]
}

type geminiRequest struct {
	Contents          []geminiContent   `json:"contents"`
	SystemInstruction *geminiContent    `json:"systemInstruction,omitempty"`
	GenerationConfig  *generationConfig `json:"generationConfig,omitempty"`
	Tools             []geminiTool      `json:"tools,omitempty"`
	ToolConfig        *geminiToolConfig `json:"toolConfig,omitempty"`
}

type geminiTool struct {
	GoogleSearch         *googleSearchTool     `json:"googleSearch,omitempty"`
	FunctionDeclarations []functionDeclaration `json:"functionDeclarations,omitempty"`
}

type googleSearchTool struct{}

type functionDeclaration struct {
	Name        string                 `json:"name"`
	Description string                 `json:"description,omitempty"`
	Parameters  map[string]interface{} `json:"parameters,omitempty"`
}

type geminiToolConfig struct {
	FunctionCallingConfig *functionCallingConfig `json:"functionCallingConfig,omitempty"`
}

type functionCallingConfig struct {
	Mode                 string   `json:"mode,omitempty"`
	AllowedFunctionNames []string `json:"allowedFunctionNames,omitempty"`
}

type geminiContent struct {
	Role  string       `json:"role,omitempty"`
	Parts []geminiPart `json:"parts"`
}

type geminiPart struct {
	Text             string            `json:"text,omitempty"`
	Thought          bool              `json:"thought,omitempty"`
	InlineData       *inlineData       `json:"inlineData,omitempty"`
	FileData         *fileData         `json:"fileData,omitempty"`
	FunctionCall     *functionCall     `json:"functionCall,omitempty"`
	FunctionResponse *functionResponse `json:"functionResponse,omitempty"`
}

type functionCall struct {
	Name string          `json:"name"`
	Args json.RawMessage `json:"args,omitempty"`
}

type functionResponse struct {
	Name     string                 `json:"name"`
	Response map[string]interface{} `json:"response"`
}

type inlineData struct {
	MimeType string `json:"mimeType"`
	Data     string `json:"data"`
}

type fileData struct {
	FileURI string `json:"fileUri"`
}

type generationConfig struct {
	Temperature        float64  `json:"temperature,omitempty"`
	MaxOutputTokens    int      `json:"maxOutputTokens,omitempty"`
	ResponseModalities []string `json:"responseModalities,omitempty"`
}

type geminiResponse struct {
	Candidates []candidate   `json:"candidates"`
	Usage      usageMetadata `json:"usageMetadata"`
}

type candidate struct {
	Content      geminiContent `json:"content"`
	FinishReason string        `json:"finishReason,omitempty"`
}

type usageMetadata struct {
	PromptTokenCount     int      `json:"promptTokenCount"`
	CandidatesTokenCount int      `json:"candidatesTokenCount"`
	TotalTokenCount      int      `json:"totalTokenCount"`
	Cost                 *float64 `json:"cost,omitempty"`
	TotalCost            *float64 `json:"totalCost,omitempty"`
}

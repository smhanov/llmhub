package llmhub

import (
	"strings"
	"time"
)

// Role represents the speaker for a given message.
type Role string

const (
	RoleSystem    Role = "system"
	RoleUser      Role = "user"
	RoleAssistant Role = "assistant"
	RoleTool      Role = "tool"
)

// ContentPart is implemented by any structure that can be included inside a message.
type ContentPart interface {
	Type() string
}

// Message represents one turn in a conversation with the provider and can mix modalities.
type Message struct {
	Role    Role
	Content []ContentPart
	Meta    map[string]string
}

// Append adds one or more content parts to the message.
func (m *Message) Append(parts ...ContentPart) {
	m.Content = append(m.Content, parts...)
}

// NewMessage constructs a message with the provided role and content parts.
func NewMessage(role Role, parts ...ContentPart) *Message {
	return &Message{Role: role, Content: append([]ContentPart(nil), parts...)}
}

// NewSystemMessage returns a message authored by the system.
func NewSystemMessage(parts ...ContentPart) *Message {
	return NewMessage(RoleSystem, parts...)
}

// NewUserMessage returns a message authored by the end-user.
func NewUserMessage(parts ...ContentPart) *Message {
	return NewMessage(RoleUser, parts...)
}

// NewAssistantMessage returns a message authored by the assistant.
func NewAssistantMessage(parts ...ContentPart) *Message {
	return NewMessage(RoleAssistant, parts...)
}

// NewToolMessage returns a message authored by a tool.
func NewToolMessage(parts ...ContentPart) *Message {
	return NewMessage(RoleTool, parts...)
}

// NewToolResultMessage returns a message containing the result for a tool call.
func NewToolResultMessage(toolCallID, name string, parts ...ContentPart) *Message {
	msg := NewToolMessage(parts...)
	msg.Meta = map[string]string{
		"tool_call_id": toolCallID,
		"name":         name,
	}
	return msg
}

// TextContent represents free-form text.
type TextContent struct {
	Text string
}

// Type identifies the piece as text.
func (t *TextContent) Type() string { return "text" }

// Text is a helper constructor for a text part.
func Text(s string) *TextContent { return &TextContent{Text: s} }

// ReasoningContent represents model-internal reasoning or thinking text when providers expose it.
type ReasoningContent struct {
	Text string
}

// Type identifies the piece as reasoning.
func (r *ReasoningContent) Type() string { return "reasoning" }

// Reasoning is a helper constructor for a reasoning part.
func Reasoning(s string) *ReasoningContent { return &ReasoningContent{Text: s} }

// ImageContent represents a reference to an image by URL or base64 payload.
type ImageContent struct {
	URL    string
	Detail string // optional granularity instruction used by some providers
}

// Type identifies the piece as an image.
func (i *ImageContent) Type() string { return "image" }

// Image is a helper constructor for an image part.
func Image(url string) *ImageContent { return &ImageContent{URL: url} }

// Tool describes a callable function the model may request.
type Tool struct {
	Name        string
	Description string
	Parameters  map[string]interface{}
}

// NewTool constructs a callable tool definition.
func NewTool(name, description string, parameters map[string]interface{}) Tool {
	return Tool{Name: name, Description: description, Parameters: parameters}
}

// ToolChoiceMode controls how providers should use supplied tools.
type ToolChoiceMode string

const (
	ToolChoiceAuto     ToolChoiceMode = "auto"
	ToolChoiceNone     ToolChoiceMode = "none"
	ToolChoiceRequired ToolChoiceMode = "required"
	ToolChoiceNamed    ToolChoiceMode = "named"
)

// ToolChoice controls whether the model may, must, or must not call tools.
type ToolChoice struct {
	Mode ToolChoiceMode
	Name string
}

// ThinkingLevel controls how much internal reasoning a model performs
// before responding. It is a shared abstraction: each provider maps it to
// its native thinking/effort control, and providers without one ignore it.
// Unset (empty string) omits the control everywhere, using the model default.
//
// Per-provider mapping:
//
//	Gemini:     maps to generationConfig.thinkingConfig.thinkingLevel on
//	            Gemini 3.x+ and 4.x models only (minimal/low/medium/high;
//	            xhigh is clamped to high). 2.5-series models reject
//	            thinkingLevel (they use the deprecated thinkingBudget
//	            control, not sent by llmhub), and minimal itself is
//	            rejected by some 3.x models (e.g. gemini-3.8-flash)
//	OpenAI-lane (openai, xai, openrouter, zai): top-level reasoning_effort
//	            (minimal/low/medium/high/xhigh, model-dependent)
//	Anthropic:  output_config.effort, modern adaptive path, requires a
//	            4.6+ model (minimal maps to low; low/medium/high/xhigh 1:1)
//	Ollama:     top-level think (minimal maps to false, low/medium/high map
//	            to "low"/"medium"/"high", xhigh is clamped to "high";
//	            unknown level strings fall back to the model default)
type ThinkingLevel string

const (
	ThinkingLevelMinimal ThinkingLevel = "minimal"
	ThinkingLevelLow     ThinkingLevel = "low"
	ThinkingLevelMedium  ThinkingLevel = "medium"
	ThinkingLevelHigh    ThinkingLevel = "high"
	ThinkingLevelXHigh   ThinkingLevel = "xhigh"
)

// AutoToolChoice lets the model choose whether to call tools.
func AutoToolChoice() ToolChoice { return ToolChoice{Mode: ToolChoiceAuto} }

// NoToolChoice prevents the model from calling tools.
func NoToolChoice() ToolChoice { return ToolChoice{Mode: ToolChoiceNone} }

// RequiredToolChoice requires the model to call at least one tool.
func RequiredToolChoice() ToolChoice { return ToolChoice{Mode: ToolChoiceRequired} }

// NamedToolChoice requires the model to call the named tool.
func NamedToolChoice(name string) ToolChoice { return ToolChoice{Mode: ToolChoiceNamed, Name: name} }

// ToolCallContent represents a model-requested call to a tool.
type ToolCallContent struct {
	Index     int
	ID        string
	Name      string
	Arguments string
}

// Type identifies the piece as a tool call.
func (t *ToolCallContent) Type() string { return "tool_call" }

// ToolCall is a helper constructor for a tool-call content part.
func ToolCall(id, name, arguments string) *ToolCallContent {
	return &ToolCallContent{ID: id, Name: name, Arguments: arguments}
}

// ToolCallWithIndex is a helper constructor for a streaming tool-call delta.
func ToolCallWithIndex(index int, id, name, arguments string) *ToolCallContent {
	return &ToolCallContent{Index: index, ID: id, Name: name, Arguments: arguments}
}

// UsageMetadata captures token consumption and cost information reported by providers.
type UsageMetadata struct {
	PromptTokens        int
	CompletionTokens    int
	TotalTokens         int
	CacheReadTokens     int
	CacheCreationTokens int
	ReasoningTokens     int
	Cost                float64 // Estimated cost in US dollars based on configured per-million-token rates.
}

// Response contains the normalized result returned from a provider.
type Response struct {
	ID      string
	Content []ContentPart
	Usage   UsageMetadata
	// Timing carries built-in latency telemetry (Connected, FirstByte,
	// Total) for non-streaming requests. It is nil only for providers
	// invoked without the Client wrapper (direct Provider calls).
	Timing *TimingMetrics
	Raw    interface{}
}

// Text concatenates the textual segments of the response for the common use case where only text matters.
func (r *Response) Text() string {
	if r == nil {
		return ""
	}
	var b strings.Builder
	for _, part := range r.Content {
		if tc, ok := part.(*TextContent); ok {
			b.WriteString(tc.Text)
		}
	}
	return b.String()
}

// ReasoningText concatenates reasoning segments exposed by providers.
func (r *Response) ReasoningText() string {
	if r == nil {
		return ""
	}
	var b strings.Builder
	for _, part := range r.Content {
		if rc, ok := part.(*ReasoningContent); ok {
			b.WriteString(rc.Text)
		}
	}
	return b.String()
}

// ToolCalls returns all normalized tool calls requested in the response.
func (r *Response) ToolCalls() []*ToolCallContent {
	if r == nil {
		return nil
	}
	var calls []*ToolCallContent
	for _, part := range r.Content {
		if call, ok := part.(*ToolCallContent); ok {
			calls = append(calls, call)
		}
	}
	return calls
}

// TimingMetrics captures built-in latency telemetry for a single request,
// so consumer applications do not have to hand-roll timestamping and
// chunk-tracking logic. All Duration fields are measured from StartedAt.
// A zero duration always means "not available", never "instant".
type TimingMetrics struct {
	// StartedAt is the moment Stream or Generate was invoked.
	StartedAt time.Time
	// Connected is the time until the HTTP connection was established
	// (httptrace GotConn), including any TLS handshake. With connection
	// reuse it is near zero. Zero when the transport did not report it.
	Connected time.Duration
	// FirstByte is the time until the upstream sent its first response
	// byte (httptrace GotFirstResponseByte). When the request was retried
	// (e.g. on 429) it reflects the successful attempt. Zero when the
	// transport did not report it.
	FirstByte time.Duration
	// FirstToken is the time until the first content-bearing stream chunk
	// (text, reasoning, or tool-call delta) was delivered — the classic
	// Time To First Token. Usage-only trailing frames do not count. Zero
	// until a content chunk arrives.
	FirstToken time.Duration
	// Total is the full request/stream duration. It is set on the terminal
	// stream chunk (Done or Err) and on Generate responses; it is zero on
	// earlier chunks.
	Total time.Duration
	// Chunks is the number of content-bearing chunks observed so far. It
	// grows on each content chunk and is final on the terminal chunk.
	Chunks int
}

// StreamChunk represents a partial streaming response.
type StreamChunk struct {
	// ID is the upstream completion identifier when the provider reports one
	// (OpenAI-compatible lanes and Anthropic). It is best-effort telemetry and
	// may be empty for lanes that do not report a stable stream id.
	ID string
	// Delta is the incremental text produced since the previous chunk.
	Delta string
	// ReasoningDelta is the incremental reasoning text produced since the previous chunk.
	ReasoningDelta string
	// ToolCalls carries streamed tool call deltas.
	ToolCalls []*ToolCallContent
	// Usage carries token and cost accounting when the upstream reports it.
	Usage *UsageMetadata
	// FinishReason describes how the response ended, using the OpenAI
	// vocabulary (stop, length, tool_calls). It is empty when the upstream
	// reports no reason or the reason cannot be mapped. It is typically set on
	// the final content frame (not on the terminal Done frame).
	FinishReason string
	// Timing carries llmhub's built-in latency telemetry as a snapshot taken
	// when this chunk was produced. It is nil only for providers invoked
	// without the Client wrapper (direct Provider calls).
	Timing *TimingMetrics
	Done   bool
	Err    error
}

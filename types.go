package llmhub

import "strings"

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

// TextContent represents free-form text.
type TextContent struct {
	Text string
}

// Type identifies the piece as text.
func (t *TextContent) Type() string { return "text" }

// Text is a helper constructor for a text part.
func Text(s string) *TextContent { return &TextContent{Text: s} }

// ImageContent represents a reference to an image by URL or base64 payload.
type ImageContent struct {
	URL    string
	Detail string // optional granularity instruction used by some providers
}

// Type identifies the piece as an image.
func (i *ImageContent) Type() string { return "image" }

// Image is a helper constructor for an image part.
func Image(url string) *ImageContent { return &ImageContent{URL: url} }

// UsageMetadata captures token consumption and cost information reported by providers.
type UsageMetadata struct {
	PromptTokens     int
	CompletionTokens int
	TotalTokens      int
	Cost             float64 // Estimated cost in US dollars based on configured per-million-token rates.
}

// Response contains the normalized result returned from a provider.
type Response struct {
	ID      string
	Content []ContentPart
	Usage   UsageMetadata
	Raw     interface{}
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

// StreamChunk represents a partial streaming response.
type StreamChunk struct {
	Delta string
	Done  bool
	Err   error
}

package openaichat

import (
	"encoding/json"
	"testing"

	"github.com/smhanov/llmhub"
)

// TestConvertToAPIMessage_ReplayedReasoningUsesDedicatedField verifies that a
// prior assistant turn's reasoning content is replayed as the OpenAI
// `reasoning_content` message field (DeepSeek-style), NOT merged into the
// plain `content` string. DeepSeek-reasoner rejects/ignores reasoning merged
// into content, and the proxy's multi-turn reasoning echo contract depends on
// the dedicated field round-tripping.
func TestConvertToAPIMessage_ReplayedReasoningUsesDedicatedField(t *testing.T) {
	msg := &llmhub.Message{
		Role: llmhub.RoleAssistant,
		Content: []llmhub.ContentPart{
			&llmhub.ReasoningContent{Text: "The user is asking for sqrt(144). 12 * 12 = 144."},
			&llmhub.TextContent{Text: "12"},
		},
	}

	got, err := ConvertToAPIMessage(msg)
	if err != nil {
		t.Fatalf("ConvertToAPIMessage failed: %v", err)
	}

	if got.ReasoningContent != "The user is asking for sqrt(144). 12 * 12 = 144." {
		t.Errorf("expected ReasoningContent to carry the reasoning text, got %q", got.ReasoningContent)
	}
	if got.Content != "12" {
		t.Errorf("expected Content to be only the answer text, got %v", got.Content)
	}

	wire := marshalWire(t, got)
	if wire["reasoning_content"] != "The user is asking for sqrt(144). 12 * 12 = 144." {
		t.Errorf("expected reasoning_content on the wire, got: %v", wire["reasoning_content"])
	}
	if wire["content"] != "12" {
		t.Errorf("expected content %q on the wire, got: %v", "12", wire["content"])
	}
}

// TestConvertToAPIMessage_ReasoningOnlyAssistantTurn checks the single-part
// reasoning-only assistant message also lands in reasoning_content, not
// content (the len(Content)==1 shortcut previously flattened it).
func TestConvertToAPIMessage_ReasoningOnlyAssistantTurn(t *testing.T) {
	msg := &llmhub.Message{
		Role: llmhub.RoleAssistant,
		Content: []llmhub.ContentPart{
			&llmhub.ReasoningContent{Text: "thinking hard"},
		},
	}

	got, err := ConvertToAPIMessage(msg)
	if err != nil {
		t.Fatalf("ConvertToAPIMessage failed: %v", err)
	}
	if got.ReasoningContent != "thinking hard" {
		t.Errorf("expected ReasoningContent %q, got %q", "thinking hard", got.ReasoningContent)
	}
	if got.Content != "" {
		t.Errorf("expected empty Content for reasoning-only turn, got %v", got.Content)
	}

	wire := marshalWire(t, got)
	if wire["reasoning_content"] != "thinking hard" {
		t.Errorf("expected reasoning_content on the wire, got: %v", wire["reasoning_content"])
	}
	if content, ok := wire["content"]; !ok || content != "" {
		t.Errorf("expected empty string content on the wire, got: %v", wire["content"])
	}
}

func TestConvertToAPIMessage_ReasoningWithToolCalls(t *testing.T) {
	msg := &llmhub.Message{
		Role: llmhub.RoleAssistant,
		Content: []llmhub.ContentPart{
			&llmhub.ReasoningContent{Text: "need weather"},
			&llmhub.TextContent{Text: "checking"},
			llmhub.ToolCall("call-1", "get_weather", `{"city":"Toronto"}`),
		},
	}

	got, err := ConvertToAPIMessage(msg)
	if err != nil {
		t.Fatalf("ConvertToAPIMessage failed: %v", err)
	}
	if got.ReasoningContent != "need weather" {
		t.Errorf("expected ReasoningContent %q, got %q", "need weather", got.ReasoningContent)
	}
	if got.Content != "checking" {
		t.Errorf("expected Content %q, got %v", "checking", got.Content)
	}
	if len(got.ToolCalls) != 1 || got.ToolCalls[0].ID != "call-1" {
		t.Errorf("expected one tool call, got %+v", got.ToolCalls)
	}

	wire := marshalWire(t, got)
	if wire["reasoning_content"] != "need weather" {
		t.Errorf("expected reasoning_content on the wire, got: %v", wire["reasoning_content"])
	}
	if wire["content"] != "checking" {
		t.Errorf("expected content %q on the wire, got: %v", "checking", wire["content"])
	}
}

func marshalWire(t *testing.T, msg ChatMessage) map[string]any {
	t.Helper()
	raw, err := json.Marshal(msg)
	if err != nil {
		t.Fatalf("marshal failed: %v", err)
	}
	var wire map[string]any
	if err := json.Unmarshal(raw, &wire); err != nil {
		t.Fatalf("unmarshal failed: %v", err)
	}
	return wire
}

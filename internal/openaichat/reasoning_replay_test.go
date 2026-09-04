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

	// Wire shape: JSON must contain reasoning_content separate from content.
	raw, err := json.Marshal(got)
	if err != nil {
		t.Fatalf("marshal failed: %v", err)
	}
	var wire map[string]any
	if err := json.Unmarshal(raw, &wire); err != nil {
		t.Fatalf("unmarshal failed: %v", err)
	}
	if _, ok := wire["reasoning_content"]; !ok {
		t.Errorf("expected reasoning_content field on the wire, got: %s", raw)
	}
	if content, ok := wire["content"].(string); ok && content == "The user is asking for sqrt(144). 12 * 12 = 144.12" {
		t.Errorf("reasoning leaked into content: %v", wire["content"])
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
}

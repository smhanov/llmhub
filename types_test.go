package llmhub

import "testing"

func TestResponseText(t *testing.T) {
	resp := &Response{
		Content: []ContentPart{
			Text("hello "),
			Reasoning("internal"),
			&ImageContent{URL: "https://example.com/image.png"},
			Text("world"),
		},
	}
	if got := resp.Text(); got != "hello world" {
		t.Fatalf("expected text to match, got %q", got)
	}
}

func TestResponseReasoningText(t *testing.T) {
	resp := &Response{
		Content: []ContentPart{
			Text("visible"),
			Reasoning("step 1 "),
			Reasoning("step 2"),
		},
	}
	if got := resp.ReasoningText(); got != "step 1 step 2" {
		t.Fatalf("expected reasoning text to match, got %q", got)
	}
}

func TestMessageHelpers(t *testing.T) {
	msg := NewUserMessage(Text("test"))
	if msg.Role != RoleUser {
		t.Fatalf("expected user role")
	}
	if len(msg.Content) != 1 {
		t.Fatalf("expected one content part")
	}
	msg.Append(Text("more"))
	if len(msg.Content) != 2 {
		t.Fatalf("expected two content parts after append")
	}
}

func TestToolHelpers(t *testing.T) {
	call := ToolCall("call-1", "lookup", `{"id":"42"}`)
	indexed := ToolCallWithIndex(1, "call-2", "search", `{"q":"x"}`)
	resp := &Response{Content: []ContentPart{Text("checking"), call, indexed}}
	calls := resp.ToolCalls()
	if len(calls) != 2 || calls[0].Name != "lookup" || calls[0].Arguments != `{"id":"42"}` {
		t.Fatalf("unexpected tool calls: %+v", calls)
	}
	if calls[1].Index != 1 || calls[1].ID != "call-2" {
		t.Fatalf("unexpected indexed tool call: %+v", calls[1])
	}

	msg := NewToolResultMessage("call-1", "lookup", Text(`{"ok":true}`))
	if msg.Role != RoleTool || msg.Meta["tool_call_id"] != "call-1" || msg.Meta["name"] != "lookup" {
		t.Fatalf("unexpected tool result message: %+v", msg)
	}
}

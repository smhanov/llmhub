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

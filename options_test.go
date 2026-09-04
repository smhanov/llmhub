package llmhub

import (
	"context"
	"encoding/json"
	"net/http"
	"testing"

	"github.com/smhanov/llmhub/auth"
)

func TestNewConfigDefaults(t *testing.T) {
	cfg := NewConfig()
	if cfg.Temperature != defaultTemperature {
		t.Fatalf("expected default temperature %v got %v", defaultTemperature, cfg.Temperature)
	}
	if cfg.Headers == nil {
		t.Fatalf("headers map should be initialized")
	}
}

func TestWithOptions(t *testing.T) {
	cfg := NewConfig(
		WithModel("test-model"),
		WithTemperature(0.5),
		WithMaxTokens(123),
		WithAPIKey("abc"),
		WithBaseURL("http://localhost"),
		WithCost(2.50, 10.0),
	)
	if cfg.Model != "test-model" || cfg.Temperature != 0.5 || cfg.MaxTokens != 123 || cfg.APIKey != "abc" || cfg.BaseURL != "http://localhost" {
		t.Fatalf("option application failed: %+v", cfg)
	}
	if cfg.InputCostPerMillionTokens != 2.50 {
		t.Fatalf("expected input cost 2.50, got %v", cfg.InputCostPerMillionTokens)
	}
	if cfg.OutputCostPerMillionTokens != 10.0 {
		t.Fatalf("expected output cost 10.0, got %v", cfg.OutputCostPerMillionTokens)
	}
}

func TestToolOptionsClone(t *testing.T) {
	schema := map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"city": map[string]interface{}{"type": "string"},
		},
	}
	cfg := NewConfig(
		WithTools(NewTool("weather", "Get weather", schema)),
		WithToolChoice(NamedToolChoice("weather")),
	)
	schema["type"] = "mutated"

	if len(cfg.Tools) != 1 || cfg.Tools[0].Parameters["type"] != "object" {
		t.Fatalf("tool option should clone schemas, got %+v", cfg.Tools)
	}
	if cfg.ToolChoice == nil || cfg.ToolChoice.Mode != ToolChoiceNamed || cfg.ToolChoice.Name != "weather" {
		t.Fatalf("unexpected tool choice: %+v", cfg.ToolChoice)
	}

	clone := cfg.Clone()
	clone.Tools[0].Parameters["type"] = "changed"
	if cfg.Tools[0].Parameters["type"] != "object" {
		t.Fatalf("config clone should isolate tool schemas")
	}
}

type staticTokenSource struct {
	token *auth.Token
}

func (s *staticTokenSource) Token(context.Context) (*auth.Token, error) {
	return s.token, nil
}

func TestTokenSourceOption(t *testing.T) {
	src := &staticTokenSource{token: &auth.Token{AccessToken: "abc"}}
	cfg := NewConfig(
		WithTokenSource(src),
		WithAPIKey("api-key"),
	)
	if cfg.TokenSource != src {
		t.Fatalf("expected TokenSource to be set")
	}
	if cfg.APIKey != "api-key" {
		t.Fatalf("expected APIKey to be preserved")
	}

	clone := cfg.Clone()
	if clone.TokenSource != src {
		t.Fatalf("expected cloned TokenSource to be preserved")
	}
}

func TestWithRetryOnStatus(t *testing.T) {
	cfg := NewConfig(
		WithRetryOnStatus(429, false),
		WithRetryOnStatus(500, true),
	)
	if cfg.RetryOnStatus[http.StatusTooManyRequests] {
		t.Fatalf("expected 429 retry disabled")
	}
	if !cfg.RetryOnStatus[http.StatusInternalServerError] {
		t.Fatalf("expected 500 retry enabled")
	}

	var emptyCfg Config
	WithRetryOnStatus(429, false)(&emptyCfg)
	if emptyCfg.RetryOnStatus == nil {
		t.Fatalf("expected RetryOnStatus map to be initialized")
	}
	if emptyCfg.RetryOnStatus[429] {
		t.Fatalf("expected 429 retry disabled on zero config")
	}

	clone := cfg.Clone()
	clone.RetryOnStatus[429] = true
	if cfg.RetryOnStatus[429] {
		t.Fatalf("retry map mutation on clone leaked to original config")
	}
}

func TestWithHeader(t *testing.T) {
	cfg := NewConfig(
		WithHeader("X-Test-1", "value1"),
		WithHeader("X-Test-2", "value2"),
	)
	if cfg.Headers["X-Test-1"] != "value1" {
		t.Fatalf("expected X-Test-1 'value1', got %q", cfg.Headers["X-Test-1"])
	}
	if cfg.Headers["X-Test-2"] != "value2" {
		t.Fatalf("expected X-Test-2 'value2', got %q", cfg.Headers["X-Test-2"])
	}

	// Verify safe population when Headers map is nil
	var emptyCfg Config
	WithHeader("X-Nil-Init", "safe")(&emptyCfg)
	if emptyCfg.Headers["X-Nil-Init"] != "safe" {
		t.Fatalf("expected X-Nil-Init 'safe' on nil headers config, got %q", emptyCfg.Headers["X-Nil-Init"])
	}

	// Verify clone isolation
	clone := cfg.Clone()
	clone.Headers["X-Test-1"] = "mutated"
	if cfg.Headers["X-Test-1"] != "value1" {
		t.Fatalf("header mutation on clone leaked to original config")
	}
}

func TestWithExtraBody(t *testing.T) {
	body1 := map[string]json.RawMessage{
		"reasoning_effort": json.RawMessage(`"high"`),
		"max_tokens":       json.RawMessage(`2048`),
	}
	body2 := map[string]json.RawMessage{
		"metadata": json.RawMessage(`{"user_id":"test-user"}`),
	}

	cfg := NewConfig(
		WithExtraBody(body1),
		WithExtraBody(body2),
	)

	if string(cfg.ExtraBody["reasoning_effort"]) != `"high"` {
		t.Fatalf("expected reasoning_effort 'high', got %s", string(cfg.ExtraBody["reasoning_effort"]))
	}
	if string(cfg.ExtraBody["max_tokens"]) != `2048` {
		t.Fatalf("expected max_tokens '2048', got %s", string(cfg.ExtraBody["max_tokens"]))
	}
	if string(cfg.ExtraBody["metadata"]) != `{"user_id":"test-user"}` {
		t.Fatalf("expected metadata '{\"user_id\":\"test-user\"}', got %s", string(cfg.ExtraBody["metadata"]))
	}

	// Modifying original map should not affect cfg.ExtraBody
	body1["reasoning_effort"] = json.RawMessage(`"low"`)
	if string(cfg.ExtraBody["reasoning_effort"]) != `"high"` {
		t.Fatalf("mutating input map affected stored ExtraBody")
	}

	// Verify safe population when ExtraBody map is nil
	var emptyCfg Config
	WithExtraBody(map[string]json.RawMessage{"foo": json.RawMessage(`"bar"`)})(&emptyCfg)
	if string(emptyCfg.ExtraBody["foo"]) != `"bar"` {
		t.Fatalf("expected foo 'bar' on nil ExtraBody config")
	}

	// Verify ApplyOptions initializes ExtraBody if nil
	var zeroCfg Config
	ApplyOptions(&zeroCfg)
	if zeroCfg.ExtraBody == nil {
		t.Fatalf("expected ApplyOptions to initialize ExtraBody map")
	}
}

func TestWithExtraBodyClone(t *testing.T) {
	cfg := NewConfig(
		WithExtraBody(map[string]json.RawMessage{
			"reasoning_effort": json.RawMessage(`"medium"`),
		}),
	)

	clone := cfg.Clone()

	// 1. Add key to clone - original must not be affected
	clone.ExtraBody["added_key"] = json.RawMessage(`123`)
	if _, ok := cfg.ExtraBody["added_key"]; ok {
		t.Fatalf("adding key to clone affected original ExtraBody")
	}

	// 2. Mutate byte slice in clone - original bytes must not be affected
	clone.ExtraBody["reasoning_effort"][1] = 'x'
	if string(cfg.ExtraBody["reasoning_effort"]) != `"medium"` {
		t.Fatalf("byte mutation on clone affected original ExtraBody value")
	}
}

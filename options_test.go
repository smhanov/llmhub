package llmhub

import "testing"

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

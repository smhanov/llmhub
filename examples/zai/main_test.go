package main

import (
	"testing"

	"github.com/smhanov/llmhub/providers/zai"
)

func TestDefaultEndpoints(t *testing.T) {
	if zai.CodingPlanBaseURL != "https://api.z.ai/api/coding/paas/v4" {
		t.Fatalf("unexpected coding plan URL: %s", zai.CodingPlanBaseURL)
	}
	if zai.GeneralBaseURL != "https://api.z.ai/api/paas/v4" {
		t.Fatalf("unexpected general URL: %s", zai.GeneralBaseURL)
	}
}

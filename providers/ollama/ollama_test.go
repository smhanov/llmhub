package ollama

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/smhanov/llmhub"
)

func TestOllamaGenerate(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != chatEndpoint {
			t.Fatalf("unexpected path: %s", r.URL.Path)
		}
		defer r.Body.Close()
		var req chatRequest
		data, _ := io.ReadAll(r.Body)
		if err := json.Unmarshal(data, &req); err != nil {
			t.Fatalf("unmarshal: %v", err)
		}
		if !req.Stream {
			w.Header().Set("Content-Type", "application/json")
			io.WriteString(w, `{"message":{"role":"assistant","content":"pong"},"prompt_eval_count":1,"eval_count":2,"done":true}`)
			return
		}
	}))
	defer server.Close()

	provider, err := New("", llmhub.WithBaseURL(server.URL), llmhub.WithModel("local"))
	if err != nil {
		t.Fatalf("new provider: %v", err)
	}
	resp, err := provider.Generate(context.Background(), []*llmhub.Message{llmhub.NewUserMessage(llmhub.Text("ping"))})
	if err != nil {
		t.Fatalf("generate: %v", err)
	}
	if resp.Text() != "pong" {
		t.Fatalf("unexpected response: %s", resp.Text())
	}
}

func TestOllamaGenerateThinkingLevel(t *testing.T) {
	cases := []struct {
		name  string
		level llmhub.ThinkingLevel
		want  interface{}
	}{
		{name: "high", level: llmhub.ThinkingLevelHigh, want: "high"},
		{name: "minimal disables", level: llmhub.ThinkingLevelMinimal, want: false},
		{name: "xhigh clamps to high", level: llmhub.ThinkingLevelXHigh, want: "high"},
		{name: "unset omits think", level: "", want: nil},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				defer r.Body.Close()
				data, _ := io.ReadAll(r.Body)
				var req chatRequest
				if err := json.Unmarshal(data, &req); err != nil {
					t.Fatalf("unmarshal: %v", err)
				}
				if tc.want == nil {
					if req.Think != nil {
						t.Fatalf("think must be absent when ThinkingLevel is unset, got %v", req.Think)
					}
				} else if req.Think != tc.want {
					t.Fatalf("expected think %v, got %v", tc.want, req.Think)
				}
				w.Header().Set("Content-Type", "application/json")
				io.WriteString(w, `{"message":{"role":"assistant","content":"pong"},"done":true}`)
			}))
			defer server.Close()

			var opts []llmhub.Option
			opts = append(opts, llmhub.WithBaseURL(server.URL), llmhub.WithModel("local"))
			if tc.level != "" {
				opts = append(opts, llmhub.WithThinkingLevel(tc.level))
			}
			provider, err := New("", opts...)
			if err != nil {
				t.Fatalf("new provider: %v", err)
			}
			if _, err := provider.Generate(context.Background(), []*llmhub.Message{llmhub.NewUserMessage(llmhub.Text("ping"))}); err != nil {
				t.Fatalf("generate: %v", err)
			}
		})
	}
}

func TestOllamaGenerateWithCost(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		io.WriteString(w, `{"message":{"role":"assistant","content":"pong"},"prompt_eval_count":1,"eval_count":2,"cost":0.00005,"done":true}`)
	}))
	defer server.Close()

	provider, err := New("", llmhub.WithBaseURL(server.URL), llmhub.WithModel("local"))
	if err != nil {
		t.Fatalf("new provider: %v", err)
	}
	resp, err := provider.Generate(context.Background(), []*llmhub.Message{llmhub.NewUserMessage(llmhub.Text("ping"))})
	if err != nil {
		t.Fatalf("generate: %v", err)
	}
	if resp.Usage.Cost != 0.00005 {
		t.Fatalf("expected cost 0.00005, got %v", resp.Usage.Cost)
	}
}

func TestOllamaGenerateThinking(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		io.WriteString(w, `{"message":{"role":"assistant","content":"final","thinking":"plan"},"prompt_eval_count":1,"eval_count":2,"done":true}`)
	}))
	defer server.Close()

	provider, err := New("", llmhub.WithBaseURL(server.URL), llmhub.WithModel("local"))
	if err != nil {
		t.Fatalf("new provider: %v", err)
	}
	resp, err := provider.Generate(context.Background(), []*llmhub.Message{llmhub.NewUserMessage(llmhub.Text("ping"))})
	if err != nil {
		t.Fatalf("generate: %v", err)
	}
	if resp.Text() != "final" {
		t.Fatalf("unexpected response: %s", resp.Text())
	}
	if resp.ReasoningText() != "plan" {
		t.Fatalf("unexpected reasoning: %s", resp.ReasoningText())
	}
}

func TestOllamaGenerateToolCalls(t *testing.T) {
	tool := llmhub.NewTool("weather", "Get weather", map[string]interface{}{"type": "object"})
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		defer r.Body.Close()
		var req chatRequest
		data, _ := io.ReadAll(r.Body)
		if err := json.Unmarshal(data, &req); err != nil {
			t.Fatalf("unmarshal: %v", err)
		}
		if len(req.Tools) != 1 || req.Tools[0].Function.Name != "weather" {
			t.Fatalf("expected weather tool, got %+v", req.Tools)
		}
		if len(req.Messages) != 3 || len(req.Messages[1].ToolCalls) != 1 || req.Messages[2].ToolCallID != "call-1" {
			t.Fatalf("unexpected tool messages: %+v", req.Messages)
		}
		io.WriteString(w, `{"message":{"role":"assistant","tool_calls":[{"id":"call-2","type":"function","function":{"name":"weather","arguments":{"city":"Toronto"}}}]},"done":true}`)
	}))
	defer server.Close()

	provider, err := New("", llmhub.WithBaseURL(server.URL), llmhub.WithModel("local"), llmhub.WithTools(tool))
	if err != nil {
		t.Fatalf("new provider: %v", err)
	}
	resp, err := provider.Generate(context.Background(), []*llmhub.Message{
		llmhub.NewUserMessage(llmhub.Text("weather?")),
		llmhub.NewAssistantMessage(llmhub.ToolCall("call-1", "weather", `{"city":"Paris"}`)),
		llmhub.NewToolResultMessage("call-1", "weather", llmhub.Text(`{"temp":20}`)),
	})
	if err != nil {
		t.Fatalf("generate: %v", err)
	}
	calls := resp.ToolCalls()
	if len(calls) != 1 || calls[0].ID != "call-2" || calls[0].Arguments != `{"city":"Toronto"}` {
		t.Fatalf("unexpected tool calls: %+v", calls)
	}
}

func TestBuildChatRequest_Temperature(t *testing.T) {
	prompt := []*llmhub.Message{
		llmhub.NewUserMessage(llmhub.Text("ping")),
	}

	// Case 1: unset temperature - Options has no temperature key.
	payload, err := buildChatRequest(prompt, llmhub.NewConfig(llmhub.WithModel("local")), false)
	if err != nil {
		t.Fatalf("buildChatRequest failed: %v", err)
	}
	var decoded map[string]json.RawMessage
	if err := json.Unmarshal(payload, &decoded); err != nil {
		t.Fatalf("unmarshal payload failed: %v", err)
	}
	var options map[string]interface{}
	if len(decoded["options"]) > 0 {
		if err := json.Unmarshal(decoded["options"], &options); err != nil {
			t.Fatalf("unmarshal options failed: %v", err)
		}
	}
	if _, ok := options["temperature"]; ok {
		t.Fatalf("temperature must be absent when unset, got %v", options["temperature"])
	}

	// Case 2: explicit zero - present as 0, not omitted.
	zeroCfg := llmhub.NewConfig(
		llmhub.WithModel("local"),
		llmhub.WithTemperature(0),
	)
	payload, err = buildChatRequest(prompt, zeroCfg, false)
	if err != nil {
		t.Fatalf("buildChatRequest failed: %v", err)
	}
	decoded = nil
	if err := json.Unmarshal(payload, &decoded); err != nil {
		t.Fatalf("unmarshal payload failed: %v", err)
	}
	options = nil
	if err := json.Unmarshal(decoded["options"], &options); err != nil {
		t.Fatalf("unmarshal options failed: %v", err)
	}
	temp, ok := options["temperature"].(float64)
	if !ok {
		t.Fatalf("expected temperature 0 in options, got %v", options["temperature"])
	}
	if temp != 0 {
		t.Fatalf("expected temperature 0, got %v", temp)
	}

	// Case 3: explicit value - rendered verbatim.
	cfg := llmhub.NewConfig(
		llmhub.WithModel("local"),
		llmhub.WithTemperature(0.5),
	)
	payload, err = buildChatRequest(prompt, cfg, false)
	if err != nil {
		t.Fatalf("buildChatRequest failed: %v", err)
	}
	decoded = nil
	if err := json.Unmarshal(payload, &decoded); err != nil {
		t.Fatalf("unmarshal payload failed: %v", err)
	}
	options = nil
	if err := json.Unmarshal(decoded["options"], &options); err != nil {
		t.Fatalf("unmarshal options failed: %v", err)
	}
	temp, ok = options["temperature"].(float64)
	if !ok || temp != 0.5 {
		t.Fatalf("expected temperature 0.5, got %v", options["temperature"])
	}

	// Case 4: MaxTokens-only config still renders num_predict without a
	// temperature key.
	maxTokensCfg := llmhub.NewConfig(
		llmhub.WithModel("local"),
		llmhub.WithMaxTokens(42),
	)
	payload, err = buildChatRequest(prompt, maxTokensCfg, false)
	if err != nil {
		t.Fatalf("buildChatRequest failed: %v", err)
	}
	decoded = nil
	if err := json.Unmarshal(payload, &decoded); err != nil {
		t.Fatalf("unmarshal payload failed: %v", err)
	}
	options = nil
	if err := json.Unmarshal(decoded["options"], &options); err != nil {
		t.Fatalf("unmarshal options failed: %v", err)
	}
	numPredict, ok := options["num_predict"].(float64)
	if !ok || numPredict != 42 {
		t.Fatalf("expected num_predict 42, got %v", options["num_predict"])
	}
	if _, ok := options["temperature"]; ok {
		t.Fatalf("temperature must be absent for MaxTokens-only config, got %v", options["temperature"])
	}
}

func TestOllamaStreamHTTPErrorIsSynchronous(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		io.WriteString(w, `{"error":"busy"}`)
	}))
	defer server.Close()

	provider, err := New("", llmhub.WithBaseURL(server.URL))
	if err != nil {
		t.Fatalf("new provider: %v", err)
	}
	_, err = provider.Stream(context.Background(), []*llmhub.Message{llmhub.NewUserMessage(llmhub.Text("ping"))})
	if err == nil {
		t.Fatal("expected stream error")
	}
	if !strings.Contains(err.Error(), "http 400") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestOllamaStream(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		defer r.Body.Close()
		var req chatRequest
		data, _ := io.ReadAll(r.Body)
		json.Unmarshal(data, &req)
		if !req.Stream {
			io.WriteString(w, `{"message":{"role":"assistant","content":"full"},"done":true}`)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		io.WriteString(w, `{"message":{"role":"assistant","content":"hel"},"done":false}`+"\n")
		io.WriteString(w, `{"message":{"role":"assistant","content":"lo"},"done":true}`+"\n")
	}))
	defer server.Close()

	provider, err := New("", llmhub.WithBaseURL(server.URL))
	if err != nil {
		t.Fatalf("new provider: %v", err)
	}
	stream, err := provider.Stream(context.Background(), []*llmhub.Message{llmhub.NewUserMessage(llmhub.Text("ping"))})
	if err != nil {
		t.Fatalf("stream: %v", err)
	}
	first := <-stream
	second := <-stream
	if first.Delta+second.Delta != "hello" {
		t.Fatalf("unexpected concatenation: %s%s", first.Delta, second.Delta)
	}
	done := <-stream
	if !done.Done {
		t.Fatalf("expected done chunk")
	}
}

func TestOllamaStreamThinking(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		io.WriteString(w, `{"message":{"role":"assistant","content":"","thinking":"step"},"done":false}`+"\n")
		io.WriteString(w, `{"message":{"role":"assistant","content":"ok"},"done":true}`+"\n")
	}))
	defer server.Close()

	provider, err := New("", llmhub.WithBaseURL(server.URL))
	if err != nil {
		t.Fatalf("new provider: %v", err)
	}
	stream, err := provider.Stream(context.Background(), []*llmhub.Message{llmhub.NewUserMessage(llmhub.Text("ping"))})
	if err != nil {
		t.Fatalf("stream: %v", err)
	}
	first := <-stream
	second := <-stream
	if first.ReasoningDelta != "step" {
		t.Fatalf("unexpected reasoning delta: %+v", first)
	}
	if second.Delta != "ok" {
		t.Fatalf("unexpected text delta: %+v", second)
	}
}

func TestOllamaStreamDoneReason(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		io.WriteString(w, `{"message":{"role":"assistant","content":"long"},"done":false}`+"\n")
		io.WriteString(w, `{"message":{"role":"assistant","content":""},"done":true,"done_reason":"length"}`+"\n")
	}))
	defer server.Close()

	provider, err := New("", llmhub.WithBaseURL(server.URL))
	if err != nil {
		t.Fatalf("new provider: %v", err)
	}
	stream, err := provider.Stream(context.Background(), []*llmhub.Message{llmhub.NewUserMessage(llmhub.Text("ping"))})
	if err != nil {
		t.Fatalf("stream: %v", err)
	}
	var chunks []llmhub.StreamChunk
	for chunk := range stream {
		chunks = append(chunks, chunk)
	}
	if len(chunks) < 2 {
		t.Fatalf("expected text and done chunks, got %+v", chunks)
	}
	if chunks[len(chunks)-1].FinishReason != "length" {
		t.Fatalf("expected length finish reason, got %+v", chunks[len(chunks)-1])
	}
}

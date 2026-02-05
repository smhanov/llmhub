package llmhub

import (
	"context"
	"errors"
	"testing"
)

func TestRegisterProvider(t *testing.T) {
	name := "stub-" + t.Name()
	err := RegisterProvider(name, func(apiKey string, opts ...Option) (Provider, error) {
		return &testProvider{name: name}, nil
	})
	if err != nil && !errors.Is(err, ErrProviderAlreadyRegistered) {
		t.Fatalf("register provider failed: %v", err)
	}
	// Duplicate registration should fail.
	if err := RegisterProvider(name, func(apiKey string, opts ...Option) (Provider, error) { return nil, nil }); err == nil {
		t.Fatalf("expected duplicate registration to fail")
	}
}

func TestNewClientFromRegistry(t *testing.T) {
	name := "callable-" + t.Name()
	err := RegisterProvider(name, func(apiKey string, opts ...Option) (Provider, error) {
		return &testProvider{name: name}, nil
	})
	if err != nil && !errors.Is(err, ErrProviderAlreadyRegistered) {
		t.Fatalf("register provider failed: %v", err)
	}
	client, err := New(name, "token")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	resp, err := client.Generate(context.Background(), []*Message{NewUserMessage(Text("hi"))})
	if err != nil {
		t.Fatalf("generate failed: %v", err)
	}
	if resp.Text() == "" {
		t.Fatalf("expected response text")
	}
}

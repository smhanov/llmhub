package llmhub

import (
	"errors"
	"fmt"
)

var (
	// ErrProviderNotFound is returned when the requested provider has not been registered.
	ErrProviderNotFound = errors.New("llmhub: provider not found")
	// ErrProviderAlreadyRegistered signals that a provider name has already been registered.
	ErrProviderAlreadyRegistered = errors.New("llmhub: provider already registered")
	// ErrNotImplemented is used by provider stubs that have not yet been wired up.
	ErrNotImplemented = errors.New("llmhub: feature not implemented")
	// ErrInvalidInput wraps validation failures on user-supplied data.
	ErrInvalidInput = errors.New("llmhub: invalid input")
)

// Error captures provider-specific failures with additional context.
type Error struct {
	Provider string
	Op       string
	Err      error
}

// Error returns the string form of the wrapped error.
func (e *Error) Error() string {
	if e == nil {
		return ""
	}
	return fmt.Sprintf("llmhub(%s): %s: %v", e.Provider, e.Op, e.Err)
}

// Unwrap exposes the underlying error for errors.Is/As support.
func (e *Error) Unwrap() error {
	if e == nil {
		return nil
	}
	return e.Err
}

// StreamError reports an error delivered by a provider inside a response
// stream, as opposed to an HTTP failure that is returned synchronously from
// Stream. It is the Err carried by the terminal StreamChunk, so callers can
// inspect the provider-reported details with errors.As:
//
//	for chunk := range stream {
//	    var streamErr *llmhub.StreamError
//	    if chunk.Err != nil && errors.As(chunk.Err, &streamErr) {
//	        log.Printf("%s failed: %s (type=%s code=%s)", streamErr.Provider, streamErr.Message, streamErr.Type, streamErr.Code)
//	    }
//	}
//
// Type, Code, and Param are empty when the provider did not supply them.
type StreamError struct {
	// Provider is the lane that reported the error (openai, anthropic, gemini, ...).
	Provider string
	// Message is the human-readable message from the provider.
	Message string
	// Type is the provider's error type/status, when supplied.
	Type string
	// Code is the provider's error code, normalized to a string.
	Code string
	// Param names the offending request parameter, when supplied.
	Param string
}

// Error returns "<provider>: <message>" plus the type and code when present.
func (e *StreamError) Error() string {
	if e == nil {
		return ""
	}
	msg := e.Message
	if e.Provider != "" {
		msg = e.Provider + ": " + msg
	}
	details := ""
	if e.Type != "" {
		details = "type=" + e.Type
	}
	if e.Code != "" {
		if details != "" {
			details += ", "
		}
		details += "code=" + e.Code
	}
	if details != "" {
		msg += " (" + details + ")"
	}
	return msg
}

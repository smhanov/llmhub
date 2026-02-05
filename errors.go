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

// Package telemetry carries request-scoped instrumentation collected deep in
// the HTTP stack up to the public Client wrapper.
//
// It intentionally depends only on the standard library: the root llmhub
// package imports it, so importing llmhub here would form an import cycle.
package telemetry

import (
	"context"
	"sync"
	"time"
)

// Attempt describes one HTTP attempt made while serving a request.
type Attempt struct {
	// StatusCode is the HTTP status returned by the attempt, or zero when
	// the attempt failed before receiving a response.
	StatusCode int
	// Duration is the time from sending the request until response headers
	// arrived, or until the transport error for a failed attempt.
	Duration time.Duration
	// Backoff is the delay slept after the attempt before the next one.
	// It is zero when no retry followed.
	Backoff time.Duration
	// RetryAfter is the server-provided Retry-After delay for the attempt,
	// before the retry policy cap is applied. It is zero when absent or
	// invalid.
	RetryAfter time.Duration
	// Err is the transport error for an attempt that received no response.
	Err string
}

// Session collects attempts for one request. It is safe for concurrent use
// and a nil *Session is a valid no-op receiver.
type Session struct {
	mu       sync.Mutex
	attempts []Attempt
}

type sessionKey struct{}

// WithSession returns a context carrying a new Session and the Session
// itself. The same context must reach the HTTP layer for attempts to be
// recorded.
func WithSession(ctx context.Context) (context.Context, *Session) {
	if ctx == nil {
		ctx = context.Background()
	}
	session := &Session{}
	return context.WithValue(ctx, sessionKey{}, session), session
}

// FromContext returns the Session stored in ctx, or nil when none is present.
func FromContext(ctx context.Context) *Session {
	if ctx == nil {
		return nil
	}
	session, _ := ctx.Value(sessionKey{}).(*Session)
	return session
}

// Record appends an attempt. A nil receiver is a no-op.
func (s *Session) Record(attempt Attempt) {
	if s == nil {
		return
	}
	s.mu.Lock()
	s.attempts = append(s.attempts, attempt)
	s.mu.Unlock()
}

// Attempts returns a copy of the recorded attempts in order. A nil receiver
// returns nil.
func (s *Session) Attempts() []Attempt {
	if s == nil {
		return nil
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if len(s.attempts) == 0 {
		return nil
	}
	out := make([]Attempt, len(s.attempts))
	copy(out, s.attempts)
	return out
}

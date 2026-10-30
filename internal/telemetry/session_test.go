package telemetry

import (
	"context"
	"testing"
	"time"
)

func TestWithSessionRoundTrip(t *testing.T) {
	ctx, session := WithSession(context.Background())
	if session == nil {
		t.Fatal("WithSession returned a nil Session")
	}
	if got := FromContext(ctx); got != session {
		t.Fatalf("FromContext = %p, want %p", got, session)
	}
	if got := FromContext(context.Background()); got != nil {
		t.Fatalf("FromContext without a session = %p, want nil", got)
	}
}

func TestSessionAttemptsReturnsCopy(t *testing.T) {
	ctx, session := WithSession(context.Background())
	if FromContext(ctx) != session {
		t.Fatal("session not stored in context")
	}
	first := Attempt{StatusCode: 429, Duration: time.Millisecond, Backoff: 2 * time.Millisecond}
	second := Attempt{StatusCode: 200, Duration: 3 * time.Millisecond}
	session.Record(first)
	session.Record(second)

	got := session.Attempts()
	if len(got) != 2 {
		t.Fatalf("Attempts length = %d, want 2", len(got))
	}
	if got[0] != first || got[1] != second {
		t.Fatalf("Attempts = %+v, want [%+v %+v]", got, first, second)
	}

	got[0].StatusCode = 500
	got = append(got, Attempt{StatusCode: 503})

	again := session.Attempts()
	if len(again) != 2 {
		t.Fatalf("second Attempts length = %d, want 2", len(again))
	}
	if again[0].StatusCode != 429 {
		t.Fatalf("recorded attempt mutated through returned slice: %+v", again[0])
	}
}

func TestNilSessionIsSafe(t *testing.T) {
	var session *Session
	session.Record(Attempt{StatusCode: 200})
	if got := session.Attempts(); got != nil {
		t.Fatalf("nil Session.Attempts = %+v, want nil", got)
	}
	if got := FromContext(nil); got != nil {
		t.Fatalf("FromContext(nil) = %p, want nil", got)
	}
}

package llmhub

import (
	"context"
	"net/http/httptrace"
	"sync/atomic"
	"time"
)

// requestTimer accumulates latency telemetry for one Generate or Stream
// request. Trace callbacks run on HTTP transport goroutines, so the
// connection and first-byte fields use atomics; the remaining fields are
// only touched by the goroutine that forwards chunks (Stream) or returns
// the response (Generate), so they need no locking.
type requestTimer struct {
	start time.Time

	// connectedNS holds nanoseconds elapsed since start when GotConn
	// fired; the first connection wins (CAS) so retries and keep-alive
	// reuse do not overwrite the cold-start measurement.
	connectedNS atomic.Int64
	// firstByteNS holds nanoseconds elapsed since start when
	// GotFirstResponseByte last fired; latest wins because the successful
	// attempt is always the last one when requests are retried.
	firstByteNS atomic.Int64

	// Wrapper-goroutine-owned state.
	firstTokenNS int64
	chunks       int
}

func newRequestTimer() *requestTimer {
	return &requestTimer{start: time.Now()}
}

// trace enriches ctx so HTTP-backed providers report connection and
// first-byte events into the timer. Providers must build their requests
// with this ctx (all built-in lanes use http.NewRequestWithContext).
func (t *requestTimer) trace(ctx context.Context) context.Context {
	trace := &httptrace.ClientTrace{
		GotConn: func(httptrace.GotConnInfo) {
			t.connectedNS.CompareAndSwap(0, int64(time.Since(t.start)))
		},
		GotFirstResponseByte: func() {
			t.firstByteNS.Store(int64(time.Since(t.start)))
		},
	}
	return httptrace.WithClientTrace(ctx, trace)
}

// markContent records TTFT on the first content-bearing chunk and counts
// the chunk. Call it before snapshot for the same chunk.
func (t *requestTimer) markContent() {
	if t.firstTokenNS == 0 {
		t.firstTokenNS = int64(time.Since(t.start))
	}
	t.chunks++
}

// snapshot builds an immutable TimingMetrics view. Total is only set when
// done is true (terminal stream chunk, or a completed Generate response).
func (t *requestTimer) snapshot(done bool) *TimingMetrics {
	m := &TimingMetrics{StartedAt: t.start}
	if ns := t.connectedNS.Load(); ns != 0 {
		m.Connected = time.Duration(ns)
	}
	if ns := t.firstByteNS.Load(); ns != 0 {
		m.FirstByte = time.Duration(ns)
	}
	if t.firstTokenNS != 0 {
		m.FirstToken = time.Duration(t.firstTokenNS)
	}
	m.Chunks = t.chunks
	if done {
		m.Total = time.Since(t.start)
	}
	return m
}

package httpretry

import (
	"context"
	"io"
	"math/rand"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/smhanov/llmhub"
	"github.com/smhanov/llmhub/internal/telemetry"
)

const (
	defaultMaxRetries = 4
	defaultBaseDelay  = 500 * time.Millisecond
	defaultMaxDelay   = 8 * time.Second
)

func init() {
	rand.Seed(time.Now().UnixNano())
}

// Config controls retry behavior for rate-limited HTTP requests.
type Config struct {
	MaxRetries int
	BaseDelay  time.Duration
	MaxDelay   time.Duration
	// RetryOnStatus overrides whether a status is retried. true means retry
	// with backoff; false means return immediately. Statuses not present
	// keep the default (retry 429 only).
	RetryOnStatus map[int]bool
}

// DefaultConfig returns a conservative retry policy suitable for API rate limits.
func DefaultConfig() Config {
	return Config{
		MaxRetries: defaultMaxRetries,
		BaseDelay:  defaultBaseDelay,
		MaxDelay:   defaultMaxDelay,
	}
}

// FromLLMHub returns the default retry policy with per-status overrides from cfg.
func FromLLMHub(cfg llmhub.Config) Config {
	rc := DefaultConfig()
	if len(cfg.RetryOnStatus) == 0 {
		return rc
	}
	rc.RetryOnStatus = make(map[int]bool, len(cfg.RetryOnStatus))
	for status, retry := range cfg.RetryOnStatus {
		rc.RetryOnStatus[status] = retry
	}
	return rc
}

// Do issues an HTTP request and retries with backoff for retryable statuses.
func Do(ctx context.Context, client *http.Client, makeReq func() (*http.Request, error), cfg Config) (*http.Response, error) {
	if cfg.MaxRetries < 0 {
		cfg.MaxRetries = 0
	}
	if cfg.BaseDelay <= 0 {
		cfg.BaseDelay = defaultBaseDelay
	}
	if cfg.MaxDelay <= 0 {
		cfg.MaxDelay = defaultMaxDelay
	}
	session := telemetry.FromContext(ctx)
	for attempt := 0; ; attempt++ {
		req, err := makeReq()
		if err != nil {
			return nil, err
		}
		started := time.Now()
		resp, err := client.Do(req)
		duration := time.Since(started)
		if err != nil {
			session.Record(telemetry.Attempt{Duration: duration, Err: err.Error()})
			return nil, err
		}
		if !shouldRetry(resp.StatusCode, cfg) {
			session.Record(telemetry.Attempt{StatusCode: resp.StatusCode, Duration: duration})
			return resp, nil
		}
		if attempt >= cfg.MaxRetries {
			session.Record(telemetry.Attempt{StatusCode: resp.StatusCode, Duration: duration})
			return resp, nil
		}
		drainAndClose(resp)
		delay, retryAfter := backoffDelay(resp, cfg, attempt)
		session.Record(telemetry.Attempt{
			StatusCode: resp.StatusCode,
			Duration:   duration,
			Backoff:    delay,
			RetryAfter: retryAfter,
		})
		if !sleepWithContext(ctx, delay) {
			return nil, ctx.Err()
		}
	}
}

func shouldRetry(status int, cfg Config) bool {
	if retry, ok := cfg.RetryOnStatus[status]; ok {
		return retry
	}
	return status == http.StatusTooManyRequests
}

// backoffDelay returns the delay to sleep before the next attempt and the raw
// Retry-After delay when the response supplied a valid one. The returned delay
// keeps the existing capped/jittered policy; retryAfter is uncapped so callers
// can report what the server asked for.
func backoffDelay(resp *http.Response, cfg Config, attempt int) (delay, retryAfter time.Duration) {
	if resp != nil {
		if parsed, ok := parseRetryAfter(resp.Header.Get("Retry-After")); ok {
			return capDelay(parsed, cfg.MaxDelay), parsed
		}
	}
	if attempt < 0 {
		attempt = 0
	}
	backoff := cfg.BaseDelay * time.Duration(1<<attempt)
	backoff = capDelay(backoff, cfg.MaxDelay)
	jitter := 0.5 + rand.Float64()
	return time.Duration(float64(backoff) * jitter), 0
}

func capDelay(delay, max time.Duration) time.Duration {
	if max <= 0 {
		return delay
	}
	if delay > max {
		return max
	}
	return delay
}

func parseRetryAfter(value string) (time.Duration, bool) {
	value = strings.TrimSpace(value)
	if value == "" {
		return 0, false
	}
	if seconds, err := strconv.Atoi(value); err == nil {
		if seconds <= 0 {
			return 0, false
		}
		return time.Duration(seconds) * time.Second, true
	}
	if parsed, err := http.ParseTime(value); err == nil {
		delay := time.Until(parsed)
		if delay <= 0 {
			return 0, false
		}
		return delay, true
	}
	return 0, false
}

func sleepWithContext(ctx context.Context, delay time.Duration) bool {
	if delay <= 0 {
		return true
	}
	timer := time.NewTimer(delay)
	defer timer.Stop()
	select {
	case <-ctx.Done():
		return false
	case <-timer.C:
		return true
	}
}

func drainAndClose(resp *http.Response) {
	if resp == nil || resp.Body == nil {
		return
	}
	_, _ = io.Copy(io.Discard, resp.Body)
	_ = resp.Body.Close()
}

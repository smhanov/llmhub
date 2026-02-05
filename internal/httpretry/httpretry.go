package httpretry

import (
	"context"
	"io"
	"math/rand"
	"net/http"
	"strconv"
	"strings"
	"time"
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
}

// DefaultConfig returns a conservative retry policy suitable for API rate limits.
func DefaultConfig() Config {
	return Config{
		MaxRetries: defaultMaxRetries,
		BaseDelay:  defaultBaseDelay,
		MaxDelay:   defaultMaxDelay,
	}
}

// Do issues an HTTP request and retries with backoff when a 429 response is received.
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
	for attempt := 0; ; attempt++ {
		req, err := makeReq()
		if err != nil {
			return nil, err
		}
		resp, err := client.Do(req)
		if err != nil {
			return nil, err
		}
		if resp.StatusCode != http.StatusTooManyRequests {
			return resp, nil
		}
		if attempt >= cfg.MaxRetries {
			return resp, nil
		}
		drainAndClose(resp)
		delay := backoffDelay(resp, cfg, attempt)
		if !sleepWithContext(ctx, delay) {
			return nil, ctx.Err()
		}
	}
}

func backoffDelay(resp *http.Response, cfg Config, attempt int) time.Duration {
	if resp != nil {
		if retryAfter, ok := parseRetryAfter(resp.Header.Get("Retry-After")); ok {
			return capDelay(retryAfter, cfg.MaxDelay)
		}
	}
	if attempt < 0 {
		attempt = 0
	}
	backoff := cfg.BaseDelay * time.Duration(1<<attempt)
	backoff = capDelay(backoff, cfg.MaxDelay)
	jitter := 0.5 + rand.Float64()
	return time.Duration(float64(backoff) * jitter)
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

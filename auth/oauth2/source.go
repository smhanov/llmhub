package oauth2

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"time"

	"github.com/smhanov/llmhub/auth"
)

const (
	defaultRefreshLeadTime = 2 * time.Minute
)

// RefreshingTokenSource implements auth.InvalidatableTokenSource using an auth.TokenStore
// and a TokenRefresher. It coordinates lazy loading, proactive refresh before expiry,
// thread-safe deduplicated refresh, and reactive cache invalidation.
type RefreshingTokenSource struct {
	store           auth.TokenStore
	refresher       TokenRefresher
	refreshLeadTime time.Duration
	now             func() time.Time

	mu              sync.Mutex
	cachedToken     *auth.Token
	lastInvalidated string
}

// TokenSourceOption configures a RefreshingTokenSource.
type TokenSourceOption func(*RefreshingTokenSource)

// WithRefreshLeadTime sets how far in advance of token expiry proactive refresh begins.
func WithRefreshLeadTime(d time.Duration) TokenSourceOption {
	return func(s *RefreshingTokenSource) {
		if d > 0 {
			s.refreshLeadTime = d
		}
	}
}

// withNow sets a custom clock function for deterministic unit testing.
func withNow(now func() time.Time) TokenSourceOption {
	return func(s *RefreshingTokenSource) {
		if now != nil {
			s.now = now
		}
	}
}

// NewRefreshingTokenSource creates a new refreshing token source.
func NewRefreshingTokenSource(store auth.TokenStore, refresher TokenRefresher, opts ...TokenSourceOption) *RefreshingTokenSource {
	s := &RefreshingTokenSource{
		store:           store,
		refresher:       refresher,
		refreshLeadTime: defaultRefreshLeadTime,
		now:             time.Now,
	}
	for _, opt := range opts {
		if opt != nil {
			opt(s)
		}
	}
	return s
}

// Token returns a valid access token, loading from the store or refreshing if needed.
func (s *RefreshingTokenSource) Token(ctx context.Context) (*auth.Token, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	default:
	}

	// 1. Lazy load on first request or if uncached
	if s.cachedToken == nil {
		if s.store == nil {
			return nil, fmt.Errorf("oauth2: %w: token store is nil", auth.ErrTokenNotFound)
		}
		loaded, err := s.store.Load(ctx)
		if err != nil {
			return nil, err
		}
		if loaded == nil || (loaded.AccessToken == "" && loaded.RefreshToken == "") {
			return nil, fmt.Errorf("oauth2: %w: stored token is empty", auth.ErrTokenNotFound)
		}
		s.cachedToken = loaded
	}

	tok := s.cachedToken
	now := s.now()

	// 2. Usable check: if token has no expiry, it is treated as usable until invalidated.
	// If it has expiry and expiry is beyond the refresh lead time, return it immediately.
	if s.isUsable(tok, now) {
		return tok.Clone(), nil
	}

	// 3. Proactive refresh or token is expired/invalidated.
	// In the refresh critical section, try reloading the store first in case another
	// goroutine or process already refreshed and saved a newer token.
	if s.store != nil {
		if stored, err := s.store.Load(ctx); err == nil && stored != nil {
			if stored.AccessToken != "" && stored.AccessToken != s.lastInvalidated && s.isUsable(stored, now) {
				s.cachedToken = stored
				return stored.Clone(), nil
			}
		}
	}

	// 4. If we need to refresh, verify we have a refresh token.
	if tok.RefreshToken == "" {
		return nil, fmt.Errorf("oauth2: %w: token expired or invalidated with no refresh token", auth.ErrReauthenticationRequired)
	}

	if s.refresher == nil {
		return nil, fmt.Errorf("oauth2: %w: token refresher is nil", auth.ErrReauthenticationRequired)
	}

	// 5. Perform the refresh request.
	refreshed, err := s.refresher.Refresh(ctx, tok)
	if err != nil {
		if errors.Is(err, auth.ErrReauthenticationRequired) || errors.Is(err, ErrInvalidGrant) {
			return nil, fmt.Errorf("oauth2: %w: %v", auth.ErrReauthenticationRequired, err)
		}
		return nil, fmt.Errorf("oauth2: token refresh failed: %w", err)
	}

	if refreshed == nil || refreshed.AccessToken == "" {
		return nil, fmt.Errorf("oauth2: refresh returned empty token: %w", ErrServerResponse)
	}

	// 6. Save the refreshed token to the store before returning.
	// If saving fails, do not adopt or return the token to avoid losing rotating refresh tokens.
	if s.store != nil {
		if err := s.store.Save(ctx, refreshed); err != nil {
			return nil, fmt.Errorf("oauth2: save refreshed token: %w", err)
		}
	}

	s.cachedToken = refreshed
	s.lastInvalidated = ""
	return refreshed.Clone(), nil
}

// Invalidate invalidates the cached token if it matches accessToken.
// A late 401 for an older token will not invalidate a newer refreshed token.
func (s *RefreshingTokenSource) Invalidate(accessToken string) {
	if accessToken == "" {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.cachedToken != nil && s.cachedToken.AccessToken == accessToken {
		s.lastInvalidated = accessToken
		// Mark expiry as past to force refresh on next Token() call
		s.cachedToken.Expiry = time.Unix(0, 0)
	}
}

func (s *RefreshingTokenSource) isUsable(tok *auth.Token, now time.Time) bool {
	if tok == nil || tok.AccessToken == "" {
		return false
	}
	if tok.AccessToken == s.lastInvalidated {
		return false
	}
	if tok.Expiry.IsZero() {
		return true
	}
	// Usable only if expiry is after now + leadTime
	return tok.Expiry.After(now.Add(s.refreshLeadTime))
}

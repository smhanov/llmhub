package oauth2

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/smhanov/llmhub/auth"
)

type memoryTokenStore struct {
	mu        sync.Mutex
	token     *auth.Token
	saveErr   error
	loadErr   error
	saveCount int
	loadCount int
}

func (m *memoryTokenStore) Load(ctx context.Context) (*auth.Token, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.loadCount++
	if m.loadErr != nil {
		return nil, m.loadErr
	}
	if m.token == nil {
		return nil, auth.ErrTokenNotFound
	}
	return m.token.Clone(), nil
}

func (m *memoryTokenStore) Save(ctx context.Context, token *auth.Token) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.saveCount++
	if m.saveErr != nil {
		return m.saveErr
	}
	m.token = token.Clone()
	return nil
}

type mockRefresher struct {
	mu           sync.Mutex
	refreshCount int
	refreshFn    func(ctx context.Context, token *auth.Token) (*auth.Token, error)
}

func (m *mockRefresher) Refresh(ctx context.Context, token *auth.Token) (*auth.Token, error) {
	m.mu.Lock()
	m.refreshCount++
	fn := m.refreshFn
	m.mu.Unlock()
	if fn != nil {
		return fn(ctx, token)
	}
	return &auth.Token{
		AccessToken:  fmt.Sprintf("refreshed-access-%d", m.refreshCount),
		TokenType:    "Bearer",
		RefreshToken: fmt.Sprintf("refreshed-refresh-%d", m.refreshCount),
		Expiry:       time.Now().Add(time.Hour),
	}, nil
}

func TestSourceValidTokenNoRefresh(t *testing.T) {
	now := time.Now()
	store := &memoryTokenStore{
		token: &auth.Token{
			AccessToken:  "valid-token",
			TokenType:    "Bearer",
			RefreshToken: "refresh-token",
			Expiry:       now.Add(10 * time.Minute),
		},
	}
	refresher := &mockRefresher{}
	src := NewRefreshingTokenSource(store, refresher, WithRefreshLeadTime(2*time.Minute), withNow(func() time.Time { return now }))

	tok, err := src.Token(context.Background())
	if err != nil {
		t.Fatalf("token: %v", err)
	}
	if tok.AccessToken != "valid-token" {
		t.Fatalf("unexpected access token: %s", tok.AccessToken)
	}
	if refresher.refreshCount != 0 {
		t.Fatalf("expected 0 refreshes, got %d", refresher.refreshCount)
	}
}

func TestSourceNearExpiryAndExpiredRefresh(t *testing.T) {
	now := time.Now()
	// Near-expiry (expires in 1 minute, lead time is 2 minutes)
	store := &memoryTokenStore{
		token: &auth.Token{
			AccessToken:  "near-expired-token",
			TokenType:    "Bearer",
			RefreshToken: "refresh-token",
			Expiry:       now.Add(1 * time.Minute),
		},
	}
	refresher := &mockRefresher{}
	src := NewRefreshingTokenSource(store, refresher, WithRefreshLeadTime(2*time.Minute), withNow(func() time.Time { return now }))

	tok, err := src.Token(context.Background())
	if err != nil {
		t.Fatalf("token: %v", err)
	}
	if tok.AccessToken != "refreshed-access-1" {
		t.Fatalf("expected refreshed token, got %s", tok.AccessToken)
	}
	if refresher.refreshCount != 1 {
		t.Fatalf("expected 1 refresh, got %d", refresher.refreshCount)
	}
	if store.token.AccessToken != "refreshed-access-1" {
		t.Fatalf("expected store to hold refreshed token, got %s", store.token.AccessToken)
	}
}

func TestSourceMissingExpiryUsableUntilInvalidated(t *testing.T) {
	store := &memoryTokenStore{
		token: &auth.Token{
			AccessToken:  "no-expiry-token",
			TokenType:    "Bearer",
			RefreshToken: "refresh-token",
			// Zero expiry
		},
	}
	refresher := &mockRefresher{}
	src := NewRefreshingTokenSource(store, refresher)

	// First call should return without refresh
	tok, err := src.Token(context.Background())
	if err != nil {
		t.Fatalf("token: %v", err)
	}
	if tok.AccessToken != "no-expiry-token" {
		t.Fatalf("expected no-expiry-token, got %s", tok.AccessToken)
	}
	if refresher.refreshCount != 0 {
		t.Fatalf("expected 0 refreshes, got %d", refresher.refreshCount)
	}

	// Invalidate and call again -> must refresh
	src.Invalidate("no-expiry-token")

	tok2, err := src.Token(context.Background())
	if err != nil {
		t.Fatalf("token after invalidate: %v", err)
	}
	if tok2.AccessToken != "refreshed-access-1" {
		t.Fatalf("expected refreshed token, got %s", tok2.AccessToken)
	}
	if refresher.refreshCount != 1 {
		t.Fatalf("expected 1 refresh, got %d", refresher.refreshCount)
	}
}

func TestSourceNoRefreshTokenReturnsReauthRequired(t *testing.T) {
	now := time.Now()
	store := &memoryTokenStore{
		token: &auth.Token{
			AccessToken:  "expired-no-refresh",
			TokenType:    "Bearer",
			RefreshToken: "", // Empty refresh token
			Expiry:       now.Add(-time.Minute),
		},
	}
	refresher := &mockRefresher{}
	src := NewRefreshingTokenSource(store, refresher, withNow(func() time.Time { return now }))

	_, err := src.Token(context.Background())
	if err == nil {
		t.Fatal("expected error on expired token with no refresh token")
	}
	if !errors.Is(err, auth.ErrReauthenticationRequired) {
		t.Fatalf("expected ErrReauthenticationRequired, got %v", err)
	}
}

func TestSourceConcurrentRefreshesSingleCall(t *testing.T) {
	now := time.Now()
	store := &memoryTokenStore{
		token: &auth.Token{
			AccessToken:  "expired-token",
			TokenType:    "Bearer",
			RefreshToken: "initial-refresh",
			Expiry:       now.Add(-time.Hour),
		},
	}

	var refreshCalls int32
	refresher := &mockRefresher{
		refreshFn: func(ctx context.Context, token *auth.Token) (*auth.Token, error) {
			atomic.AddInt32(&refreshCalls, 1)
			time.Sleep(50 * time.Millisecond) // simulate latency
			return &auth.Token{
				AccessToken:  "single-refreshed-access",
				TokenType:    "Bearer",
				RefreshToken: "single-refreshed-refresh",
				Expiry:       now.Add(time.Hour),
			}, nil
		},
	}

	src := NewRefreshingTokenSource(store, refresher, withNow(func() time.Time { return now }))

	const numGoroutines = 30
	var wg sync.WaitGroup
	wg.Add(numGoroutines)

	results := make([]*auth.Token, numGoroutines)
	errorsList := make([]error, numGoroutines)

	for i := 0; i < numGoroutines; i++ {
		idx := i
		go func() {
			defer wg.Done()
			tok, err := src.Token(context.Background())
			results[idx] = tok
			errorsList[idx] = err
		}()
	}

	wg.Wait()

	for i, err := range errorsList {
		if err != nil {
			t.Fatalf("goroutine %d failed: %v", i, err)
		}
		if results[i].AccessToken != "single-refreshed-access" {
			t.Fatalf("goroutine %d got unexpected token: %v", i, results[i])
		}
	}

	if atomic.LoadInt32(&refreshCalls) != 1 {
		t.Fatalf("expected exactly 1 refresh call across %d callers, got %d", numGoroutines, atomic.LoadInt32(&refreshCalls))
	}
}

func TestSourceInvalidatingOldTokenDoesNotInvalidateNewToken(t *testing.T) {
	now := time.Now()
	store := &memoryTokenStore{
		token: &auth.Token{
			AccessToken:  "new-token",
			TokenType:    "Bearer",
			RefreshToken: "refresh",
			Expiry:       now.Add(time.Hour),
		},
	}
	refresher := &mockRefresher{}
	src := NewRefreshingTokenSource(store, refresher, withNow(func() time.Time { return now }))

	// Invalidate an OLD token that is not current
	src.Invalidate("old-stale-token")

	tok, err := src.Token(context.Background())
	if err != nil {
		t.Fatalf("token: %v", err)
	}
	if tok.AccessToken != "new-token" {
		t.Fatalf("expected new-token, got %s", tok.AccessToken)
	}
	if refresher.refreshCount != 0 {
		t.Fatalf("old token invalidation triggered an unexpected refresh")
	}
}

func TestSourceSaveFailureDoesNotExposeUnsavedToken(t *testing.T) {
	now := time.Now()
	store := &memoryTokenStore{
		token: &auth.Token{
			AccessToken:  "expired-token",
			TokenType:    "Bearer",
			RefreshToken: "refresh-1",
			Expiry:       now.Add(-time.Hour),
		},
		saveErr: errors.New("disk full"),
	}

	refresher := &mockRefresher{}
	src := NewRefreshingTokenSource(store, refresher, withNow(func() time.Time { return now }))

	_, err := src.Token(context.Background())
	if err == nil {
		t.Fatal("expected error when store.Save fails")
	}
	if !errors.Is(err, store.saveErr) {
		t.Fatalf("expected error wrapping store.saveErr, got: %v", err)
	}

	// Verify the unsaved token was not adopted into cache as valid
	if src.cachedToken != nil && src.cachedToken.AccessToken == "refreshed-access-1" {
		t.Fatal("cachedToken should not have adopted unsaved token")
	}
}

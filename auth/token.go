package auth

import (
	"context"
	"errors"
	"time"
)

var (
	// ErrTokenNotFound is returned when no token exists in a store or the stored token is empty.
	ErrTokenNotFound = errors.New("auth: token not found")

	// ErrReauthenticationRequired indicates that the token is invalid or expired and cannot be refreshed,
	// requiring an interactive re-authentication flow.
	ErrReauthenticationRequired = errors.New("auth: reauthentication required")
)

// Token holds credentials and metadata for an authenticated session.
type Token struct {
	AccessToken  string    `json:"access_token"`
	TokenType    string    `json:"token_type,omitempty"`
	RefreshToken string    `json:"refresh_token,omitempty"`
	Expiry       time.Time `json:"expiry,omitempty"`
}

// Clone returns a copy of the token.
func (t *Token) Clone() *Token {
	if t == nil {
		return nil
	}
	cp := *t
	return &cp
}

// TypeOrDefault returns the token type or "Bearer" if unset.
func (t *Token) TypeOrDefault() string {
	if t == nil || t.TokenType == "" {
		return "Bearer"
	}
	return t.TokenType
}

// Valid reports whether the token is non-nil, has a non-empty access token, and is not expired.
// A zero Expiry time is considered valid until explicitly invalidated.
func (t *Token) Valid() bool {
	if t == nil || t.AccessToken == "" {
		return false
	}
	if t.Expiry.IsZero() {
		return true
	}
	return t.Expiry.After(time.Now())
}

// TokenSource provides access tokens for API requests.
type TokenSource interface {
	Token(ctx context.Context) (*Token, error)
}

// InvalidatableTokenSource is an optional interface implemented by token sources
// that support reactive cache invalidation after an HTTP 401 response.
type InvalidatableTokenSource interface {
	TokenSource
	Invalidate(accessToken string)
}

// TokenStore defines persistent storage for authentication tokens.
type TokenStore interface {
	Load(ctx context.Context) (*Token, error)
	Save(ctx context.Context, token *Token) error
}

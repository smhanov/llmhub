package oauth2

import (
	"errors"
	"fmt"
)

var (
	// ErrAuthorizationPending indicates the user has not yet completed the authorization flow.
	ErrAuthorizationPending = errors.New("oauth2: authorization pending")

	// ErrSlowDown indicates the client is polling too frequently and must increase interval by 5s.
	ErrSlowDown = errors.New("oauth2: slow down")

	// ErrAccessDenied indicates the authorization request was denied by the user or server.
	ErrAccessDenied = errors.New("oauth2: access denied")

	// ErrExpiredToken indicates the device code or token has expired.
	ErrExpiredToken = errors.New("oauth2: expired token")

	// ErrInvalidGrant indicates the provided grant or refresh token is invalid or revoked.
	ErrInvalidGrant = errors.New("oauth2: invalid grant")

	// ErrInvalidClient indicates client authentication failed.
	ErrInvalidClient = errors.New("oauth2: invalid client")

	// ErrInvalidRequest indicates the request is missing a parameter or is otherwise malformed.
	ErrInvalidRequest = errors.New("oauth2: invalid request")

	// ErrUnauthorizedClient indicates the client is not authorized for this grant type.
	ErrUnauthorizedClient = errors.New("oauth2: unauthorized client")

	// ErrUnsupportedGrantType indicates the grant type is not supported.
	ErrUnsupportedGrantType = errors.New("oauth2: unsupported grant type")

	// ErrInvalidScope indicates the requested scope is invalid, unknown, or malformed.
	ErrInvalidScope = errors.New("oauth2: invalid scope")

	// ErrServerResponse indicates an unexpected or malformed response from the OAuth server.
	ErrServerResponse = errors.New("oauth2: server error response")
)

// Error represents an OAuth 2.0 error response (RFC 6749 / RFC 8628).
// It never includes client secrets, codes, or tokens in its formatted string.
type Error struct {
	ErrorCode        string `json:"error"`
	ErrorDescription string `json:"error_description,omitempty"`
	ErrorURI         string `json:"error_uri,omitempty"`
	StatusCode       int    `json:"-"`
}

func (e *Error) Error() string {
	if e == nil {
		return ""
	}
	if e.ErrorDescription != "" {
		return fmt.Sprintf("oauth2: %s: %s", e.ErrorCode, e.ErrorDescription)
	}
	return fmt.Sprintf("oauth2: %s", e.ErrorCode)
}

// Unwrap maps the error code to standard sentinel errors for errors.Is checking.
func (e *Error) Unwrap() error {
	if e == nil {
		return nil
	}
	switch e.ErrorCode {
	case "authorization_pending":
		return ErrAuthorizationPending
	case "slow_down":
		return ErrSlowDown
	case "access_denied":
		return ErrAccessDenied
	case "expired_token":
		return ErrExpiredToken
	case "invalid_grant":
		return ErrInvalidGrant
	case "invalid_client":
		return ErrInvalidClient
	case "invalid_request":
		return ErrInvalidRequest
	case "unauthorized_client":
		return ErrUnauthorizedClient
	case "unsupported_grant_type":
		return ErrUnsupportedGrantType
	case "invalid_scope":
		return ErrInvalidScope
	default:
		return ErrServerResponse
	}
}

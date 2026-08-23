package xai

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/smhanov/llmhub/auth"
)

// FileTokenStore implements auth.TokenStore compatible with the standard Grok CLI auth file format.
// It stores tokens under an issuer::client_id key and preserves unrelated top-level keys and
// unrecognized metadata fields on save.
type FileTokenStore struct {
	path     string
	issuer   string
	clientID string
	mu       sync.RWMutex
}

// FileTokenStoreOption configures FileTokenStore.
type FileTokenStoreOption func(*FileTokenStore)

// WithIssuer overrides the default OAuth issuer used as part of the auth.json map key.
func WithIssuer(issuer string) FileTokenStoreOption {
	return func(s *FileTokenStore) {
		if issuer != "" {
			s.issuer = issuer
		}
	}
}

// WithClientID overrides the default client ID used as part of the auth.json map key.
func WithClientID(clientID string) FileTokenStoreOption {
	return func(s *FileTokenStore) {
		if clientID != "" {
			s.clientID = clientID
		}
	}
}

// NewFileTokenStore creates a new Grok-compatible auth file store.
func NewFileTokenStore(path string, opts ...FileTokenStoreOption) *FileTokenStore {
	s := &FileTokenStore{
		path:     path,
		issuer:   DefaultOAuthIssuer,
		clientID: DefaultClientID,
	}
	for _, opt := range opts {
		if opt != nil {
			opt(s)
		}
	}
	return s
}

// Path returns the filesystem path of the token store.
func (s *FileTokenStore) Path() string {
	return s.path
}

// Load reads and parses the token from the Grok auth JSON file.
func (s *FileTokenStore) Load(ctx context.Context) (*auth.Token, error) {
	if s.path == "" {
		return nil, fmt.Errorf("xai: file store path is empty: %w", auth.ErrTokenNotFound)
	}

	s.mu.RLock()
	defer s.mu.RUnlock()

	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	default:
	}

	data, err := os.ReadFile(s.path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, fmt.Errorf("xai: file store %q: %w", s.path, auth.ErrTokenNotFound)
		}
		return nil, fmt.Errorf("xai: file store %q: %w", s.path, err)
	}

	var root map[string]json.RawMessage
	if err := json.Unmarshal(data, &root); err != nil {
		return nil, fmt.Errorf("xai: file store %q: malformed json: %w", s.path, err)
	}

	scopeKey := fmt.Sprintf("%s::%s", s.issuer, s.clientID)
	entryRaw, exists := root[scopeKey]
	if !exists {
		return nil, fmt.Errorf("xai: file store %q: entry for %s not found: %w", s.path, scopeKey, auth.ErrTokenNotFound)
	}

	var entry map[string]interface{}
	if err := json.Unmarshal(entryRaw, &entry); err != nil {
		return nil, fmt.Errorf("xai: file store %q: malformed entry: %w", s.path, err)
	}

	// Grok auth.json uses "key" for access token, and optionally "access_token"
	accessToken, _ := entry["key"].(string)
	if accessToken == "" {
		accessToken, _ = entry["access_token"].(string)
	}

	refreshToken, _ := entry["refresh_token"].(string)
	tokenType, _ := entry["token_type"].(string)
	if tokenType == "" {
		tokenType = "Bearer"
	}

	var expiry time.Time
	if expiresAtStr, ok := entry["expires_at"].(string); ok && expiresAtStr != "" {
		expiry = parseExpiryString(expiresAtStr)
	}

	if accessToken == "" && refreshToken == "" {
		return nil, fmt.Errorf("xai: file store %q: %w: credentials are empty", s.path, auth.ErrTokenNotFound)
	}

	return &auth.Token{
		AccessToken:  accessToken,
		TokenType:    tokenType,
		RefreshToken: refreshToken,
		Expiry:       expiry,
	}, nil
}

// Save writes the token into the Grok auth JSON file atomically, preserving other entries and metadata.
func (s *FileTokenStore) Save(ctx context.Context, token *auth.Token) error {
	if s.path == "" {
		return fmt.Errorf("xai: file store path is empty")
	}
	if token == nil {
		return fmt.Errorf("xai: file store %q: token is nil", s.path)
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	select {
	case <-ctx.Done():
		return ctx.Err()
	default:
	}

	// Keep unrelated entries as raw JSON. Shared Grok auth files can contain
	// credentials for other clients and non-object top-level metadata, both of
	// which must survive a token rotation unchanged.
	existingRoot := make(map[string]json.RawMessage)
	if data, err := os.ReadFile(s.path); err == nil {
		if err := json.Unmarshal(data, &existingRoot); err != nil {
			return fmt.Errorf("xai: file store %q: malformed json: %w", s.path, err)
		}
	} else if !os.IsNotExist(err) {
		return fmt.Errorf("xai: file store %q: read existing file: %w", s.path, err)
	}

	scopeKey := fmt.Sprintf("%s::%s", s.issuer, s.clientID)
	entry := make(map[string]interface{})
	if existingEntry, ok := existingRoot[scopeKey]; ok {
		if err := json.Unmarshal(existingEntry, &entry); err != nil {
			return fmt.Errorf("xai: file store %q: malformed entry for %s: %w", s.path, scopeKey, err)
		}
	}

	entry["key"] = token.AccessToken
	if token.RefreshToken != "" {
		entry["refresh_token"] = token.RefreshToken
	}
	if token.TokenType != "" {
		entry["token_type"] = token.TokenType
	}
	if !token.Expiry.IsZero() {
		entry["expires_at"] = token.Expiry.UTC().Format("2006-01-02T15:04:05.000Z")
	}

	entryJSON, err := json.Marshal(entry)
	if err != nil {
		return fmt.Errorf("xai: file store %q: marshal entry: %w", s.path, err)
	}
	existingRoot[scopeKey] = entryJSON

	dir := filepath.Dir(s.path)
	if err := os.MkdirAll(dir, 0700); err != nil {
		return fmt.Errorf("xai: file store %q: create dir: %w", s.path, err)
	}

	encoded, err := json.MarshalIndent(existingRoot, "", "  ")
	if err != nil {
		return fmt.Errorf("xai: file store %q: marshal: %w", s.path, err)
	}
	encoded = append(encoded, '\n')

	tmpFile, err := os.CreateTemp(dir, "auth-*.tmp")
	if err != nil {
		return fmt.Errorf("xai: file store %q: create temp file: %w", s.path, err)
	}
	tmpName := tmpFile.Name()
	defer func() {
		if tmpName != "" {
			_ = os.Remove(tmpName)
		}
	}()

	if err := os.Chmod(tmpName, 0600); err != nil {
		_ = tmpFile.Close()
		return fmt.Errorf("xai: file store %q: chmod: %w", s.path, err)
	}

	if _, err := tmpFile.Write(encoded); err != nil {
		_ = tmpFile.Close()
		return fmt.Errorf("xai: file store %q: write: %w", s.path, err)
	}

	if err := tmpFile.Sync(); err != nil {
		_ = tmpFile.Close()
		return fmt.Errorf("xai: file store %q: sync: %w", s.path, err)
	}

	if err := tmpFile.Close(); err != nil {
		return fmt.Errorf("xai: file store %q: close: %w", s.path, err)
	}

	if err := os.Rename(tmpName, s.path); err != nil {
		return fmt.Errorf("xai: file store %q: rename: %w", s.path, err)
	}

	tmpName = "" // Prevent defer cleanup
	return nil
}

func parseExpiryString(s string) time.Time {
	formats := []string{
		"2006-01-02T15:04:05.000Z",
		time.RFC3339Nano,
		time.RFC3339,
		"2006-01-02T15:04:05",
	}
	for _, f := range formats {
		if t, err := time.Parse(f, s); err == nil {
			return t
		}
	}
	return time.Time{}
}

// DefaultAuthPath resolves the path to the Grok/xAI auth file in priority order:
// 1. Explicit path if provided.
// 2. XAI_AUTH_FILE environment variable.
// 3. Existing ~/.grok/auth.json.
// 4. Existing ~/.xgroxy/auth.json (migration fallback).
// 5. Default fallback to ~/.grok/auth.json.
func DefaultAuthPath(explicit string) string {
	if explicit != "" {
		return explicit
	}
	if env := os.Getenv("XAI_AUTH_FILE"); env != "" {
		return env
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return ""
	}

	grokPath := filepath.Join(home, ".grok", "auth.json")
	if _, err := os.Stat(grokPath); err == nil {
		return grokPath
	}

	xgroxyPath := filepath.Join(home, ".xgroxy", "auth.json")
	if _, err := os.Stat(xgroxyPath); err == nil {
		return xgroxyPath
	}

	return grokPath
}

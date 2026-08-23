package auth

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sync"
)

// FileTokenStore implements TokenStore by saving and loading tokens as JSON on the filesystem.
// Writes are performed atomically using a temporary file in the same directory followed by a rename.
// On Unix platforms, directories are created with 0700 permissions and token files with 0600.
//
// FileTokenStore coordinates concurrent file access within a single process via an internal mutex.
// While atomic file writes protect file integrity across processes, cross-process refresh-token
// rotation requires application-level coordination or a custom store.
type FileTokenStore struct {
	path string
	mu   sync.RWMutex
}

// NewFileTokenStore creates a new FileTokenStore targeting the specified path.
func NewFileTokenStore(path string) *FileTokenStore {
	return &FileTokenStore{path: path}
}

// Path returns the filesystem path used by this store.
func (s *FileTokenStore) Path() string {
	return s.path
}

// Load reads and decodes the token from disk.
// Returns ErrTokenNotFound if the file does not exist or if the stored token contains neither
// an access token nor a refresh token.
func (s *FileTokenStore) Load(ctx context.Context) (*Token, error) {
	if s.path == "" {
		return nil, fmt.Errorf("auth: file store path is empty: %w", ErrTokenNotFound)
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
			return nil, fmt.Errorf("auth: file store %q: %w", s.path, ErrTokenNotFound)
		}
		return nil, fmt.Errorf("auth: file store %q: read error: %w", s.path, err)
	}

	var token Token
	if err := json.Unmarshal(data, &token); err != nil {
		return nil, fmt.Errorf("auth: file store %q: malformed json: %w", s.path, err)
	}

	if token.AccessToken == "" && token.RefreshToken == "" {
		return nil, fmt.Errorf("auth: file store %q: %w: token is empty", s.path, ErrTokenNotFound)
	}

	return &token, nil
}

// Save writes the token to disk atomically.
func (s *FileTokenStore) Save(ctx context.Context, token *Token) error {
	if s.path == "" {
		return fmt.Errorf("auth: file store path is empty")
	}
	if token == nil {
		return fmt.Errorf("auth: file store %q: token is nil", s.path)
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	select {
	case <-ctx.Done():
		return ctx.Err()
	default:
	}

	dir := filepath.Dir(s.path)
	if err := os.MkdirAll(dir, 0700); err != nil {
		return fmt.Errorf("auth: file store %q: create directory: %w", s.path, err)
	}

	encoded, err := json.MarshalIndent(token, "", "  ")
	if err != nil {
		return fmt.Errorf("auth: file store %q: marshal token: %w", s.path, err)
	}
	encoded = append(encoded, '\n')

	tmpFile, err := os.CreateTemp(dir, "token-*.tmp")
	if err != nil {
		return fmt.Errorf("auth: file store %q: create temp file: %w", s.path, err)
	}
	tmpName := tmpFile.Name()
	defer func() {
		if tmpName != "" {
			_ = os.Remove(tmpName)
		}
	}()

	if err := os.Chmod(tmpName, 0600); err != nil {
		_ = tmpFile.Close()
		return fmt.Errorf("auth: file store %q: chmod temp file: %w", s.path, err)
	}

	if _, err := tmpFile.Write(encoded); err != nil {
		_ = tmpFile.Close()
		return fmt.Errorf("auth: file store %q: write temp file: %w", s.path, err)
	}

	if err := tmpFile.Sync(); err != nil {
		_ = tmpFile.Close()
		return fmt.Errorf("auth: file store %q: sync temp file: %w", s.path, err)
	}

	if err := tmpFile.Close(); err != nil {
		return fmt.Errorf("auth: file store %q: close temp file: %w", s.path, err)
	}

	if err := os.Rename(tmpName, s.path); err != nil {
		return fmt.Errorf("auth: file store %q: rename temp file: %w", s.path, err)
	}

	tmpName = "" // Prevent defer cleanup
	return nil
}

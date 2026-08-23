package auth

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestFileStoreMissingFile(t *testing.T) {
	tmpDir := t.TempDir()
	store := NewFileTokenStore(filepath.Join(tmpDir, "nonexistent.json"))

	_, err := store.Load(context.Background())
	if err == nil {
		t.Fatal("expected error on missing file")
	}
	if !errors.Is(err, ErrTokenNotFound) {
		t.Fatalf("expected ErrTokenNotFound, got %v", err)
	}
}

func TestFileStoreSaveAndLoad(t *testing.T) {
	tmpDir := t.TempDir()
	path := filepath.Join(tmpDir, "subdir", "auth.json")
	store := NewFileTokenStore(path)

	expiry := time.Now().Add(2 * time.Hour).Truncate(time.Second)
	token := &Token{
		AccessToken:  "secret-access-token",
		TokenType:    "Bearer",
		RefreshToken: "secret-refresh-token",
		Expiry:       expiry,
	}

	err := store.Save(context.Background(), token)
	if err != nil {
		t.Fatalf("save token: %v", err)
	}

	// Verify permissions
	dirInfo, err := os.Stat(filepath.Dir(path))
	if err != nil {
		t.Fatalf("stat dir: %v", err)
	}
	if dirInfo.Mode().Perm() != 0700 {
		t.Errorf("dir mode = %o, want 0700", dirInfo.Mode().Perm())
	}

	fileInfo, err := os.Stat(path)
	if err != nil {
		t.Fatalf("stat file: %v", err)
	}
	if fileInfo.Mode().Perm() != 0600 {
		t.Errorf("file mode = %o, want 0600", fileInfo.Mode().Perm())
	}

	// Verify no temporary files remain
	files, err := os.ReadDir(filepath.Dir(path))
	if err != nil {
		t.Fatalf("read dir: %v", err)
	}
	if len(files) != 1 {
		t.Fatalf("expected 1 file in dir, found %d", len(files))
	}

	loaded, err := store.Load(context.Background())
	if err != nil {
		t.Fatalf("load token: %v", err)
	}

	if loaded.AccessToken != token.AccessToken || loaded.RefreshToken != token.RefreshToken || loaded.TokenType != token.TokenType {
		t.Fatalf("loaded token mismatch: got %+v, want %+v", loaded, token)
	}
	if !loaded.Expiry.Equal(token.Expiry) {
		t.Fatalf("loaded expiry mismatch: got %v, want %v", loaded.Expiry, token.Expiry)
	}
}

func TestFileStoreMalformedJSON(t *testing.T) {
	tmpDir := t.TempDir()
	path := filepath.Join(tmpDir, "corrupted.json")
	if err := os.WriteFile(path, []byte("{not-json"), 0600); err != nil {
		t.Fatalf("write file: %v", err)
	}

	store := NewFileTokenStore(path)
	_, err := store.Load(context.Background())
	if err == nil {
		t.Fatal("expected error on malformed json")
	}
	if !strings.Contains(err.Error(), "malformed json") {
		t.Fatalf("expected malformed json error, got: %v", err)
	}
	if !strings.Contains(err.Error(), path) {
		t.Fatalf("expected error to include file path, got: %v", err)
	}
}

func TestFileStoreEmptyToken(t *testing.T) {
	tmpDir := t.TempDir()
	path := filepath.Join(tmpDir, "empty.json")
	if err := os.WriteFile(path, []byte(`{"access_token":"","refresh_token":""}`), 0600); err != nil {
		t.Fatalf("write file: %v", err)
	}

	store := NewFileTokenStore(path)
	_, err := store.Load(context.Background())
	if err == nil {
		t.Fatal("expected error on empty stored token")
	}
	if !errors.Is(err, ErrTokenNotFound) {
		t.Fatalf("expected ErrTokenNotFound, got %v", err)
	}
}

func TestFileStoreContextCancellation(t *testing.T) {
	tmpDir := t.TempDir()
	path := filepath.Join(tmpDir, "auth.json")
	store := NewFileTokenStore(path)

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	err := store.Save(ctx, &Token{AccessToken: "abc"})
	if err == nil {
		t.Fatal("expected error on canceled context save")
	}

	_, err = store.Load(ctx)
	if err == nil {
		t.Fatal("expected error on canceled context load")
	}
}

func TestFileStoreNoTokensInErrors(t *testing.T) {
	store := NewFileTokenStore("")
	err := store.Save(context.Background(), &Token{AccessToken: "super-secret-token-value"})
	if err != nil && strings.Contains(err.Error(), "super-secret-token-value") {
		t.Fatalf("error leaked token: %v", err)
	}
}

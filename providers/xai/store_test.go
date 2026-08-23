package xai

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/smhanov/llmhub/auth"
)

func TestGrokStoreLoadMissingFile(t *testing.T) {
	tmpDir := t.TempDir()
	store := NewFileTokenStore(filepath.Join(tmpDir, "missing.json"))
	_, err := store.Load(context.Background())
	if err == nil {
		t.Fatal("expected error on missing file")
	}
	if !errors.Is(err, auth.ErrTokenNotFound) {
		t.Fatalf("expected ErrTokenNotFound, got %v", err)
	}
}

func TestGrokStoreSaveAndLoadKeyedByIssuerAndClient(t *testing.T) {
	tmpDir := t.TempDir()
	path := filepath.Join(tmpDir, "sub", "auth.json")
	store := NewFileTokenStore(path)

	expiry := time.Now().Add(2 * time.Hour).Truncate(time.Second)
	token := &auth.Token{
		AccessToken:  "xai-access-key-1",
		TokenType:    "Bearer",
		RefreshToken: "xai-refresh-1",
		Expiry:       expiry,
	}

	err := store.Save(context.Background(), token)
	if err != nil {
		t.Fatalf("save: %v", err)
	}

	// Verify file permissions
	dirInfo, err := os.Stat(filepath.Dir(path))
	if err != nil {
		t.Fatalf("stat dir: %v", err)
	}
	if dirInfo.Mode().Perm() != 0700 {
		t.Errorf("dir perm = %o, want 0700", dirInfo.Mode().Perm())
	}
	fileInfo, err := os.Stat(path)
	if err != nil {
		t.Fatalf("stat file: %v", err)
	}
	if fileInfo.Mode().Perm() != 0600 {
		t.Errorf("file perm = %o, want 0600", fileInfo.Mode().Perm())
	}

	// Read raw JSON and verify key is issuer::client_id
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read file: %v", err)
	}
	var raw map[string]map[string]interface{}
	if err := json.Unmarshal(data, &raw); err != nil {
		t.Fatalf("decode raw json: %v", err)
	}
	expectedScopeKey := DefaultOAuthIssuer + "::" + DefaultClientID
	entry, exists := raw[expectedScopeKey]
	if !exists {
		t.Fatalf("expected scope key %q in JSON, got keys: %+v", expectedScopeKey, raw)
	}
	if entry["key"] != "xai-access-key-1" || entry["refresh_token"] != "xai-refresh-1" {
		t.Fatalf("unexpected entry content: %+v", entry)
	}

	// Load via store
	loaded, err := store.Load(context.Background())
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if loaded.AccessToken != "xai-access-key-1" || loaded.RefreshToken != "xai-refresh-1" {
		t.Fatalf("loaded token mismatch: %+v", loaded)
	}
	if !loaded.Expiry.Equal(expiry) {
		t.Fatalf("expiry mismatch: got %v, want %v", loaded.Expiry, expiry)
	}
}

func TestGrokStorePreservesUnrelatedEntriesAndMetadata(t *testing.T) {
	tmpDir := t.TempDir()
	path := filepath.Join(tmpDir, "auth.json")

	initialJSON := `{
  "unrelated::client": {
    "key": "unrelated-key",
    "some_field": 123
  },
  "https://auth.x.ai::b1a00492-073a-47ea-816f-4c329264a828": {
    "key": "old-key",
    "refresh_token": "old-refresh",
    "email": "user@example.com",
    "account_id": "acc-99"
  }
}`
	if err := os.WriteFile(path, []byte(initialJSON), 0600); err != nil {
		t.Fatalf("write initial json: %v", err)
	}

	store := NewFileTokenStore(path)
	newToken := &auth.Token{
		AccessToken:  "new-key",
		RefreshToken: "new-refresh",
		Expiry:       time.Now().Add(time.Hour),
	}

	if err := store.Save(context.Background(), newToken); err != nil {
		t.Fatalf("save: %v", err)
	}

	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read: %v", err)
	}
	var raw map[string]map[string]interface{}
	if err := json.Unmarshal(data, &raw); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	// Verify unrelated entry was preserved
	unrelated, exists := raw["unrelated::client"]
	if !exists || unrelated["key"] != "unrelated-key" {
		t.Fatalf("unrelated entry was corrupted: %+v", raw)
	}

	// Verify existing metadata was preserved
	xaiEntry := raw["https://auth.x.ai::b1a00492-073a-47ea-816f-4c329264a828"]
	if xaiEntry["key"] != "new-key" || xaiEntry["refresh_token"] != "new-refresh" {
		t.Fatalf("xai entry tokens not updated: %+v", xaiEntry)
	}
	if xaiEntry["email"] != "user@example.com" || xaiEntry["account_id"] != "acc-99" {
		t.Fatalf("metadata was lost: %+v", xaiEntry)
	}
}

func TestGrokStoreDoesNotSelectUnrelatedCredentials(t *testing.T) {
	tmpDir := t.TempDir()
	path := filepath.Join(tmpDir, "auth.json")
	if err := os.WriteFile(path, []byte(`{
  "another-issuer::another-client": {"key":"unrelated-key","refresh_token":"unrelated-refresh"}
}`), 0600); err != nil {
		t.Fatalf("write auth file: %v", err)
	}

	_, err := NewFileTokenStore(path).Load(context.Background())
	if !errors.Is(err, auth.ErrTokenNotFound) {
		t.Fatalf("expected ErrTokenNotFound rather than another client's token, got %v", err)
	}
}

func TestGrokStoreSavePreservesNonObjectTopLevelValues(t *testing.T) {
	tmpDir := t.TempDir()
	path := filepath.Join(tmpDir, "auth.json")
	initialJSON := `{
  "version": 3,
  "metadata": ["preserve", true],
  "https://auth.x.ai::b1a00492-073a-47ea-816f-4c329264a828": {"key":"old-key","email":"user@example.com"}
}`
	if err := os.WriteFile(path, []byte(initialJSON), 0600); err != nil {
		t.Fatalf("write auth file: %v", err)
	}

	store := NewFileTokenStore(path)
	if err := store.Save(context.Background(), &auth.Token{AccessToken: "new-key", RefreshToken: "new-refresh"}); err != nil {
		t.Fatalf("save: %v", err)
	}

	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read auth file: %v", err)
	}
	var root map[string]json.RawMessage
	if err := json.Unmarshal(data, &root); err != nil {
		t.Fatalf("unmarshal root: %v", err)
	}
	if got := string(root["version"]); got != "3" {
		t.Fatalf("version = %s, want 3", got)
	}
	var metadata []interface{}
	if err := json.Unmarshal(root["metadata"], &metadata); err != nil {
		t.Fatalf("unmarshal metadata: %v", err)
	}
	if len(metadata) != 2 || metadata[0] != "preserve" || metadata[1] != true {
		t.Fatalf("metadata = %#v", metadata)
	}
}

func TestGrokStoreSaveRejectsMalformedExistingFile(t *testing.T) {
	tmpDir := t.TempDir()
	path := filepath.Join(tmpDir, "auth.json")
	malformed := []byte(`{"not-valid":`)
	if err := os.WriteFile(path, malformed, 0600); err != nil {
		t.Fatalf("write malformed auth file: %v", err)
	}

	err := NewFileTokenStore(path).Save(context.Background(), &auth.Token{AccessToken: "new-key"})
	if err == nil {
		t.Fatal("expected malformed existing file to prevent save")
	}
	data, readErr := os.ReadFile(path)
	if readErr != nil {
		t.Fatalf("read malformed auth file: %v", readErr)
	}
	if string(data) != string(malformed) {
		t.Fatalf("malformed file was overwritten: %q", data)
	}
}

func TestDefaultAuthPath(t *testing.T) {
	// 1. Explicit path
	if DefaultAuthPath("/custom/path.json") != "/custom/path.json" {
		t.Errorf("expected explicit path")
	}

	// 2. Env variable
	os.Setenv("XAI_AUTH_FILE", "/env/path.json")
	defer os.Unsetenv("XAI_AUTH_FILE")
	if DefaultAuthPath("") != "/env/path.json" {
		t.Errorf("expected env path")
	}
	os.Unsetenv("XAI_AUTH_FILE")

	// 3. Fallback contains .grok
	path := DefaultAuthPath("")
	if !strings.Contains(path, ".grok") && !strings.Contains(path, ".xgroxy") {
		t.Errorf("unexpected default path: %s", path)
	}
}

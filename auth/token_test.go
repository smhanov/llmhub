package auth

import (
	"testing"
	"time"
)

func TestTokenClone(t *testing.T) {
	tok := &Token{
		AccessToken:  "access-1",
		TokenType:    "Bearer",
		RefreshToken: "refresh-1",
		Expiry:       time.Now().Add(time.Hour),
	}
	clone := tok.Clone()
	if clone == tok {
		t.Fatal("expected clone to return a new pointer")
	}
	if *clone != *tok {
		t.Fatalf("expected cloned values to match: got %+v, want %+v", clone, tok)
	}

	var nilTok *Token
	if nilTok.Clone() != nil {
		t.Fatal("expected nil token clone to be nil")
	}
}

func TestTokenTypeOrDefault(t *testing.T) {
	tok := &Token{TokenType: "Custom"}
	if tok.TypeOrDefault() != "Custom" {
		t.Fatalf("expected Custom, got %s", tok.TypeOrDefault())
	}
	tokEmpty := &Token{}
	if tokEmpty.TypeOrDefault() != "Bearer" {
		t.Fatalf("expected Bearer, got %s", tokEmpty.TypeOrDefault())
	}
	var nilTok *Token
	if nilTok.TypeOrDefault() != "Bearer" {
		t.Fatalf("expected Bearer, got %s", nilTok.TypeOrDefault())
	}
}

func TestTokenValid(t *testing.T) {
	var nilTok *Token
	if nilTok.Valid() {
		t.Fatal("nil token should not be valid")
	}

	emptyTok := &Token{}
	if emptyTok.Valid() {
		t.Fatal("empty access token should not be valid")
	}

	zeroExpiryTok := &Token{AccessToken: "token"}
	if !zeroExpiryTok.Valid() {
		t.Fatal("zero expiry token with access token should be valid")
	}

	futureTok := &Token{AccessToken: "token", Expiry: time.Now().Add(time.Hour)}
	if !futureTok.Valid() {
		t.Fatal("future expiry token should be valid")
	}

	pastTok := &Token{AccessToken: "token", Expiry: time.Now().Add(-time.Hour)}
	if pastTok.Valid() {
		t.Fatal("past expiry token should not be valid")
	}
}

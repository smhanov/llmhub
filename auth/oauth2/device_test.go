package oauth2

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"net/url"
	"slices"
	"sync/atomic"
	"testing"
	"time"

	"github.com/smhanov/llmhub/auth"
)

func TestDeviceFlowStartSendsRequiredFields(t *testing.T) {
	now := time.Date(2026, time.August, 23, 12, 0, 0, 0, time.UTC)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			t.Fatalf("method = %s, want POST", r.Method)
		}
		if err := r.ParseForm(); err != nil {
			t.Fatalf("parse form: %v", err)
		}
		if got := r.Form.Get("client_id"); got != "real-client" {
			t.Fatalf("client_id = %q", got)
		}
		if got := r.Form.Get("scope"); got != "openid offline_access" {
			t.Fatalf("scope = %q", got)
		}
		if got := r.Form.Get("audience"); got != "https://api.example.test" {
			t.Fatalf("audience = %q", got)
		}
		if got := r.Form.Get("client_id"); got == "attempted-override" {
			t.Fatal("extra parameters must not override client_id")
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"device_code":"device-code","user_code":"ABCD-EFGH","verification_uri":"https://verify.example.test","verification_uri_complete":"https://verify.example.test/?code=ABCD-EFGH","expires_in":"120","interval":3}`))
	}))
	defer server.Close()

	flow := NewDeviceFlow(DeviceFlowConfig{
		DeviceAuthURL: server.URL,
		ClientID:      "real-client",
		Scopes:        []string{"openid", "offline_access"},
		ExtraDeviceAuthParams: url.Values{
			"audience":  {"https://api.example.test"},
			"client_id": {"attempted-override"},
		},
		now: func() time.Time { return now },
	})

	authz, err := flow.Start(context.Background())
	if err != nil {
		t.Fatalf("start: %v", err)
	}
	if authz.DeviceCode != "device-code" || authz.UserCode != "ABCD-EFGH" {
		t.Fatalf("unexpected authorization: %+v", authz)
	}
	if authz.Interval != 3 || !authz.Expiry.Equal(now.Add(120*time.Second)) {
		t.Fatalf("unexpected interval/expiry: %+v", authz)
	}
}

func TestDeviceFlowWaitHandlesPendingAndSlowDown(t *testing.T) {
	now := time.Date(2026, time.August, 23, 12, 0, 0, 0, time.UTC)
	var polls int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := r.ParseForm(); err != nil {
			t.Fatalf("parse form: %v", err)
		}
		if got := r.Form.Get("device_code"); got != "device-code" {
			t.Fatalf("device_code = %q", got)
		}
		switch atomic.AddInt32(&polls, 1) {
		case 1:
			w.WriteHeader(http.StatusBadRequest)
			_, _ = w.Write([]byte(`{"error":"authorization_pending"}`))
		case 2:
			w.WriteHeader(http.StatusBadRequest)
			_, _ = w.Write([]byte(`{"error":"slow_down"}`))
		default:
			_, _ = w.Write([]byte(`{"access_token":"new-access","token_type":"Bearer","refresh_token":"new-refresh","expires_in":"3600"}`))
		}
	}))
	defer server.Close()

	var sleeps []time.Duration
	flow := NewDeviceFlow(DeviceFlowConfig{
		TokenURL: server.URL,
		ClientID: "client",
		now:      func() time.Time { return now },
		sleeper: func(ctx context.Context, d time.Duration) error {
			sleeps = append(sleeps, d)
			return nil
		},
	})

	token, err := flow.Wait(context.Background(), &DeviceAuthorization{
		DeviceCode: "device-code",
		Interval:   3,
		Expiry:     now.Add(time.Hour),
	})
	if err != nil {
		t.Fatalf("wait: %v", err)
	}
	if token.AccessToken != "new-access" || token.RefreshToken != "new-refresh" {
		t.Fatalf("unexpected token: %+v", token)
	}
	if got, want := sleeps, []time.Duration{3 * time.Second, 3 * time.Second, 8 * time.Second}; !slices.Equal(got, want) {
		t.Fatalf("sleeps = %v, want %v", got, want)
	}
	if got := atomic.LoadInt32(&polls); got != 3 {
		t.Fatalf("polls = %d, want 3", got)
	}
}

func TestDeviceFlowWaitReturnsTerminalOAuthError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		_, _ = w.Write([]byte(`{"error":"access_denied","error_description":"denied"}`))
	}))
	defer server.Close()

	flow := NewDeviceFlow(DeviceFlowConfig{
		TokenURL: server.URL,
		ClientID: "client",
		sleeper:  func(context.Context, time.Duration) error { return nil },
	})
	_, err := flow.Wait(context.Background(), &DeviceAuthorization{
		DeviceCode: "device-code",
		Expiry:     time.Now().Add(time.Hour),
	})
	if !errors.Is(err, ErrAccessDenied) {
		t.Fatalf("expected ErrAccessDenied, got %v", err)
	}
}

func TestDeviceFlowRefreshPreservesRefreshToken(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := r.ParseForm(); err != nil {
			t.Fatalf("parse form: %v", err)
		}
		if got := r.Form.Get("grant_type"); got != grantTypeRefreshToken {
			t.Fatalf("grant_type = %q", got)
		}
		if got := r.Form.Get("refresh_token"); got != "old-refresh" {
			t.Fatalf("refresh_token = %q", got)
		}
		_, _ = w.Write([]byte(`{"access_token":"new-access","expires_in":3600}`))
	}))
	defer server.Close()

	now := time.Date(2026, time.August, 23, 12, 0, 0, 0, time.UTC)
	flow := NewDeviceFlow(DeviceFlowConfig{
		TokenURL: server.URL,
		ClientID: "client",
		now:      func() time.Time { return now },
	})
	token, err := flow.Refresh(context.Background(), &auth.Token{RefreshToken: "old-refresh"})
	if err != nil {
		t.Fatalf("refresh: %v", err)
	}
	if token.AccessToken != "new-access" || token.RefreshToken != "old-refresh" || token.TokenType != "Bearer" {
		t.Fatalf("unexpected token: %+v", token)
	}
	if !token.Expiry.Equal(now.Add(time.Hour)) {
		t.Fatalf("expiry = %v, want %v", token.Expiry, now.Add(time.Hour))
	}
}

func TestDeviceFlowStartRejectsMissingResponseFields(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"device_code":"only-device-code"}`))
	}))
	defer server.Close()

	flow := NewDeviceFlow(DeviceFlowConfig{DeviceAuthURL: server.URL, ClientID: "client"})
	_, err := flow.Start(context.Background())
	if !errors.Is(err, ErrServerResponse) {
		t.Fatalf("expected ErrServerResponse, got %v", err)
	}
}

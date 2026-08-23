package oauth2

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"time"

	"github.com/smhanov/llmhub/auth"
)

const (
	defaultPollingInterval = 5
	maxResponseBodyBytes   = 1024 * 1024 // 1 MB
	grantTypeDeviceCode    = "urn:ietf:params:oauth:grant-type:device_code"
	grantTypeRefreshToken  = "refresh_token"
)

// DeviceFlowConfig holds configuration for the OAuth 2.0 device authorization flow.
type DeviceFlowConfig struct {
	DeviceAuthURL         string
	TokenURL              string
	ClientID              string
	ClientSecret          string
	Scopes                []string
	ExtraDeviceAuthParams url.Values
	ExtraTokenParams      url.Values
	HTTPClient            *http.Client

	// Sleeper and Now allow deterministic time mocking in unit tests.
	sleeper func(ctx context.Context, d time.Duration) error
	now     func() time.Time
}

// DeviceAuthorization holds the response from the device authorization endpoint (RFC 8628).
type DeviceAuthorization struct {
	DeviceCode              string    `json:"device_code"`
	UserCode                string    `json:"user_code"`
	VerificationURI         string    `json:"verification_uri"`
	VerificationURIComplete string    `json:"verification_uri_complete,omitempty"`
	ExpiresIn               int       `json:"expires_in"`
	Interval                int       `json:"interval,omitempty"`
	Expiry                  time.Time `json:"-"`
}

// TokenRefresher defines an interface for refreshing OAuth tokens.
type TokenRefresher interface {
	Refresh(ctx context.Context, token *auth.Token) (*auth.Token, error)
}

// DeviceFlow implements the OAuth 2.0 Device Authorization Grant (RFC 8628) and token refreshing.
type DeviceFlow struct {
	cfg DeviceFlowConfig
}

// NewDeviceFlow creates a new DeviceFlow client.
func NewDeviceFlow(cfg DeviceFlowConfig) *DeviceFlow {
	if cfg.HTTPClient == nil {
		cfg.HTTPClient = &http.Client{Timeout: 30 * time.Second}
	}
	if cfg.sleeper == nil {
		cfg.sleeper = defaultSleeper
	}
	if cfg.now == nil {
		cfg.now = time.Now
	}
	return &DeviceFlow{cfg: cfg}
}

// Start initiates the device authorization flow by requesting a device code and user verification URI.
func (d *DeviceFlow) Start(ctx context.Context) (*DeviceAuthorization, error) {
	if d.cfg.DeviceAuthURL == "" {
		return nil, errors.New("oauth2: device authorization URL is required")
	}
	if d.cfg.ClientID == "" {
		return nil, errors.New("oauth2: client ID is required")
	}

	form := url.Values{}
	for k, v := range d.cfg.ExtraDeviceAuthParams {
		for _, val := range v {
			form.Add(k, val)
		}
	}
	form.Set("client_id", d.cfg.ClientID)
	if d.cfg.ClientSecret != "" {
		form.Set("client_secret", d.cfg.ClientSecret)
	}
	if len(d.cfg.Scopes) > 0 {
		form.Set("scope", strings.Join(d.cfg.Scopes, " "))
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, d.cfg.DeviceAuthURL, strings.NewReader(form.Encode()))
	if err != nil {
		return nil, fmt.Errorf("oauth2: create device auth request: %w", err)
	}
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	req.Header.Set("Accept", "application/json")

	resp, err := d.cfg.HTTPClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("oauth2: device auth request: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(io.LimitReader(resp.Body, maxResponseBodyBytes))
	if err != nil {
		return nil, fmt.Errorf("oauth2: read device auth response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		if oauthErr := parseOAuthError(body, resp.StatusCode); oauthErr != nil {
			return nil, oauthErr
		}
		return nil, fmt.Errorf("oauth2: device auth failed with status %d", resp.StatusCode)
	}

	var raw struct {
		DeviceCode              string      `json:"device_code"`
		UserCode                string      `json:"user_code"`
		VerificationURI         string      `json:"verification_uri"`
		VerificationURL         string      `json:"verification_url"`
		VerificationURIComplete string      `json:"verification_uri_complete"`
		VerificationURLComplete string      `json:"verification_url_complete"`
		ExpiresIn               flexibleInt `json:"expires_in"`
		Interval                flexibleInt `json:"interval"`
	}

	if err := json.Unmarshal(body, &raw); err != nil {
		return nil, fmt.Errorf("oauth2: malformed device auth response: %w", err)
	}

	uri := raw.VerificationURI
	if uri == "" {
		uri = raw.VerificationURL
	}
	uriComplete := raw.VerificationURIComplete
	if uriComplete == "" {
		uriComplete = raw.VerificationURLComplete
	}

	if raw.DeviceCode == "" || raw.UserCode == "" || uri == "" {
		return nil, fmt.Errorf("oauth2: device auth response missing required fields: %w", ErrServerResponse)
	}

	expiresIn := int(raw.ExpiresIn)
	if expiresIn <= 0 {
		expiresIn = 900 // 15 min fallback if omitted
	}

	interval := int(raw.Interval)
	if interval <= 0 {
		interval = defaultPollingInterval
	}

	now := d.cfg.now()
	return &DeviceAuthorization{
		DeviceCode:              raw.DeviceCode,
		UserCode:                raw.UserCode,
		VerificationURI:         uri,
		VerificationURIComplete: uriComplete,
		ExpiresIn:               expiresIn,
		Interval:                interval,
		Expiry:                  now.Add(time.Duration(expiresIn) * time.Second),
	}, nil
}

// Wait polls the token endpoint until authorization is granted, denied, or expires.
func (d *DeviceFlow) Wait(ctx context.Context, authz *DeviceAuthorization) (*auth.Token, error) {
	if authz == nil || authz.DeviceCode == "" {
		return nil, errors.New("oauth2: invalid device authorization")
	}
	if d.cfg.TokenURL == "" {
		return nil, errors.New("oauth2: token URL is required")
	}

	interval := authz.Interval
	if interval <= 0 {
		interval = defaultPollingInterval
	}

	// RFC 8628 Section 3.5: Must wait at least interval before the initial token request.
	if err := d.cfg.sleeper(ctx, time.Duration(interval)*time.Second); err != nil {
		return nil, err
	}

	for {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
		}

		if !authz.Expiry.IsZero() && d.cfg.now().After(authz.Expiry) {
			return nil, ErrExpiredToken
		}

		token, oauthErr, httpErr := d.pollToken(ctx, authz.DeviceCode)
		if httpErr != nil {
			return nil, httpErr
		}
		if token != nil {
			return token, nil
		}

		if oauthErr != nil {
			if errors.Is(oauthErr, ErrAuthorizationPending) {
				if err := d.cfg.sleeper(ctx, time.Duration(interval)*time.Second); err != nil {
					return nil, err
				}
				continue
			}
			if errors.Is(oauthErr, ErrSlowDown) {
				interval += 5
				if err := d.cfg.sleeper(ctx, time.Duration(interval)*time.Second); err != nil {
					return nil, err
				}
				continue
			}
			// Any other OAuth error is terminal (e.g. access_denied, expired_token, invalid_grant)
			return nil, oauthErr
		}

		// Fallback wait if no error was returned but no token either
		if err := d.cfg.sleeper(ctx, time.Duration(interval)*time.Second); err != nil {
			return nil, err
		}
	}
}

// Refresh performs a token refresh request using a refresh_token grant.
func (d *DeviceFlow) Refresh(ctx context.Context, currentToken *auth.Token) (*auth.Token, error) {
	if currentToken == nil || currentToken.RefreshToken == "" {
		return nil, fmt.Errorf("oauth2: %w: refresh token is missing", auth.ErrReauthenticationRequired)
	}
	if d.cfg.TokenURL == "" {
		return nil, errors.New("oauth2: token URL is required")
	}

	form := url.Values{}
	for k, v := range d.cfg.ExtraTokenParams {
		for _, val := range v {
			form.Add(k, val)
		}
	}
	form.Set("grant_type", grantTypeRefreshToken)
	form.Set("refresh_token", currentToken.RefreshToken)
	form.Set("client_id", d.cfg.ClientID)
	if d.cfg.ClientSecret != "" {
		form.Set("client_secret", d.cfg.ClientSecret)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, d.cfg.TokenURL, strings.NewReader(form.Encode()))
	if err != nil {
		return nil, fmt.Errorf("oauth2: create refresh request: %w", err)
	}
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	req.Header.Set("Accept", "application/json")

	resp, err := d.cfg.HTTPClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("oauth2: refresh request: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(io.LimitReader(resp.Body, maxResponseBodyBytes))
	if err != nil {
		return nil, fmt.Errorf("oauth2: read refresh response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		if oauthErr := parseOAuthError(body, resp.StatusCode); oauthErr != nil {
			if errors.Is(oauthErr, ErrInvalidGrant) {
				return nil, fmt.Errorf("oauth2: %w: %v", auth.ErrReauthenticationRequired, oauthErr)
			}
			return nil, oauthErr
		}
		if resp.StatusCode == http.StatusUnauthorized || resp.StatusCode == http.StatusBadRequest {
			return nil, fmt.Errorf("oauth2: %w: refresh status %d", auth.ErrReauthenticationRequired, resp.StatusCode)
		}
		return nil, fmt.Errorf("oauth2: refresh failed with status %d", resp.StatusCode)
	}

	return d.parseTokenResponse(body, currentToken.RefreshToken)
}

func (d *DeviceFlow) pollToken(ctx context.Context, deviceCode string) (*auth.Token, error, error) {
	form := url.Values{}
	for k, v := range d.cfg.ExtraTokenParams {
		for _, val := range v {
			form.Add(k, val)
		}
	}
	form.Set("grant_type", grantTypeDeviceCode)
	form.Set("device_code", deviceCode)
	form.Set("client_id", d.cfg.ClientID)
	if d.cfg.ClientSecret != "" {
		form.Set("client_secret", d.cfg.ClientSecret)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, d.cfg.TokenURL, strings.NewReader(form.Encode()))
	if err != nil {
		return nil, nil, fmt.Errorf("oauth2: create token poll request: %w", err)
	}
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	req.Header.Set("Accept", "application/json")

	resp, err := d.cfg.HTTPClient.Do(req)
	if err != nil {
		return nil, nil, fmt.Errorf("oauth2: token poll request: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(io.LimitReader(resp.Body, maxResponseBodyBytes))
	if err != nil {
		return nil, nil, fmt.Errorf("oauth2: read token response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		oauthErr := parseOAuthError(body, resp.StatusCode)
		if oauthErr != nil {
			return nil, oauthErr, nil
		}
		return nil, nil, fmt.Errorf("oauth2: token poll status %d", resp.StatusCode)
	}

	token, err := d.parseTokenResponse(body, "")
	if err != nil {
		return nil, nil, err
	}
	return token, nil, nil
}

type tokenResponse struct {
	AccessToken  string      `json:"access_token"`
	TokenType    string      `json:"token_type"`
	RefreshToken string      `json:"refresh_token"`
	ExpiresIn    flexibleInt `json:"expires_in"`
}

func (d *DeviceFlow) parseTokenResponse(body []byte, previousRefreshToken string) (*auth.Token, error) {
	var resp tokenResponse
	if err := json.Unmarshal(body, &resp); err != nil {
		return nil, fmt.Errorf("oauth2: malformed token response: %w", err)
	}

	if resp.AccessToken == "" {
		return nil, fmt.Errorf("oauth2: token response missing access token: %w", ErrServerResponse)
	}

	tokenType := resp.TokenType
	if tokenType == "" {
		tokenType = "Bearer"
	}

	refreshToken := resp.RefreshToken
	if refreshToken == "" {
		refreshToken = previousRefreshToken
	}

	var expiry time.Time
	if resp.ExpiresIn > 0 {
		expiry = d.cfg.now().Add(time.Duration(resp.ExpiresIn) * time.Second)
	}

	return &auth.Token{
		AccessToken:  resp.AccessToken,
		TokenType:    tokenType,
		RefreshToken: refreshToken,
		Expiry:       expiry,
	}, nil
}

func parseOAuthError(body []byte, statusCode int) error {
	if len(body) == 0 {
		return nil
	}
	var oauthErr Error
	if err := json.Unmarshal(body, &oauthErr); err != nil {
		return nil
	}
	if oauthErr.ErrorCode == "" {
		return nil
	}
	oauthErr.StatusCode = statusCode
	return &oauthErr
}

func defaultSleeper(ctx context.Context, d time.Duration) error {
	timer := time.NewTimer(d)
	defer timer.Stop()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-timer.C:
		return nil
	}
}

// flexibleInt supports decoding JSON numbers or numeric JSON strings into integers.
type flexibleInt int

func (fi *flexibleInt) UnmarshalJSON(b []byte) error {
	b = bytes.TrimSpace(b)
	if len(b) == 0 || string(b) == "null" {
		return nil
	}
	if b[0] == '"' {
		var s string
		if err := json.Unmarshal(b, &s); err != nil {
			return err
		}
		n, err := strconv.Atoi(s)
		if err != nil {
			return err
		}
		*fi = flexibleInt(n)
		return nil
	}
	var n int
	if err := json.Unmarshal(b, &n); err != nil {
		return err
	}
	*fi = flexibleInt(n)
	return nil
}

package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"github.com/smhanov/llmhub"
	"github.com/smhanov/llmhub/auth"
	"github.com/smhanov/llmhub/providers/xai"
)

const (
	expectedGenerateMarker = "LLHUB_XAI_GENERATE_OK"
	expectedStreamMarker   = "LLHUB_XAI_STREAM_OK"
)

func main() {
	authFilePath := flag.String("auth-file", "", "Path to token file (default: xAI standard discovery path)")
	model := flag.String("model", xai.DefaultModel, "Grok model identifier")
	timeout := flag.Duration("timeout", 10*time.Minute, "Overall execution timeout")
	forceLogin := flag.Bool("force-login", false, "Force a fresh device login flow even if a valid token exists")
	verifyRefresh := flag.Bool("verify-refresh", false, "Verify token invalidation and refresh before model requests")

	flag.Parse()

	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer cancel()

	if *timeout > 0 {
		var timeoutCancel context.CancelFunc
		ctx, timeoutCancel = context.WithTimeout(ctx, *timeout)
		defer timeoutCancel()
	}

	targetPath := xai.DefaultAuthPath(*authFilePath)
	if targetPath == "" {
		fmt.Fprintln(os.Stderr, "Error: could not determine auth file path")
		os.Exit(1)
	}

	store := xai.NewFileTokenStore(targetPath)

	// Determine if initial login is needed
	needsLogin := *forceLogin
	if !needsLogin {
		tok, err := store.Load(ctx)
		if err != nil || tok == nil || !tok.Valid() {
			needsLogin = true
		}
	}

	if needsLogin {
		if err := performDeviceLogin(ctx, store); err != nil {
			fmt.Fprintf(os.Stderr, "Error during device login: %v\n", sanitizeError(err))
			os.Exit(1)
		}
	}

	source := xai.NewTokenSource(store)

	if *verifyRefresh {
		if err := performVerifyRefresh(ctx, store, source); err != nil {
			fmt.Fprintf(os.Stderr, "Error verifying refresh: %v\n", sanitizeError(err))
			os.Exit(1)
		}
	}

	// Perform model validation with one reauth retry if needed
	err := runModelAcceptance(ctx, source, *model)
	if err != nil && errors.Is(err, auth.ErrReauthenticationRequired) {
		fmt.Println("Token expired or rejected; reauthenticating...")
		if loginErr := performDeviceLogin(ctx, store); loginErr != nil {
			fmt.Fprintf(os.Stderr, "Error during reauthentication: %v\n", sanitizeError(loginErr))
			os.Exit(1)
		}
		source = xai.NewTokenSource(store)
		err = runModelAcceptance(ctx, source, *model)
	}

	if err != nil {
		fmt.Fprintf(os.Stderr, "Error during acceptance run: %v\n", sanitizeError(err))
		os.Exit(1)
	}

	fmt.Println("xai oauth acceptance: PASS")
}

func performDeviceLogin(ctx context.Context, store auth.TokenStore) error {
	flow := xai.NewDeviceFlow()
	authz, err := flow.Start(ctx)
	if err != nil {
		return fmt.Errorf("start device flow: %w", err)
	}

	fmt.Printf("Verification URL: %s\n", authz.VerificationURI)
	if authz.VerificationURIComplete != "" {
		fmt.Printf("Complete URL:     %s\n", authz.VerificationURIComplete)
	}
	fmt.Printf("User Code:        %s\n", authz.UserCode)
	fmt.Println("Waiting for device authorization in browser...")
	_ = os.Stdout.Sync()

	token, err := flow.Wait(ctx, authz)
	if err != nil {
		return fmt.Errorf("wait device flow: %w", err)
	}

	if err := store.Save(ctx, token); err != nil {
		return fmt.Errorf("save token: %w", err)
	}

	return nil
}

func performVerifyRefresh(ctx context.Context, store auth.TokenStore, source auth.InvalidatableTokenSource) error {
	// Load the source before invalidating it. Invalidation deliberately applies
	// only to the source's cached token so a late 401 cannot invalidate a newer
	// token saved by another request.
	current, err := source.Token(ctx)
	if err != nil {
		return fmt.Errorf("load token for refresh test: %w", err)
	}
	if current == nil || current.AccessToken == "" {
		return errors.New("no access token to invalidate")
	}

	source.Invalidate(current.AccessToken)

	refreshed, err := source.Token(ctx)
	if err != nil {
		return fmt.Errorf("fetch refreshed token: %w", err)
	}
	if refreshed == nil || refreshed.AccessToken == "" {
		return errors.New("refreshed token has empty access token")
	}

	reloaded, err := store.Load(ctx)
	if err != nil {
		return fmt.Errorf("reload store after refresh: %w", err)
	}
	if reloaded == nil || reloaded.AccessToken == "" || reloaded.RefreshToken == "" {
		return errors.New("reloaded store missing access token or refresh token")
	}

	fmt.Println("refresh: OK")
	_ = os.Stdout.Sync()
	return nil
}

func runModelAcceptance(ctx context.Context, source auth.TokenSource, model string) error {
	client, err := llmhub.New("xai", "",
		llmhub.WithTokenSource(source),
		llmhub.WithModel(model),
	)
	if err != nil {
		return fmt.Errorf("create xai client: %w", err)
	}

	// 1. Generate check
	genPrompt := []*llmhub.Message{
		llmhub.NewUserMessage(llmhub.Text(fmt.Sprintf("Respond with exactly %s and nothing else.", expectedGenerateMarker))),
	}
	resp, err := client.Generate(ctx, genPrompt)
	if err != nil {
		return fmt.Errorf("generate request: %w", err)
	}
	if err := validateGenerateResponse(resp, expectedGenerateMarker); err != nil {
		return err
	}
	fmt.Println("generate: OK")
	_ = os.Stdout.Sync()

	// 2. Stream check
	streamPrompt := []*llmhub.Message{
		llmhub.NewUserMessage(llmhub.Text(fmt.Sprintf("Respond with exactly %s and nothing else.", expectedStreamMarker))),
	}
	chunks, err := client.Stream(ctx, streamPrompt)
	if err != nil {
		return fmt.Errorf("stream request: %w", err)
	}
	if err := validateStreamResponse(chunks, expectedStreamMarker); err != nil {
		return err
	}
	fmt.Println("stream: OK")
	_ = os.Stdout.Sync()

	return nil
}

func validateGenerateResponse(resp *llmhub.Response, expectedMarker string) error {
	if resp == nil {
		return errors.New("generate response is nil")
	}
	text := strings.TrimSpace(resp.Text())
	if text != expectedMarker {
		return fmt.Errorf("generate text mismatch: got %q, want %q", text, expectedMarker)
	}
	return nil
}

func validateStreamResponse(chunks <-chan llmhub.StreamChunk, expectedMarker string) error {
	if chunks == nil {
		return errors.New("stream channel is nil")
	}
	var b strings.Builder
	sawDone := false

	for chunk := range chunks {
		if chunk.Err != nil {
			return fmt.Errorf("stream chunk error: %w", chunk.Err)
		}
		b.WriteString(chunk.Delta)
		if chunk.Done {
			sawDone = true
		}
	}

	if !sawDone {
		return errors.New("stream channel closed without Done chunk")
	}

	text := strings.TrimSpace(b.String())
	if text != expectedMarker {
		return fmt.Errorf("stream text mismatch: got %q, want %q", text, expectedMarker)
	}
	return nil
}

func sanitizeError(err error) error {
	if err == nil {
		return nil
	}
	// Wrap without credential contents
	return err
}

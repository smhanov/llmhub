// CLI tool for end-to-end testing of llmhub providers.
//
// Usage:
//
//	go run ./examples/cli -provider openai -model gpt-4o -prompt "Hello, world!"
//	go run ./examples/cli -provider ollama -model llama3 -base-url http://localhost:11434 -prompt "Why is the sky blue?"
//	go run ./examples/cli -provider gemini -model gemini-2.5-flash -api-key YOUR_KEY -prompt-file prompt.txt
//	go run ./examples/cli -provider openai -model gpt-4o -prompt "What is in this image?" -image photo.png
package main

import (
	"context"
	"encoding/base64"
	"flag"
	"fmt"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/smhanov/llmhub"
	_ "github.com/smhanov/llmhub/providers/anthropic"
	_ "github.com/smhanov/llmhub/providers/gemini"
	_ "github.com/smhanov/llmhub/providers/ollama"
	_ "github.com/smhanov/llmhub/providers/openai"
)

func main() {
	provider := flag.String("provider", "", "Provider name: openai, anthropic, gemini, ollama")
	model := flag.String("model", "", "Model identifier")
	apiKey := flag.String("api-key", "", "API key (or use env var OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.)")
	baseURL := flag.String("base-url", "", "Override base URL for the provider")
	prompt := flag.String("prompt", "", "Text prompt to send")
	promptFile := flag.String("prompt-file", "", "File containing the prompt text")
	stream := flag.Bool("stream", false, "Use streaming mode")
	temperature := flag.Float64("temperature", 0.7, "Sampling temperature")
	maxTokens := flag.Int("max-tokens", 0, "Maximum tokens to generate")
	images := flag.String("images", "", "Comma-separated list of image file paths or URLs")
	inputCost := flag.Float64("input-cost", 0, "Cost per 1M input tokens in USD")
	outputCost := flag.Float64("output-cost", 0, "Cost per 1M output tokens in USD")
	timeout := flag.Duration("timeout", 5*time.Minute, "Request timeout (e.g. 30s, 2m, 10m)")

	flag.Parse()

	if *provider == "" {
		fmt.Fprintln(os.Stderr, "Error: -provider is required")
		flag.Usage()
		os.Exit(1)
	}

	// Resolve prompt
	promptText := *prompt
	if *promptFile != "" {
		data, err := os.ReadFile(*promptFile)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error reading prompt file: %v\n", err)
			os.Exit(1)
		}
		promptText = string(data)
	}
	if promptText == "" {
		fmt.Fprintln(os.Stderr, "Error: -prompt or -prompt-file is required")
		flag.Usage()
		os.Exit(1)
	}

	// Resolve API key from env if not provided
	key := *apiKey
	if key == "" {
		switch *provider {
		case "openai":
			key = os.Getenv("OPENAI_API_KEY")
		case "anthropic":
			key = os.Getenv("ANTHROPIC_API_KEY")
		case "gemini":
			key = os.Getenv("GEMINI_API_KEY")
		}
	}

	// Build options
	var opts []llmhub.Option
	if *model != "" {
		opts = append(opts, llmhub.WithModel(*model))
	}
	if *baseURL != "" {
		opts = append(opts, llmhub.WithBaseURL(*baseURL))
	}
	if *temperature != 0.7 {
		opts = append(opts, llmhub.WithTemperature(*temperature))
	}
	if *maxTokens > 0 {
		opts = append(opts, llmhub.WithMaxTokens(*maxTokens))
	}
	if *inputCost > 0 || *outputCost > 0 {
		opts = append(opts, llmhub.WithCost(*inputCost, *outputCost))
	}
	opts = append(opts, llmhub.WithHTTPClient(&http.Client{Timeout: *timeout}))

	client, err := llmhub.New(*provider, key, opts...)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error creating client: %v\n", err)
		os.Exit(1)
	}

	// Construct message content parts
	var parts []llmhub.ContentPart
	parts = append(parts, llmhub.Text(promptText))

	// Add images if specified
	if *images != "" {
		for _, imgPath := range strings.Split(*images, ",") {
			imgPath = strings.TrimSpace(imgPath)
			if imgPath == "" {
				continue
			}
			imgURL, err := resolveImage(imgPath)
			if err != nil {
				fmt.Fprintf(os.Stderr, "Error processing image %s: %v\n", imgPath, err)
				os.Exit(1)
			}
			parts = append(parts, llmhub.Image(imgURL))
		}
	}

	msgs := []*llmhub.Message{
		llmhub.NewUserMessage(parts...),
	}

	ctx, cancel := context.WithTimeout(context.Background(), *timeout)
	defer cancel()

	fmt.Printf("Provider: %s\n", *provider)
	fmt.Printf("Model: %s\n", *model)
	if *baseURL != "" {
		fmt.Printf("BaseURL: %s\n", *baseURL)
	}
	fmt.Printf("Prompt: %s\n", truncate(promptText, 100))
	fmt.Println("---")

	if *stream {
		chunks, err := client.Stream(ctx, msgs)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error starting stream: %v\n", err)
			os.Exit(1)
		}
		for chunk := range chunks {
			if chunk.Err != nil {
				fmt.Fprintf(os.Stderr, "\nStream error: %v\n", chunk.Err)
				os.Exit(1)
			}
			if chunk.ReasoningDelta != "" {
				fmt.Fprintf(os.Stderr, "[reasoning] %s", chunk.ReasoningDelta)
			}
			fmt.Print(chunk.Delta)
			if chunk.Done {
				break
			}
		}
		fmt.Println()
	} else {
		resp, err := client.Generate(ctx, msgs)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error generating: %v\n", err)
			os.Exit(1)
		}
		fmt.Println(resp.Text())
		if reasoning := resp.ReasoningText(); reasoning != "" {
			fmt.Println("---")
			fmt.Println("Reasoning:")
			fmt.Println(reasoning)
		}
		fmt.Println("---")
		fmt.Printf("Tokens: prompt=%d, completion=%d, total=%d\n",
			resp.Usage.PromptTokens, resp.Usage.CompletionTokens, resp.Usage.TotalTokens)
		if resp.Usage.Cost > 0 {
			fmt.Printf("Cost: $%.6f\n", resp.Usage.Cost)
		}
	}
}

func resolveImage(path string) (string, error) {
	// If it's already a URL or data URI, return as-is
	if strings.HasPrefix(path, "http://") || strings.HasPrefix(path, "https://") || strings.HasPrefix(path, "data:") {
		return path, nil
	}
	// Read file and convert to data URI
	data, err := os.ReadFile(path)
	if err != nil {
		return "", err
	}
	mimeType := http.DetectContentType(data)
	encoded := base64.StdEncoding.EncodeToString(data)
	return fmt.Sprintf("data:%s;base64,%s", mimeType, encoded), nil
}

func truncate(s string, maxLen int) string {
	s = strings.ReplaceAll(s, "\n", " ")
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}

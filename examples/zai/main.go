// zai demonstrates using llmhub's Z.AI provider with the OpenAI-compatible
// Chat Completions API. Coding Plan is the default; pass -base-url for a
// general API-balance account.
package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"time"

	"github.com/smhanov/llmhub"
	"github.com/smhanov/llmhub/providers/zai"
)

func main() {
	model := flag.String("model", zai.DefaultModel, "Z.AI model identifier")
	prompt := flag.String("prompt", "Reply with exactly LLHUB_ZAI_OK", "Prompt to send")
	stream := flag.Bool("stream", false, "Print the response as it streams")
	baseURL := flag.String("base-url", zai.CodingPlanBaseURL, "Z.AI API base URL (use "+zai.GeneralBaseURL+" for general API billing)")
	timeout := flag.Duration("timeout", 2*time.Minute, "Request timeout")
	flag.Parse()

	apiKey := os.Getenv("ZAI_API_KEY")
	if apiKey == "" {
		fmt.Fprintln(os.Stderr, "zai: set ZAI_API_KEY; credentials are intentionally not accepted as flags")
		os.Exit(2)
	}

	client, err := llmhub.New("zai", apiKey,
		llmhub.WithBaseURL(*baseURL),
		llmhub.WithModel(*model),
	)
	if err != nil {
		exitError(err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), *timeout)
	defer cancel()
	messages := []*llmhub.Message{llmhub.NewUserMessage(llmhub.Text(*prompt))}
	if *stream {
		chunks, err := client.Stream(ctx, messages)
		if err != nil {
			exitError(err)
		}
		for chunk := range chunks {
			if chunk.Err != nil {
				exitError(chunk.Err)
			}
			fmt.Print(chunk.Delta)
		}
		fmt.Println()
		return
	}

	resp, err := client.Generate(ctx, messages)
	if err != nil {
		exitError(err)
	}
	fmt.Println(resp.Text())
	fmt.Printf("tokens: input=%d output=%d total=%d\n", resp.Usage.PromptTokens, resp.Usage.CompletionTokens, resp.Usage.TotalTokens)
}

func exitError(err error) {
	fmt.Fprintf(os.Stderr, "zai: %v\n", err)
	os.Exit(1)
}

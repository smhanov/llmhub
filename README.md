# llmhub

Unified, provider-agnostic Go client for modern Large Language Models (LLMs). llmhub wraps multiple vendors (OpenAI, Anthropic, Gemini, Ollama, and your own) behind a single, expressive API that understands multi-modal messages, streaming, and provider registries.

## Why llmhub?

- **One API, many vendors** – swap providers without rewriting your business logic.
- **Multi-modal ready** – mix text and images in both requests and responses.
- **Streaming made simple** – consume deltas through idiomatic Go channels.
- **Extensible registry** – register first-party or external providers at runtime.
- **Functional options** – configure models, endpoints, and credentials cleanly.

## Installation

```bash
go get github.com/smhanov/llmhub
```

## Quick Start

```go
package main

import (
    "context"
    "fmt"

    "github.com/smhanov/llmhub"
    _ "github.com/smhanov/llmhub/providers/openai"
)

func main() {
    client, err := llmhub.New("openai", "sk-YOUR-KEY", llmhub.WithModel("gpt-4o-mini"))
    if err != nil {
        panic(err)
    }

    prompt := []*llmhub.Message{
        llmhub.NewSystemMessage(llmhub.Text("You are a witty assistant.")),
        llmhub.NewUserMessage(llmhub.Text("Explain quantum mechanics in five words.")),
    }

    resp, err := client.Generate(context.Background(), prompt)
    if err != nil {
        panic(err)
    }

    fmt.Println(resp.Text())
}
```

## Streaming Responses

```go
stream, err := client.Stream(ctx, prompt)
if err != nil {
    log.Fatal(err)
}
for chunk := range stream {
    if chunk.Err != nil {
        log.Printf("stream error: %v", chunk.Err)
        break
    }
    fmt.Print(chunk.Delta)
    if chunk.Done {
        break
    }
}
```

## Vision & Multi-modal Inputs

```go
prompt := []*llmhub.Message{
    llmhub.NewUserMessage(
        llmhub.Text("What is shown here?"),
        llmhub.Image("https://example.com/diagram.png"),
    ),
}
resp, _ := client.Generate(ctx, prompt)
for _, part := range resp.Content {
    if text, ok := part.(*llmhub.TextContent); ok {
        fmt.Println(text.Text)
    }
}
```

## Provider Registry

Add your own provider in a different module:

```go
func init() {
    llmhub.MustRegisterProvider("my-llm", func(apiKey string, opts ...llmhub.Option) (llmhub.Provider, error) {
        return newMyClient(apiKey, opts...) // implement llmhub.Provider
    })
}
```

At runtime, consumers simply call `llmhub.New("my-llm", "token")`.

## Built-in Providers

| Provider  | Status        | Notes                                                    |
| --------- | ------------- | -------------------------------------------------------- |
| OpenAI    | ✅ Production | Chat Completions, multi-modal prompts, SSE streaming.    |
| Anthropic | ✅ Production | Claude 3 Messages API with streaming deltas.             |
| Gemini    | ✅ Production | Gemini 1.5 multi-modal text+vision APIs, streaming JSON. |
| Ollama    | ✅ Production | Local inference via `/api/chat`, streaming friendly.     |

### OpenAI Provider Details

**Automatic `/v1` suffix:** When a custom base URL is provided (via `WithBaseURL`), the
OpenAI provider ensures the URL ends with `/v1`. If it doesn't, `/v1` is appended
automatically. This means both `https://api.openai.com` and
`https://api.openai.com/v1` are accepted and behave identically.

**`"default"` model:** When the model is set to `"default"` (case-insensitive), the
provider queries the `/v1/models` endpoint at initialization and automatically
selects the first available model. This is especially useful for self-hosted
OpenAI-compatible servers (e.g. Ollama, vLLM, LocalAI) where you may not know
the model name in advance:

```go
client, err := llmhub.New("openai", "key",
    llmhub.WithBaseURL("http://localhost:11434"),
    llmhub.WithModel("default"),
)
// The provider will query http://localhost:11434/v1/models and use the first model.
```

To reduce binary size, providers self-register when imported, enabling tree-shaking when unused.

```go
import (
    _ "github.com/smhanov/llmhub/providers/openai"
    _ "github.com/smhanov/llmhub/providers/anthropic"
    _ "github.com/smhanov/llmhub/providers/gemini"
    _ "github.com/smhanov/llmhub/providers/ollama"
)
```

Each provider reads the shared functional options:

- `WithAPIKey` – supply SaaS credentials (`openai`, `anthropic`, `gemini`).
- `WithBaseURL` – point to proxies/self-hosted gateways.
- `WithModel`, `WithTemperature`, `WithMaxTokens` – customize LLM behavior per call.
- `WithWebSearch` – enable web search/grounding (Gemini: `google_search` tool).
- `WithCost` – set per-million-token pricing for cost accounting.

## Cost Accounting

llmhub can track the estimated cost of each request based on token usage and
configured per-million-token rates. Costs are expressed in US dollars per 1
million tokens, matching standard LLM provider pricing.

```go
client, _ := llmhub.New("openai", apiKey,
    llmhub.WithModel("gpt-4o"),
    llmhub.WithCost(2.50, 10.00), // $2.50 per 1M input, $10.00 per 1M output tokens
)

resp, _ := client.Generate(ctx, prompt)
fmt.Printf("Tokens: %d in, %d out\n",
    resp.Usage.PromptTokens, resp.Usage.CompletionTokens)
fmt.Printf("Cost: $%.6f\n", resp.Usage.Cost)
```

Cost is computed automatically after each `Generate` call:

$$
\text{Cost} = \frac{\text{PromptTokens} \times \text{InputRate}}{1{,}000{,}000}
+ \frac{\text{CompletionTokens} \times \text{OutputRate}}{1{,}000{,}000}
$$

If no cost rates are configured, `Usage.Cost` will be zero.

You can also override cost rates on a per-request basis:

```go
// Use cheaper rates for a specific call
resp, _ := client.Generate(ctx, prompt,
    llmhub.WithCost(0.15, 0.60),
)
```

## Web Search / Grounding

Some providers support web search to ground responses in real-time information:

```go
client, _ := llmhub.New("gemini", apiKey,
    llmhub.WithModel("gemini-2.5-flash"),
    llmhub.WithWebSearch(true),
)

prompt := []*llmhub.Message{
    llmhub.NewUserMessage(llmhub.Text("What are the latest news about Apple Inc?")),
}

resp, _ := client.Generate(ctx, prompt)
fmt.Println(resp.Text())
```

| Provider  | Web Search Support           |
| --------- | ---------------------------- |
| Gemini    | ✅ Uses `google_search` tool |
| OpenAI    | ❌ Not supported             |
| Anthropic | ❌ Not supported             |
| Ollama    | ❌ Not supported             |

Need multi-provider routing? Instantiate one `llmhub.Client` per provider and switch at runtime:

```go
openaiClient := llmhub.MustNew("openai", os.Getenv("OPENAI_API_KEY"), llmhub.WithModel("gpt-4o"))
claudeClient := llmhub.MustNew("anthropic", os.Getenv("ANTHROPIC_API_KEY"), llmhub.WithModel("claude-3-opus-20240229"))

func answer(ctx context.Context, prompt []*llmhub.Message, vendor string) (*llmhub.Response, error) {
    switch vendor {
    case "anthropic":
        return claudeClient.Generate(ctx, prompt)
    default:
        return openaiClient.Generate(ctx, prompt)
    }
}
```

## Testing

```bash
go test ./...
```

## CLI Test Tool

A command-line tool is included for end-to-end testing of providers. Build and run it from the repository root:

```bash
go run ./examples/cli [options]
```

### Options

| Flag           | Description                                                                       |
| -------------- | --------------------------------------------------------------------------------- |
| `-provider`    | Provider name: `openai`, `anthropic`, `gemini`, `ollama` (required)               |
| `-model`       | Model identifier (e.g., `gpt-4o`, `claude-3-haiku-20240307`, `gemini-2.5-flash`)  |
| `-api-key`     | API key (or use env vars `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`) |
| `-base-url`    | Override provider base URL (useful for Ollama or proxies)                         |
| `-prompt`      | Text prompt to send                                                               |
| `-prompt-file` | File containing the prompt text                                                   |
| `-images`      | Comma-separated list of image file paths or URLs                                  |
| `-stream`      | Enable streaming mode                                                             |
| `-temperature` | Sampling temperature (default: 0.7)                                               |
| `-max-tokens`  | Maximum tokens to generate                                                        |
| `-input-cost`  | Cost per 1M input tokens in USD (for cost accounting)                             |
| `-output-cost` | Cost per 1M output tokens in USD (for cost accounting)                            |

### Examples

**Text generation with Ollama (self-hosted):**

```bash
go run ./examples/cli \
  -provider ollama \
  -model qwen3:32b \
  -base-url https://ollama.example.com \
  -prompt "Why is the sky blue?"
```

**Text generation with Gemini:**

```bash
go run ./examples/cli \
  -provider gemini \
  -model gemini-2.5-flash \
  -api-key YOUR_GEMINI_KEY \
  -prompt "Explain quantum entanglement simply."
```

**Vision/image input with Gemini:**

```bash
go run ./examples/cli \
  -provider gemini \
  -model gemini-2.5-flash \
  -api-key YOUR_GEMINI_KEY \
  -prompt "Describe this image in detail." \
  -images cat.jpg
```

**Streaming mode with OpenAI:**

```bash
go run ./examples/cli \
  -provider openai \
  -model gpt-4o \
  -api-key YOUR_OPENAI_KEY \
  -prompt "Write a haiku about coding." \
  -stream
```

**Using environment variables:**

```bash
export OPENAI_API_KEY=sk-...
go run ./examples/cli -provider openai -model gpt-4o -prompt "Hello!"
```

**With cost accounting:**

```bash
go run ./examples/cli \
  -provider openai \
  -model gpt-4o \
  -input-cost 2.50 \
  -output-cost 10.00 \
  -prompt "Explain Go interfaces."
```

## Contributing

Issues and PRs are welcome! Start by filing an issue describing the provider or feature you would like to add, then open a PR with tests and documentation. Check the existing provider stubs (Anthropic, Gemini, Ollama) for extension points.

## License

MIT License © 2026 llmhub contributors

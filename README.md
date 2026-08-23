# llmhub

Unified, provider-agnostic Go client for modern Large Language Models (LLMs). llmhub wraps multiple vendors (OpenAI, Anthropic, Gemini, Ollama, and your own) behind a single, expressive API that understands multi-modal messages, streaming, and provider registries.

## Why llmhub?

- **One API, many vendors** – swap providers without rewriting your business logic.
- **Multi-modal ready** – mix text and images in both requests and responses.
- **Tool calling** – declare provider-agnostic tools and handle normalized tool calls.
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
    if chunk.ReasoningDelta != "" {
        log.Printf("reasoning delta: %s", chunk.ReasoningDelta)
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

## Reasoning / Thinking Blocks

Some models expose reasoning as separate blocks in the response payload. llmhub preserves those blocks in `Response.Content` as `*llmhub.ReasoningContent`.

```go
resp, _ := client.Generate(ctx, prompt)

fmt.Println("final answer:", resp.Text())
fmt.Println("reasoning:", resp.ReasoningText())

for _, part := range resp.Content {
    if r, ok := part.(*llmhub.ReasoningContent); ok {
        fmt.Println("reasoning block:", r.Text)
    }
}
```

For streaming, reasoning is exposed separately on each chunk via `StreamChunk.ReasoningDelta`.

## Tool Calling

Declare tools with `WithTools`, read requested calls from `Response.ToolCalls()`, execute them in your code, then send the result back with `NewToolResultMessage`.

```go
weather := llmhub.NewTool("weather", "Get current weather", map[string]interface{}{
    "type": "object",
    "properties": map[string]interface{}{
        "city": map[string]interface{}{"type": "string"},
    },
    "required": []string{"city"},
})

client, _ := llmhub.New("openai", apiKey,
    llmhub.WithModel("gpt-4o-mini"),
    llmhub.WithTools(weather),
    llmhub.WithToolChoice(llmhub.AutoToolChoice()),
)

messages := []*llmhub.Message{
    llmhub.NewUserMessage(llmhub.Text("What is the weather in Toronto?")),
}

resp, _ := client.Generate(ctx, messages)
for _, call := range resp.ToolCalls() {
    result := runTool(call.Name, call.Arguments)
    messages = append(messages,
        llmhub.NewAssistantMessage(llmhub.ToolCall(call.ID, call.Name, call.Arguments)),
        llmhub.NewToolResultMessage(call.ID, call.Name, llmhub.Text(result)),
    )
}

finalResp, _ := client.Generate(ctx, messages)
fmt.Println(finalResp.Text())
```

Tool choice helpers include `AutoToolChoice`, `NoToolChoice`, `RequiredToolChoice`, and `NamedToolChoice("tool_name")` for providers that expose tool-choice controls. Streaming tool calls are exposed on `StreamChunk.ToolCalls`.

| Provider  | Tool Calling Support |
| --------- | -------------------- |
| OpenAI    | ✅ Chat Completions tools |
| Anthropic | ✅ Messages API tools |
| Gemini    | ✅ Function declarations |
| Ollama    | ✅ Native `/api/chat` tools |
| xAI       | ✅ Chat Completions tools |

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
| xAI       | ✅ Production | Grok models, API key or OAuth device flow, SSE streaming.|

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

### xAI (Grok) Provider Details

Import `providers/xai` to register the `xai` provider. It uses the
OpenAI-compatible Chat Completions API, supports streaming and tool calling,
and defaults to `grok-4.6`. Override the model when your account has access to
a different Grok model.

For xAI API keys, use it exactly like any other API-key provider—OAuth setup is
not involved:

```go
import (
    "github.com/smhanov/llmhub"
    _ "github.com/smhanov/llmhub/providers/xai"
)

// Using an API key:
client, err := llmhub.New("xai", apiKey,
    llmhub.WithModel("grok-4.6"),
)

if err != nil {
    panic(err)
}
```

To reduce binary size, providers self-register when imported, enabling tree-shaking when unused.

```go
import (
    _ "github.com/smhanov/llmhub/providers/openai"
    _ "github.com/smhanov/llmhub/providers/anthropic"
    _ "github.com/smhanov/llmhub/providers/gemini"
    _ "github.com/smhanov/llmhub/providers/ollama"
    _ "github.com/smhanov/llmhub/providers/xai"
)
```

Each provider reads the shared functional options:

- `WithAPIKey` – supply SaaS credentials (`openai`, `anthropic`, `gemini`, `xai`).
- `WithTokenSource` – supply an OAuth / dynamic token source (`xai`).
- `WithBaseURL` – point to proxies/self-hosted gateways.
- `WithModel`, `WithTemperature` – customize LLM behavior per call. Often it is best to omit and go with the defaults.
- `WithMaxTokens` – only set this when you truly need a hard output cap; otherwise leave it unset to reduce the risk of truncated responses.
- `WithWebSearch` – enable web search/grounding (Gemini: `google_search` tool).
- `WithTools`, `WithToolChoice` – enable user-defined tool calling.
- `WithResponseModalities` – control output modalities (e.g. `"IMAGE"` for Gemini image generation).
- `WithCost` – set per-million-token pricing for cost accounting.

> [!WARNING]
> Prefer not to use `WithMaxTokens` in normal application code. Provider defaults usually produce more complete answers, while an explicit cap that is too low commonly causes cut-off output.

## Image Generation / Output Modalities

Gemini image-generation models (e.g. `gemini-2.5-flash-image`) can return images
instead of—or alongside—text. Use `WithResponseModalities` to tell the model
which output types you want:

```go
import (
    "github.com/smhanov/llmhub"
    _ "github.com/smhanov/llmhub/providers/gemini"
)

client, _ := llmhub.New("gemini", apiKey,
    llmhub.WithModel("gemini-2.5-flash-image"),
    llmhub.WithResponseModalities("IMAGE"),
)

prompt := []*llmhub.Message{
    llmhub.NewUserMessage(
        llmhub.Text("Upscale this image to 800 pixels wide."),
        llmhub.Image("data:image/jpeg;base64,/9j/4AAQ..."),
    ),
}

resp, _ := client.Generate(ctx, prompt)

for _, part := range resp.Content {
    if img, ok := part.(*llmhub.ImageContent); ok {
        // img.URL is a data URL: "data:image/png;base64,..."
        fmt.Println("Got image:", len(img.URL), "bytes")
    }
}
```

Pass `"TEXT"` and `"IMAGE"` together to allow mixed text+image output:

```go
llmhub.WithResponseModalities("TEXT", "IMAGE")
```

| Provider  | Image Output Support                          |
| --------- | --------------------------------------------- |
| Gemini    | ✅ Via `WithResponseModalities("IMAGE")`      |
| OpenAI    | ❌ Use the Images API directly                |
| Anthropic | ❌ Not supported                              |
| Ollama    | ❌ Not supported                              |

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
| xAI       | ❌ Not supported             |

## OAuth Token Sources

OAuth is opt-in and provider-dependent. It is not required by existing
API-key providers. Currently, `xai` is the built-in provider that accepts
`WithTokenSource`; future OAuth providers can reuse the generic `auth` and
`auth/oauth2` packages.

OAuth login is always explicit and application-controlled: `llmhub.New`,
`Generate`, and `Stream` will never open a browser or wait for a user to sign
in.

### Using an Existing Token File

```go
import (
    "github.com/smhanov/llmhub"
    "github.com/smhanov/llmhub/providers/xai"
)

// Discover an existing ~/.grok/auth.json, ~/.xgroxy/auth.json, or XAI_AUTH_FILE.
authPath := xai.DefaultAuthPath("")
store := xai.NewFileTokenStore(authPath)
source := xai.NewTokenSource(store)

client, err := llmhub.New("xai", "",
    llmhub.WithTokenSource(source),
    llmhub.WithModel("grok-4.6"),
)
if err != nil {
    panic(err)
}
```

`WithTokenSource` takes precedence over the positional `apiKey` argument and
`WithAPIKey`. If neither one is supplied to `xai`, construction fails with
`llmhub.ErrInvalidInput`.

### Performing an Interactive Device Login

Login is an explicit, application-controlled operation that never blocks inside library constructors:

```go
import (
    "context"
    "fmt"

    "github.com/smhanov/llmhub/providers/xai"
)

func login(ctx context.Context) error {
    store := xai.NewFileTokenStore(xai.DefaultAuthPath(""))
    flow := xai.NewDeviceFlow()

    authz, err := flow.Start(ctx)
    if err != nil {
        return err
    }

    fmt.Printf("Please visit: %s\n", authz.VerificationURI)
    if authz.VerificationURIComplete != "" {
        fmt.Printf("Or open: %s\n", authz.VerificationURIComplete)
    }
    fmt.Printf("And enter code: %s\n", authz.UserCode)

    token, err := flow.Wait(ctx, authz)
    if err != nil {
        return err
    }

    return store.Save(ctx, token)
}
```

### Credential Precedence & Error Handling

- **Precedence:** `WithTokenSource` takes precedence over `WithAPIKey` and the positional `apiKey` argument.
- **Refresh:** `xai.NewTokenSource` refreshes access tokens before they expire and performs one refresh/retry after an API 401 response. It safely handles rotating refresh tokens within one process.
- **Reauthentication:** If a refresh token is expired or revoked, provider calls return an error wrapping `auth.ErrReauthenticationRequired`. Use `errors.Is` to prompt for a new interactive login:

  ```go
  if errors.Is(err, auth.ErrReauthenticationRequired) {
      // Start the device flow again and save the new token.
  }
  ```

- **Security:** New token directories/files are created with private `0700`/`0600` permissions on Unix. Do not print, commit, or pass access/refresh tokens as command-line arguments.

## xAI OAuth acceptance example

`examples/xai-oauth` is a real end-to-end example. It performs an explicit
device login when needed, can force a token refresh, then verifies both a
non-streaming and a streaming Grok response. It requires an authorized xAI
account and intentionally is not part of the offline test suite.

```bash
go run ./examples/xai-oauth \
  -auth-file /private/path/auth.json \
  -force-login \
  -verify-refresh \
  -model grok-4.6
```

The command prints a verification URL and one-time code, waits for you to
authorize it in a browser, and succeeds only after it prints:

```text
refresh: OK
generate: OK
stream: OK
xai oauth acceptance: PASS
```

Use a dedicated private auth-file path for this check. The example never
prints access or refresh tokens.

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

| Flag           | Description                                                                                 |
| -------------- | ------------------------------------------------------------------------------------------- |
| `-provider`    | Provider name: `openai`, `anthropic`, `gemini`, `ollama`, `xai` (required)                 |
| `-model`       | Model identifier (e.g., `gpt-4o`, `claude-3-haiku-20240307`, `gemini-2.5-flash`, `grok-4.6`)|
| `-api-key`     | API key (or use env vars `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `XAI_API_KEY`)|
| `-auth-file`   | Path to token file for OAuth providers (e.g. xAI)                                           |
| `-base-url`    | Override provider base URL (useful for Ollama or proxies)                                   |
| `-prompt`      | Text prompt to send                                                                         |
| `-prompt-file` | File containing the prompt text                                                             |
| `-images`      | Comma-separated list of image file paths or URLs                                            |
| `-stream`      | Enable streaming mode                                                                       |
| `-temperature` | Sampling temperature (default: 0.7)                                                         |
| `-max-tokens`  | Hard cap on generated tokens; leave unset unless needed to avoid truncation                 |
| `-input-cost`  | Cost per 1M input tokens in USD (for cost accounting)                                       |
| `-output-cost` | Cost per 1M output tokens in USD (for cost accounting)                                      |
| `-timeout`     | Request timeout duration (e.g. `30s`, `2m`, `10m`)                                           |

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

**Using an existing xAI OAuth token file:**

```bash
go run ./examples/cli \
  -provider xai \
  -auth-file ~/.grok/auth.json \
  -model grok-4.6 \
  -prompt "Reply with exactly: OK"
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

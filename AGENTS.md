# Agent notes

## Example CLI

Live provider checks go through `examples/cli`:

```bash
go run ./examples/cli -provider anthropic -model MODEL -prompt "..."
```

Useful flags:

- `-stream` — streaming mode. Prints `[id]`, `[tool_call]`, `[usage]`, and `[finish_reason]` on stderr.
- `-tools` — demo `get_weather` tool with required tool choice. Use this to exercise tool-calling / `finish_reason=tool_calls`.
- `-header 'Key:Value'` — extra request headers (comma-separated for more than one).
- `-api-key` — optional; otherwise the CLI reads `ANTHROPIC_API_KEY` for Anthropic.

Never put API keys in this file, in flags committed to the repo, or in command examples.

## Accessing Claude (Anthropic)

1. Export `ANTHROPIC_API_KEY` in the shell. Do not pass the key on the command line if it would land in shell history you cannot control; the env var is enough.
2. Identity-linked Anthropic keys require `anthropic-workspace-id`. Without it, Messages returns:

   `anthropic-workspace-id is required when authenticating with an identity-linked API key`

3. List workspaces (no inference):

   ```bash
   curl -sS \
     -H "x-api-key: $ANTHROPIC_API_KEY" \
     -H "anthropic-version: 2023-06-01" \
     https://api.anthropic.com/v1/organizations/workspaces
   ```

   Use the workspace `id` (`wrkspc_...`), not `compartment_id` and not the organization id.

4. If Messages 404s with `Workspace \`<uuid>\` not found`, that listed workspace's inference compartment is missing. Create a dedicated workspace and use its `id`:

   ```bash
   curl -sS \
     -H "x-api-key: $ANTHROPIC_API_KEY" \
     -H "anthropic-version: 2023-06-01" \
     -H "content-type: application/json" \
     https://api.anthropic.com/v1/organizations/workspaces \
     -d '{"name":"llmhub-test"}'
   ```

5. List models available to that workspace, then pick an id from the response (do not assume the provider default `claude-3-haiku-20240307` exists):

   ```bash
   curl -sS \
     -H "x-api-key: $ANTHROPIC_API_KEY" \
     -H "anthropic-version: 2023-06-01" \
     -H "anthropic-workspace-id: wrkspc_..." \
     https://api.anthropic.com/v1/models
   ```

6. Stream a normal completion:

   ```bash
   go run ./examples/cli \
     -provider anthropic \
     -model MODEL \
     -stream \
     -header "anthropic-workspace-id:wrkspc_..." \
     -prompt "Reply with exactly the word ping."
   ```

   Expect `[finish_reason] stop` on stderr.

7. Stream a forced tool call:

   ```bash
   go run ./examples/cli \
     -provider anthropic \
     -model MODEL \
     -stream \
     -tools \
     -header "anthropic-workspace-id:wrkspc_..." \
     -prompt "What's the weather in Toronto right now? Use the get_weather tool."
   ```

   Expect a `[tool_call]` for `get_weather` and `[finish_reason] tool_calls`.

Wire note: Anthropic `message_delta` carries `stop_reason` under `delta.stop_reason`, not at the event root. The Anthropic provider maps `end_turn`/`stop_sequence` → `stop`, `max_tokens` → `length`, `tool_use` → `tool_calls`.

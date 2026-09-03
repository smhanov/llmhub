# Z.AI example

Set the key in the environment so it is not written to source or shell arguments:

```sh
export ZAI_API_KEY='your-key'
go run ./examples/zai -model glm-5.3 -prompt 'Reply with exactly OK'
go run ./examples/zai -model glm-5.3 -stream -prompt 'Reply with exactly OK'
```

The default is Z.AI's Coding Plan endpoint. For a general API-balance key, use:

```sh
go run ./examples/zai -base-url https://api.z.ai/api/paas/v4 -model glm-5.3 -prompt 'Hello'
```

This example uses the dedicated `zai` provider, which keeps Z.AI's versioned
base URL intact instead of appending `/v1`.

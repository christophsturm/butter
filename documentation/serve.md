# Serve

`butter serve` is an OpenAI-compatible HTTP server. One process holds **one** loaded checkpoint. Completions clients talk to it over HTTP/1.1. In-process `generate()` is unchanged.

On Apple Silicon the Swift CLI loads HuggingFace / MLX checkpoints. On other platforms the Rust CLI loads a GGUF. The HTTP contract is the same.

## Start

Apple (Swift):

```bash
butter serve --model mlx-community/Qwen3.5-0.8B-MLX-4bit
```

Elsewhere (Rust, GGUF path required):

```bash
butter serve --model /path/to/model.gguf
```

Default bind is `127.0.0.1:8000`. Override with `--host` and `--port`.

```bash
butter serve --model <id-or-path> --host 0.0.0.0 --port 8000
```

The process prints the routes when it is listening:

```
GET  /health
GET  /v1/models
POST /v1/chat/completions
POST /v1/messages
```

`POST /v1/chat/completions` is the product path. Point a Completions client at `http://127.0.0.1:8000/v1`.

## Discovery

`GET /v1/models` lists the **resident** checkpoint: `id` plus `max_model_len` / `context_length` when known.

`GET /health` returns `ok`.

## Completions

`POST /v1/chat/completions` accepts a JSON body with `messages`. Optional fields the server honors:

| Field | Notes |
|---|---|
| `model` | Resident id, or a different resolvable id to swap (see below). |
| `stream` | SSE chunks when `true`. |
| `stream_options.include_usage` | Extra usage-only chunk after `finish_reason`. |
| `max_tokens` / `max_completion_tokens` | Generation cap. |
| `temperature`, `top_p`, `seed` | Sampling. |
| `tools` | Function tools; leaked markup is recovered as structured `tool_calls`. |
| `chat_template_kwargs.enable_thinking` | Thinking on/off when the template supports it. |

Authorization headers are ignored (no auth). Unknown JSON fields are ignored.

### Streaming

SSE `data:` lines share one completion `id`. Content deltas come first. A recovered tool call is two `tool_calls` deltas: the first carries `id` / `name`, later deltas are arguments only. Loop on presence of `delta.tool_calls`, not only `finish_reason`.

The finish chunk always has a string `finish_reason` (`stop`, `tool_calls`, or `length`). When `include_usage` is set, a usage-only chunk (empty `choices`) follows with `prompt_tokens`, `completion_tokens`, `total_tokens`, and `cached_tokens`. Then `data: [DONE]`.

Closing the TCP connection mid-prefill or mid-decode aborts GPU work. Leftover prefill chunks do not run.

### Prefix reuse

`usage.cached_tokens` is the matched prefix length (`nPast`) of the last prompt against the live KV. Overlapping prompts and tool rounds that resend the full `messages` (stable system + tools stem) reuse that prefix. It is not a planted constant.

A tool round that forks the conversation rebuilds the cache. Recurrent hybrid layers cannot rewind.

### Tools

If the model emits `<tool_call>…</tool_call>` JSON or XML markup, the server strips think blocks, recovers structured `tool_calls`, and does not leave raw markup in `content`. Non-stream responses put `tool_calls` on the assistant message with `finish_reason: "tool_calls"`.

## One resident model

The process keeps a single loaded checkpoint and its session/KV.

- Completions `model` equal to the resident id: no reload, prefix stays.
- A different resolvable id: abort in-flight generate, drop the old session/KV, load the new checkpoint, then serve. After a successful swap the machine holds one model, not two.
- Unknown or unloadable id: `404` (or `400`). The previous resident stays on `GET /v1/models`. No hang, no half-loaded process.

KV is not written to disk. After a swap the client resends full `messages`; the new model prefills from scratch.

Boot loads `--model` before accepting traffic. A Completions call can still swap the resident later.

## Flags

| Flag | Meaning |
|---|---|
| `--model` | HuggingFace id or local path (Swift). GGUF path (Rust). Required. |
| `--host` | Bind address. Default `127.0.0.1`. |
| `--port` | Bind port. Default `8000`. |
| `--enable-thinking` | Default `chat_template_kwargs.enable_thinking` when the client omits it. |
| `--reasoning-effort` | `low` / `medium` / `high` when the client omits it. |
| `--max-context` | Size the KV for this many positions. |
| `--kv-cache` | `raw`, `affine8`, `affine4`, `aura`, or an `auraNvM` recipe. Affine/AURA are Apple-only; Rust logs and uses raw. |
| `--aura-decode-path` | `compressed` (default) or `dequant-mirror`. Ignored unless `--kv-cache` is AURA. |
| `--kv-window-size` | FIFO KV window. Unset = unbounded up to `--max-context`. |
| `--kv-window-keep` | Attention-sink slots. Needs `--kv-window-size`. |

## Library (Apple)

HTTP is optional. An Apple app can keep a `Model` and call `generate()` / `generateStream()` in-process, or wrap a loaded model in `ButterServeBackend` + `ServeServer` if it wants the same HTTP surface.

```swift
let model = try await Model.load("mlx-community/Qwen3.5-0.8B-MLX-4bit")
let backend = ButterServeBackend(model: model, modelId: "my-model")
let server = ServeServer(host: "127.0.0.1", port: 8000, backend: backend)
try await server.runForever()
```

`ResidentServeBackend` wraps that for Completions-driven swap with a loader closure.

## Not this server

No `/v1/responses`. No multi-checkpoint LRU. No disk session snapshot. One-shot `generate()` stays the library entrypoint for a single prompt.

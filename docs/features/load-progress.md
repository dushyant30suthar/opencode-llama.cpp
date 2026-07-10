# Load & prefill progress

Two formerly-invisible waits, made visible:

## Model load into VRAM

Loading a 30B-class model takes a while. While a request is in flight on a
local model, the TUI subscribes to the router's `GET /models/sse` stream and
shows llama-server's real-time load progress next to the busy spinner — stage
and percent when available, e.g.:

```
loading Qwen3.6-27B-GGUF text_model 42%
```

The subscription only exists while a request is pending and degrades silently
if the router is down or predates the endpoint.

## Prompt prefill

Long prefills (context compaction, huge pastes) used to show an indefinite
spinner. The status row now polls the router's `GET /slots` endpoint while a
request is in flight and renders live prefill progress from
`n_prompt_tokens_processed / n_prompt_tokens`:

```
reading prompt 47032/63071 tok · 74%
```

Code: `packages/tui/src/component/llamastack-status.tsx`.

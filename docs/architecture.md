# Architecture

Two moving parts, one contract: opencode is the agent frontend, `llama-server`
(from llama.cpp) is the engine, and everything between them is an
OpenAI-compatible HTTP API plus one INI file.

```
┌─────────────────────────────┐
│ opencode (fork)             │
│  TUI ── /config screen      │
│  provider/llamastack.ts     │
└──────────┬──────────────────┘
           │ http://127.0.0.1:9337
           ▼
┌─────────────────────────────┐     ┌──────────────────────────────┐
│ llama-server (router mode)  │────▶│ ~/.lmstudio/models/          │
│  --models-max 1             │     │   <publisher>/<repo>/*.gguf  │
│  --models-preset models.ini │     └──────────────────────────────┘
└──────────┬──────────────────┘
           ▼
┌─────────────────────────────┐
│ ~/.local/state/llamastack/  │
│  models.ini  server.json    │
│  router.log  router.pid     │
└─────────────────────────────┘
```

## Startup detection (zero config)

On launch the provider probes, in order:

1. An already-running OpenAI-compatible server on port `9337`.
2. If nothing answers and a `llama-server` binary can be found
   (env override → `$PATH` → conventional build dirs), it **spawns the router
   itself** — detached, so it survives opencode exiting — and polls
   `/v1/models` for up to 10 s.

Router mode with `--models-max 1` means selecting a different model **swaps**
it in VRAM instead of stacking servers. Detection is memoized per process and
never fails opencode's startup.

## models.ini is the single source of truth

`~/.local/state/llamastack/models.ini` is a llama-server preset INI, one
section per model. The contract:

- **opencode appends** a default section for newly discovered models and
  **never modifies or removes** existing sections — the file is user-owned.
- **Sampling lives here** (`temp`, `top-p`, `top-k`, `min-p`). The fork
  deliberately sends *no* sampler overrides for llamastack models, so INI
  values (and the `/config` temperature field) actually take effect. Stock
  opencode's per-model-family sampler heuristics are bypassed for this
  provider.
- Any llama-server flag works as a key (without leading dashes) — new engine
  features cost zero frontend code.

Why an INI and not opencode config: the engine reads it natively
(`--models-preset`), it survives fork rebuilds, and hand edits are
first-class.

## Router API surface used by the fork

| Endpoint | Used for |
| --- | --- |
| `/v1/chat/completions` etc. | The actual inference (OpenAI-compatible) |
| `GET /models` | Model list + loaded-model parameters (ctx, ngl, split, kv) |
| `GET /models?reload=1` | Re-read presets after a `/config` save |
| `POST /models/load` | VRAM warm-up the moment a model is selected |
| `GET /models/sse` | Live load-progress stream while a request is pending |
| `GET /slots` | Prefill progress (`reading prompt N/TOTAL tok · P%`) |

Everything degrades gracefully when the router predates an endpoint — the
fork works against a stock llama-server too, just with fewer live displays.

## Files & ports

| What | Where |
| --- | --- |
| Per-model settings | `~/.local/state/llamastack/models.ini` (user-owned) |
| Global server settings | `~/.local/state/llamastack/server.json` (`{"expose": true}`) |
| Router log / pidfile | `~/.local/state/llamastack/router.{log,pid}` |
| Router port | `9337` |
| Models directory | `$LLAMASTACK_MODELS_DIR`, else `~/.lmstudio/models` |
| llama-server binary | `$LLAMASTACK_SERVER_BIN`, else `$PATH`, else conventional build dirs |
| Escape hatch | `OPENCODE_DISABLE_LLAMASTACK=1` |

## Where the code lives (opencode submodule, branch `opencode-llama.cpp`)

| Area | Path |
| --- | --- |
| Provider: detection, router spawn, preset generation | `packages/opencode/src/provider/llamastack.ts` |
| Sampler ownership (no overrides for this provider) | `packages/opencode/src/provider/transform.ts` |
| TUI helpers: INI parse/edit, router control | `packages/tui/src/util/llamastack.ts` |
| `/config` settings screen | `packages/tui/src/component/dialog-llamastack-config.tsx` |
| Sidebar server panel, status line | `packages/tui/src/feature-plugins/sidebar/llamastack.tsx`, `packages/tui/src/component/llamastack-status.tsx` |
| Self-update guard | `packages/opencode/src/installation/index.ts`, `cli/upgrade.ts` |

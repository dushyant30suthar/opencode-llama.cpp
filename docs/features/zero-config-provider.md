# Zero-config llamastack provider

Launch `opencode` and local models are in the model picker. No config files,
env vars, auth entries, or scripts.

On startup the provider probes for a local OpenAI-compatible server on port
`9337`. If nothing answers and a `llama-server` binary can be found
(`$LLAMASTACK_SERVER_BIN` → `$PATH` → conventional build locations), it spawns
the router itself:

- **detached** — the router survives opencode exiting, so the next launch (or
  another machine on the LAN) reuses it,
- **router mode with `--models-max 1`** — selecting a different model swaps it
  in VRAM instead of stacking servers until OOM,
- logging to `~/.local/state/llamastack/router.log`,

then polls `/v1/models` for up to 10 s. Discovered models appear under
"Llama Stack (local)" with tool calling and web search enabled, zero cost, and
the real per-model context advertised by the server (from `models.ini`).

Detection is memoized per process and never fails opencode's startup path.
Set `OPENCODE_DISABLE_LLAMASTACK=1` to skip it entirely.

Code: `packages/opencode/src/provider/llamastack.ts`.

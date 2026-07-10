# Model discovery from the LM Studio layout

Models are read from `~/.lmstudio/models` (override:
`$LLAMASTACK_MODELS_DIR`) in the LM Studio layout —
`<publisher>/<repo>/*.gguf` — so anything downloaded in LM Studio appears in
opencode automatically.

llama-server's own `--models-dir` scan is only one level deep, so the provider
generates a preset INI (`~/.local/state/llamastack/models.ini`, passed as
`--models-preset`) covering the nested repos, mirroring llama.cpp's own
heuristics:

- `mmproj` files are attached as multimodal projectors, not listed as models,
- the first shard of multi-shard models is used,
- **every quant variant in a repo is its own selectable model** (a repo with
  Q4_K_XL and Q5_K_XL GGUFs gives you both, each with its own settings
  section).

New models get a sensible default section appended (`ctx-size = 32768`,
`gpu-layers = 99`, `flash-attn = on`, `cache-type-k/v = q8_0`, `jinja = true`,
`cache-ram = 2048`, `reasoning-preserve = true`). Existing sections are
**never modified or removed** — the file is yours.

Code: `packages/opencode/src/provider/llamastack.ts` (provider),
`packages/tui/src/util/llamastack.ts` (TUI mirror).

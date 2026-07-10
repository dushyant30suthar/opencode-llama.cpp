# `/config` — in-TUI model settings screen

Type `/config` in the TUI (command palette: "Configure local models").

1. **Pick a model** — the list is the union of `models.ini` sections and
   models discovered on disk (marked "(defaults)" if they have no section
   yet). A **Server** entry at the top shows the endpoint URL and the loaded
   model's parameters.
2. **Edit settings** — pre-filled with current values, marked "(default)"
   when they match the defaults:
   - Context window (`ctx-size`)
   - GPU layers (`gpu-layers`)
   - Tensor split across GPUs (`tensor-split`, e.g. `0.5,0.5`; empty = auto)
   - KV cache quant K/V (`f16` | `q8_0` | `q4_0`)
   - Flash attention (on/off)
   - Temperature (empty = server default)
3. **Save** — writes the section back to `models.ini` preserving comments,
   unmanaged keys (`model`, `mmproj`, `jinja`, …) and all other sections
   verbatim, then asks the router to reload presets (`GET /models?reload=1`,
   which also unloads a running model whose preset changed). Router down?
   It just saves — settings apply on next load.

Extras:

- **"Use this model"** action switches the current session to the model
  straight from the settings screen (and warms it in VRAM immediately).
- **"Reset to recommended"** restores the machine-tuned values from the probe
  runs (see [tuning.md](../tuning.md)).
- Esc steps back one screen; numeric and split inputs are validated; if
  `models.ini` doesn't exist yet the screen creates it on save.

Code: `packages/tui/src/component/dialog-llamastack-config.tsx`,
`packages/tui/src/util/llamastack.ts` (line-preserving INI editor).

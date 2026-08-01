# config/

A mirror of `~/.config/opencode/` from the 2× RTX 5060 Ti box — the **real
deployed files**, not templates. `models.ini.example` predates this and is kept
as the generic starting point.

Everything here carries the measured reasoning inline: each tuned value has a
comment saying what it was swept against and what the alternative cost. That is
the point of committing them — the numbers are the documentation, and a config
without them is a set of magic constants.

```
config/
  opencode.jsonc                     plugin registration
  tui.jsonc
  providers/
    exl3/
      server.ini                     bin, tabby-dir, models-dir, host/port
      models/                        one TabbyAPI YAML per model
        Qwen3.6-27B-exl3-5.00bpw.yml            MTP, 4 slots, 311296 cache
        Qwen3.6-27B-exl3-5.00bpw-dflash.yml     DFlash, 1 slot, 294912
        Qwen3.6-27B-exl3-6.00bpw.yml            MTP, 4 slots, 163840
        Qwen3.6-27B-exl3-6.00bpw-dflash.yml     DFlash, 1 slot, 131072
    llamacpp/
      server.ini
      models.ini                     llama.cpp's own format, --models-preset
      templates/
    openvino/
      server.ini
```

## Reading order

`docs/HANDOVER-2026-08-01.md` explains where these values come from. In short:

- **`cache_mode: 6,4`** — exllamav3 dequantises inside the attention kernel at
  no latency cost, so fewer KV bits is *faster*. The opposite of llama.cpp,
  where f16 keys halve the depth slope. Do not carry KV settings across engines.
- **`draft_num_tokens: 4`** — on the DFlash configs this is load-bearing twice
  over: it truncates the drafted block *and* sets the cache's `max_history`.
  Leaving it unset takes the drafter's default of 15 and costs 15.5% acceptance
  instead of 50.5%.
- **`tensor_parallel: true`** — 74.5 vs 53.1 t/s @4k. exllamav3 #245 is
  WSL2-only.
- **`inline_model_loading`** — deliberately absent. exllamav3 does not return
  VRAM on unload, so an in-process model swap strands it and the next load
  fails. Switching models is a process restart, by design.
- **`cache_size` vs `max_seq_len`** — size the pool as a *multiple* of the max
  sequence, not equal to it. The 6.00bpw file's 1:1 ratio caused cache thrashing
  with 4 concurrent agents (39% reuse where an uncontended request got 96%).

## Machine-specific

Paths under `/home/dushyant30suthar/` and the model directory
`~/.lmstudio/models/turboderp` are this box's. GPU splits, cache sizes and
`sysmem_recurrent_cache` are tuned to 31.86 GiB of VRAM and 31 GB of RAM — they
are starting points elsewhere, not defaults.

No secrets: every `api-key` is empty (localhost-only serving).

# Benchmark & probe harness

The scripts that produced every number in [docs/tuning.md](../docs/tuning.md),
with their raw results committed alongside. Two design rules:

- **Production-faithful:** probes go through `llama-server` with the exact
  flags the router uses (mmproj loaded, tensor split, quantized KV,
  flash-attn), and success means a real chat completion — not just a load.
- **Adapt before running:** paths at the top of each script point at
  `~/Projects/llama/llama.cpp` and `~/.lmstudio/models`; adjust for your
  layout. Scripts assume idle GPUs and use port 9444 to avoid the live router.

| Script | What it measures | Results file |
| --- | --- | --- |
| `ctx-search-v2.sh` | Max loadable context per model, binary search via llama-server with production flags, verified by load + generation | `ctx-results-v2.txt` |
| `ctx-experiment.sh` | Earlier bare-model context search (kept to show why production-faithful probing matters — it overestimates by 60–80k) | `ctx-results.txt` |
| `flag-sweep.sh` | split-mode × ubatch sweep via llama-bench (pp2048 + tg128) | `sweep-results.txt` |
| `mtp-bench.sh` | MTP speculative decoding: baseline vs draft-mtp at several n-max values × KV quant, measured server-side over ~600 tokens of code-flavored generation | `mtp-bench-results.txt` |
| `mtp-ctx-search.sh` | Max context for the MTP build with speculative decoding active | `mtp-ctx-results.txt` |

Result files are raw script output: `label: pp_t/s tg_t/s` for bench sweeps,
`model max_ctx` for context searches.

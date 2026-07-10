# Tuning — how every number was found

Nothing in `config/models.ini.example` is a guess. Two principles:

1. **Probe through the production path.** Max context is validated by loading
   the model in `llama-server` with the exact production flags (mmproj
   included, tensor split, quantized KV, flash-attn) and requiring a real chat
   completion. Bare-model probes overestimate by 60–80k tokens — the ~900 MB
   vision projector and split-mode buffers are real.
2. **Don't re-litigate from single runs.** Single-run t/s differences of
   ±1–2 t/s are noise. A value only changes on repeated evidence.

The harness lives in [`bench/`](../bench/) — see its README for
script-by-script details. Raw results are committed next to the scripts.

## Max context per model (binary search, load + generate to verify)

| Model | Max ctx | Split | ubatch | Generation | Prompt |
| --- | --- | --- | --- | --- | --- |
| Qwen3.6-35B-A3B Q4_K_M (MoE) | 245,760 | tensor | 2048 | 152 t/s | 2,522 t/s |
| **Qwen3.6-27B-MTP Q4_K_XL** | 180,224 | tensor | 2048 | **70.6 t/s** (MTP n=4) | — |
| Qwen3.6-27B-MTP Q5_K_XL | 180,224 | tensor | 1024 | ~70 t/s (MTP n=4) | — |
| Qwen3.6-27B Q4_K_M | 180,224 (258,048 without mmproj) | tensor | 2048 | 40 t/s | 724 t/s |
| gemma-4-31B QAT Q4_0 | 147,456 | tensor | 512 | 37 t/s | 745 t/s |

Dropping the 27B's vision projector by choice freed ~900 MB VRAM for the full
258k context — vision is rarely worth 78k tokens of agent context.

## Split-mode / ubatch sweep (dual GPU without P2P)

From `bench/flag-sweep.sh` on the 35B MoE (pp2048 / tg128, `-ngl 99 -fa on`):

| Config | Prompt t/s | Gen t/s |
| --- | --- | --- |
| layer split, ub512 (baseline) | 3,404 | 133.4 |
| layer split, ub1024 | 3,613 | 133.3 |
| layer split, ub2048 | 3,184 | 133.8 |
| row split | **fails** — needs GPU P2P; unavailable across a PCH-attached x4 slot |
| tensor split, ub2048 | 2,522 | **152.5** |

**Generation-first is the deliberate trade:** `split-mode = tensor` costs ~30%
prompt speed but buys +14% generation on the MoE — and +72% on the dense 27B
(23 → 40 t/s). For agent workloads that generate far more than they ever
re-read cold prompts, generation wins. Prompt speed stays above 2,500 t/s.

## MTP speculative decoding (the champion config)

Qwen3.6-27B-MTP has built-in multi-token prediction — draft-free speculative
decoding. Measured server-side on Q4_K_XL, q8_0 KV (`bench/mtp-bench.sh`):

| Config | Gen t/s |
| --- | --- |
| baseline (no MTP) | 37.2 |
| draft-mtp, n-max 2 | 61.6 |
| draft-mtp, n-max 3 | 66.0 |
| **draft-mtp, n-max 4** | **70.6** |
| draft-mtp, n-max 5 | 68.3 |
| draft-mtp, n-max 6 | 61.5 |

n-max 4 is the peak — **1.9× the baseline**. Later single-run re-probes put
2/3/4 within ~1 t/s of each other; per principle 2, the proven 4 stays.
Constraints: MTP requires `parallel = 1` (single slot) and doesn't support
vision.

## Sampling

Thinking models use the official Qwen3.6 "precise coding" card values:
`temp = 0.6`, `top-p = 0.95`, `top-k = 20`, `min-p = 0`. If loops ever
reappear, `presence-penalty` up to 1.5 is the escape hatch. The frontend never
overrides these — see [architecture.md](architecture.md).

## Host-RAM lessons (32 GB box)

- **`cache-ram` (host prompt cache) is dangerous by default.** The 8 GiB
  default ballooned to 24 GB of host RAM with a 157k-token session — full
  memory exhaustion, load average 36. Standard transformers get
  `cache-ram = 2048`.
- **On hybrid-attention models the cap does not hold** (a 2048 cap still
  swapped ~27 GB) and prompt-cache restore is inert for them anyway
  ([llama.cpp #22615](https://github.com/ggml-org/llama.cpp/issues/22615)) —
  hybrids get `cache-ram = 0`.
- **`ctx-checkpoints` stays at stock 32** (~280 MB host RAM each, bounded
  ~9 GB). Lowering it to 8–16 caused full 30-second prompt re-evaluations
  every few turns — measurably worse than the RAM it saves.

## Engine build the numbers were measured on

llama.cpp **b9891** (`f36e5c348`) — the exact commit the `llama.cpp/`
submodule pins — built with CUDA 13.3.1, host g++-15, `GGML_CUDA=ON`,
`CMAKE_CUDA_ARCHITECTURES=120` (Blackwell), `GGML_CUDA_FA_ALL_QUANTS=ON`,
`GGML_CUDA_GRAPHS=ON`, `GGML_LTO=ON`, `GGML_NATIVE=ON`, Release, Ninja.

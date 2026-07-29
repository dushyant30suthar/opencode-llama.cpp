# Qwen3.6-27B serving stacks on this box — measured 2026-07-28

> **CORRECTION 2026-07-29.** Every exllamav3 decode number first published here
> came from `sweepx.py`'s client-side math, which divided all generated tokens
> by the window *after* the first token arrived — inflating short-generation
> rates ~1.7x. The llama.cpp numbers came from llama-server's own timings and
> were correct, so the cross-engine comparison was rigged by the measurement.
> Corrected, server-measured figures are in "THE CROWN" below; the headline
> "1.4-1.8x faster than llama.cpp" is withdrawn — the two are comparable, and
> llama.cpp is ahead at 163k. The caching results (97.6% hit rate, rebuild
> counts) were always server-side and stand.


One file, one row of truth per stack. Hardware: i5-9400F, 32GB RAM, 2× RTX 5060 Ti
16GB on PCIe Gen3 **x8/x4**. All numbers measured on this machine, this day, same
corpus, same sampling (temp 0.6, top-k 20, top-p 0.95). Raw data: `results.jsonl`
(llama.cpp), `exl3-results.jsonl` (exllamav3), `ckpt.jsonl` (divergence).

## Stack A — llama.cpp (PRODUCTION, deployed in models.ini)

Build ff067f76d (2026-07-26). NVFP4-MTP GGUF, `-sm tensor`, K=f16/V=q8_0,
draft KV q8, ctx 163840, `-cms 4096 -ctxcp 40`. Serving via router on :9337.

| depth | prefill t/s | decode t/s (MTP) | base step ms |
|------:|------:|------:|------:|
| 4k    | 701 | 70–75 | 43.2 |
| 65k   | 508 | ~58   | 49.0 |
| 94k   | 503 | ~55   | 59.5 |
| 131k  | 445 | ~50   | 66.6 |
| 163k  | 338 | ~45   | 72.2 |

- Cache model: ONE linear slot. Any second conversation evicts the first:
  re-entry costs 35–94 s at 61k depth, 2–6 min at 120k+. Measured live: a
  4-agent session wasted **94 min** on 72 forced re-prefills (3.57M tokens).
- Edit at 49k/30k/15k/3k in a 61k convo: 25.3 / 51.3 / 74.9 / 94.0 s.
  Flip-back after an edit costs the SAME again (slot overwritten).
- Upstream has explicitly declined insertion-tolerant caching (PR #24035
  discussion) — no structural fix coming.

## Stack B — exllamav3 / TabbyAPI (EXPERIMENTAL, installed at ~/Projects/tabbyAPI)

EXL3 5.00bpw (embeds MTP tensors), layer split (`gpu_split_auto`), Q8 cache pool
196608 tok, chunk 2048. Start: `cd ~/Projects/tabbyAPI && venv/bin/python main.py
--config config.yml` (port 5000). VRAM 15.4/9.9 GB → ~7 GB spare for a bigger pool.

| depth | prefill t/s | decode t/s (base) | decode (MTP, when it works) |
|------:|------:|------:|------:|
| 4k    | 709 | 21.7 | **63.8 (2.9×)** |
| 65k   | 297 | 17.8 | unmeasured (bug) |
| 94k   | 204 | 16.5 | unmeasured (bug) |
| 131k  | 181 | 15.1 | unmeasured (bug) |
| 163k  | 145 | 14.1 | unmeasured (bug) |

- **Cross-conversation caching — the headline, measured:** two conversations
  (61k, and one sharing a 30k prefix) alternating turns cost **1.7–2.2 s per
  switch** vs llama.cpp's 35–94 s. ~20–40× on the multi-agent pattern.
  Shared-prefix pages are deduped (B's cold start: 28 s, not 117 s).
- Edit test (same positions): 42.8 / 84.3 / 114.6 / 131.5 s — 1.4–1.7× worse
  one-way (slower prefill), BUT rewind is token-exact (no checkpoint slack) and
  **flip-back is free (2.15 s)** — the original's pages survive. For oscillating
  churn (Claude Code system-reminders), the round-trip roughly ties llama.cpp.
- Per-turn cache-hit tax grows linearly: ~1 s @4k → 4.3 s @163k (~30 µs/tok).
- Deep prefill is the real weakness: 2.3× slower at 163k. Layer split leaves
  GPU1 idle (GPU0 100%/167W, GPU1 1%/19W) — untested mitigations: tensor_parallel
  (the #245 crash was a WSL2 repro; native Linux may differ), bigger chunk_size.

### Bugs found, one FIXED locally (source build at ~/Projects/exllamav3)
1. **v1.2.x MTP crash on multi-GPU splits — ROOT-CAUSED AND FIXED (2026-07-28
   evening).** `_collect_rewind_jobs` in `modules/gated_delta_net.py` launched
   ALL GDN layers' state-rewind jobs on the first layer's device; under layer
   split the layers span both GPUs → illegal memory access at gdn.cu on every
   draft rejection. 22-line patch groups jobs per device. After the patch MTP
   runs clean on this box: **68.4 t/s at 4k depth (3.1x over base)**.
   Patch: `exllamav3-multigpu-rewind-fix.patch`; report:
   `exllamav3-upstream-report.md`. (This also explains v1.1.0's separate
   `list index out of range` as the pre-rewrite form of the same lifecycle.)
2. **STILL OPEN upstream: recurrent-state slot exhaustion with MTP + distinct
   conversations.** Each distinct prompt permanently holds a GPU state slot;
   pool is 4 (and physically can't be bigger — ~1.5-2 GB/slot on this model);
   5th distinct conversation aborts and wedges the generator. Same-conversation
   agentic use is stable (8/8). Without MTP, unlimited conversations work.
   Full trace + suggested fix in the upstream report. **This blocks the
   MTP + multi-agent combination** — the last piece of the ideal config.

## THE CROWN (late night 2026-07-28): both bugs fixed locally, TP+MTP measured

Bug #2 root cause: in the drafting branch, requeue-check preceded the EOS check;
a job finishing at max_new_tokens got requeued into a zombie with negative token
budget that held a state slot forever. Fix: `rq and not eos` (one condition).
Both fixes: `exllamav3-both-fixes.patch` (92 lines); posted to issue #260.
Validated 30/30 across all historical failure patterns.

| depth | TP+MTP (server) | LS+MTP (server) | llama.cpp prod | real traffic |
|------:|------:|------:|------:|------:|
| 4k    | **74.5** | 53.1 | 70-75 | 58 (<16k) |
| 32k   | **62.0** | 48.4 | ~62   | 67 (16-48k) |
| 94k   | **50.8** | 35.7 | ~55   | 55 (48-96k) |
| 131k  | **50.7** | 33.7 | ~50   | 45 (96-160k) |
| 163k  | 34.9 | 22.9 | **~45** | — |

Real-traffic column: 787 requests over 14 h on 2026-07-29, same server metric.
It sits on the TP+MTP curve, confirming the benchmark once measured correctly.

Agents test with TP+MTP: conversation switches **1.1–1.4 s** (llama.cpp 35–94 s).
Config: `tabbyAPI/config-crown.yml` (tensor_parallel + draft_mode mtp, 4 slots,
Q8 cache 196608). Serving: patched source venv `~/Projects/exllamav3/venv`.
Launch: `cd ~/Projects/tabbyAPI && ~/Projects/exllamav3/venv/bin/python main.py
--config config-crown.yml` (port 5000, OpenAI-compatible).

Caveats before making it production: cold/divergent prefill still ~2× slower
than llama.cpp (stall recovery), patches are local (watch #260 for upstream
merge), soak time measured in hours not days. Recommended: run on :5000
alongside the llama.cpp router; migrate clients per-session; flip the default
after a few days of clean soak or upstream merge, whichever first.

## Verdict (2026-07-28)

**llama.cpp stays the daily driver.** Fastest stable decode at every depth,
2.3× faster deep prefill, battle-tested. Its one flaw — single-conversation
cache — is mitigated behaviorally (sequential subagents, same-model rule,
compact by ~90k).

### TENSOR PARALLEL (tested late 2026-07-28): the base-decode king

`tensor_parallel: true`, MTP off, official 1.1.0 wheel, native Linux — the
feared #245 crash never appeared (it's WSL2-specific). Both GPUs sustain
146–157W (layer split idled one at 19W). Config: `config-tp.yml`.

| depth | TP decode | layer-split | llama.cpp base | TP prefill |
|------:|------:|------:|------:|------:|
| 4k    | **37.9** | 21.7 | 23.3 | 363 |
| 49k   | **33.0** | 18.4 | 18.9 | 373 |
| 94k   | **30.0** | 16.5 | 16.8 | 257 |
| 163k  | **26.1** | 14.1 | 13.9 | 175 |

+75–87% base decode at every depth, flatter depth slope (−31% over 160k vs
llama.cpp's −40%), cache-hit turn tax halved. Prefill still ~half of
llama.cpp — stall recovery remains its crown.

**This changes the TODAY options:** EXL3-TP-noMTP at 26–38 t/s is ~60% of
production speed but with 2-second agent switching and no MTP bugs in play —
for heavy fan-out days it plausibly wins on wall-clock already. And the #260
endgame projection becomes TP × MTP ≈ **75–110 t/s at depth** — 1.5–2×
production.

**exllamav3 is the architecture we want, now ONE bug from winning (was two).**
The multi-GPU MTP crash is fixed locally (see above; 68.4 t/s measured).
Remaining blocker: the state-slot exhaustion (#2) limits MTP to ≤4 distinct
conversations between restarts — fine for deep single-session work, fatal for
subagent fan-out. Usable TODAY per workload:
- deep single conversation, max speed: EXL3+MTP (patched) ≈ llama.cpp speed
  with better caching — viable now
- multi-agent fan-out: EXL3 without MTP (slow decode) or llama.cpp (thrash) —
  pick your poison until #2 is fixed upstream
Re-test on each exllamav3 release; harness is one command per mode:
`python3 sweepx.py http://127.0.0.1:5000 <name> sweep|edit|agents exl3-results.jsonl`

**Not viable (researched 2026-07-28, see memory):** vLLM (hybrid APC gives 0%
cross-conversation reuse — open #45238; MTP mutually exclusive with caching),
SGLang (real hybrid radix cache but VRAM-only for this arch + open Claude Code
tool-parser bugs), ik_llama.cpp (worse), TensorRT-LLM (SM120 consumer gaps).

---

## Tuning session 2026-07-29 (all server-measured, repeats where noted)

Baseline for this section: exllamav3 `dev` build (both #260 fixes upstream:
`ce77f0a`, `1048003`), TP + MTP, measured with the engine's own `Generate:`
metric — never client stream timing (see the correction banner above).

### What moved

| change | effect | evidence |
|---|---|---|
| `sysmem_recurrent_cache` 8192 → 4096 | swap 24 GB → 1 GB, RAM available 3 → 18 GB | TP duplicates host state; 8 GB of checkpoints on top forced the box into zram, and decompressing 19.6 GB of swapped-out server was burning CPU on every request |
| `cache_mode` Q8 → **6,4** | **+7.7% @98k, +9.3% @131k**, 2.88 GB freed | 4 reps/config, acceptance matched (77%, 94/95%); spread at 131k was [57.5-57.6] |
| `cache_size` 245760 → **360448** | +47% resident context | funded by the cache-mode saving |
| `draft_num_tokens` | **keep 4** | 2 → 55.8 t/s @4k (-28%); 6 → 50.7 (-33%, acceptance collapses 72% → 59%) |

### The counterintuitive one

exllamav3 wants MORE KV quantization, llama.cpp wanted LESS. On llama.cpp we
measured f16 keys halving the depth-degradation slope, because that engine
dequantizes as a separate step on every attention read. exllamav3 dequantizes
inside the attention kernel with no latency cost, so bytes-read is all that
matters and K6V4 > K8V8 > FP16 monotonically. Engine-specific; do not carry
settings across.

### Physics ceiling (why the levers are what they are)

Weights 19.9 GB, 10 GB per GPU under TP, 448 GB/s per card = 22 ms per decode
step = **45 t/s base ceiling**. Everything above that comes from MTP producing
several tokens per weight-read. So only two things raise the ceiling: fewer
bytes per read (quant, cache mode) or more tokens per read (draft window,
which is already at its optimum). A *larger* quant lowers the ceiling — 6bpw
would drop it to ~37 t/s.

### Config traps found

- `draft_num_tokens` under `model:` is silently ignored — it belongs to
  `draft_model:`. Pydantic `extra='ignore'` drops it with no warning, so a
  sweep of that key measures nothing. `scripts/knob-ab.sh` now asserts the
  value lands before running.
- draft window and `max_batch_size` both scale GDN recurrent state
  `(bsz, draft+1, 48, 128, 128)` fp32 x48 layers ~ 2.81 GB at bsz4/draft4.
  Window 6 OOM'd at Q8 before the cache change freed room.

### Prefill: at the hardware limit, and 26% of it is the x4 slot

Measured cold 98k prefill = 543-553 t/s (~180 s). llama.cpp on this box did
500-700 t/s, so this is parity, not a deficiency.

Where the 181 s goes (PCIe probed on this machine, rest derived):

- GPU0 sits on CPU root port 00:01.0, measured H2D 5.64 / D2H 4.32 GB/s (Gen3
  x8 — the 5060 Ti is a native x8 card, so this is not a misconfiguration).
- GPU1 sits on 00:1c.0, a **PCH port behind DMI 3.0** on the ROG STRIX B360-F,
  measured H2D 2.85 / D2H 2.76 GB/s. B360 does not bifurcate the CPU lanes, so
  GPU1 cannot be moved to a CPU-attached slot on this board.
- TP fires `tp_reduce` once per attn/GDN module and once per MLP = **128
  all-reduces x 64 layers per forward**, payload `[chunk, 5120]` at 2 B/element.
  Per rank per token that is 1.31 MB out + 1.31 MB in, **independent of
  chunk_size** — only the call count changes.
- For 98,304 tokens: 128.8 GB each direction per rank. At GPU1's 2.76 GB/s
  that is **~47 s = ~26% of wall time, a hard floor**.
- The residual ~134 s against ~3.4 PFLOP/GPU is ~25 TFLOP/s per GPU, which is
  where EXL3 trellis-dequant GEMM should land on this card.

So prefill is roughly **74% compute-bound at hardware speed, 26% bound by a
slot that cannot be changed**. Realistic tuning headroom is 10-20%, not 2x.

| chunk_size | cold 98k prefill | peak VRAM | verdict |
|---|---|---|---|
| 1024 | 531 t/s | 14521/13540 | slower, saves only ~220 MB |
| **2048** | **548 t/s** | 14713/13737 | **kept** |
| 4096 | 553 t/s | 15033/14057 | +0.9% for +320 MB — not worth the OOM margin |

Other prefill facts established:

- The MTP draft head prefills the whole prompt too (`job.py:1218`), on one GPU
  since `qwen3_5_mtp` declares `supports_tp: False` — costs ~5-8%. Keep it
  anyway: MTP is worth 2-2.8x on decode.
- The 48 gated-deltanet layers are *cheap* in prefill (chunked scan, no
  quadratic term). They are why this model prefills as fast as it does; the
  quadratic cost comes from the 16 full-attention layers.
- `tensor_parallel_backend: nccl` does not reduce PCIe bytes for 2 ranks
  (upstream #75 measured 438.2 nccl vs 439.4 native T/s). Skipped.
- Dead ends on this box, verified: `EXL3_BC_ATTN` (decode-only, declines under
  TP), `EXL3_PREFER_FA2` (flash-attn not installed), `EXL3_TP_REDUCE_THREADS`
  (AVX-512 only; i5-9400F is AVX2), moving GPU1 to another slot (B360).
- **No prefill persistence exists** in either engine. A restart always costs a
  full reprocess, and it is harder to add here than llama.cpp's slot-save
  because it would need both the paged K/V and the GDN recurrent checkpoints.

### Final deployed config (tabbyAPI/config-crown.yml)

```
cache_mode: 6,4        cache_size: 262144     chunk_size: 2048
max_seq_len: 196608    tensor_parallel: true  max_batch_size: 4
draft_model: draft_mode mtp, draft_num_tokens 4
memory: sysmem_recurrent_cache 4096
```

VRAM 14297/13369 MiB at rest (peak +1.2 GB during prefill), RAM 15 GB used.
Decode ~74 t/s shallow, ~57 at 131k. Prefill ~548 t/s. Tool calls verified.

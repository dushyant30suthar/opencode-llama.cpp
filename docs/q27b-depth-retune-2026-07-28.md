# Qwen3.6-27B depth retune — 2026-07-28

Controlled follow-up to `docs/q27b-production-depth-2026-07-28.md`. Everything
below was measured on the production API path (`/completion`, MTP active,
production sampling) with exact-depth token-array prompts built from real
source code — not scraped from a live session. Method: `sweep.py` /`ckpt.py`,
raw data `results.jsonl` / `ckpt.jsonl` (session scratchpad, copies alongside
this file if committed).

The deployed outcome is at the bottom; the original doc's errors are corrected
inline because several of its numbers are now known to be artifacts.

---

## 1. Corrections to the original doc

| original claim | reality (measured) |
| --- | --- |
| "Decode has a knee, not a slope. Flat to 64k, then −37%, then flat." | **Wrong — it is a clean linear slope.** Base step latency: 43 ms @4k → 48.8 @32k → 57.4 @64k → 65.3 @98k → 82.5 ms @164k. The "knee" came from having 10 samples below 64k and treating MTP-acceptance noise (0.6–1.0 per request) as depth signal. |
| "MTP acceptance measured 0.847 at 64–128k depth" | Session-level acceptance is right (median 0.849), but the per-record `depth` field in the collector is not conversation depth — the by-depth attribution was never in the data. Controlled runs show acceptance is depth-independent (0.60–1.0 noise at every depth). |
| Checkpoint ≈ 144 MiB (48×6144×128×4B estimate) | **~150 MiB fixed + ~3.9 KiB/token** (the f16 MTP-draft KV snapshot). Measured from trace logs: 214 MiB @16k, 278 @32k, 391 @61k. At 120k depth a checkpoint is ~620 MiB, not 144. |
| "Restart with `-lv 1`" for checkpoint logs | Checkpoint create/restore lines are `SLT_TRC` = verbosity **4**. Use `-lv 4` (`LLAMA_SERVER_SLOTS_DEBUG=1` additionally enables `/slots` debug fields). |
| "prefill ~500–570 for planning" | Confirmed: 701 @4k → 508 @65k → 445 @131k → 338 @164k (sustained, ub 2048). |

The doc's core mechanism (§3, hybrid rewind → checkpoint fallback → giant
re-prefill) is **correct** and was verified in source and reproduced on demand
(§4 below). Its dead-ends table also held up where tested.

---

## 2. What actually makes depth expensive: the KV dequant

Same depths, same everything, only KV cache type changed (ctx 98304):

| depth | q8/q8 step ms | f16/f16 | K=f16 V=q8 | K=q8 V=f16 |
| --- | --- | --- | --- | --- |
| 4096  | 42.2 | 42.5 | 42.6 | 43.2 |
| 49152 | 52.4 | 47.0 | 49.2 | 48.9 |
| 94208 | 63.4 | 52.4 | 57.8 | 57.7 |

- Depth slope: q8/q8 ≈ 0.235 ms/step per 1k tokens; f16/f16 ≈ **0.110** — half.
  The deep-decode tax is mostly the FlashAttention dequant of quantized KV,
  not memory bandwidth (f16 reads 2× the bytes and is still 2× cheaper per
  token of depth).
- K and V each pay exactly half. Either mix = half the win.
- **Mixed KV beats both purs on prefill**: 551/550 t/s @94k vs 501 (q8) and
  503 (f16). ~+10% prefill for free.
- Allocated ctx is speed-neutral: q8 @98304-alloc == q8 @196608-alloc at every
  matched depth. You only pay for actual depth, never for the allocation.

Split-mode re-check (196k, q8): layer split = 60 ms steps @4k → 134 @164k,
GPUs ~35% util (pipeline-serialized), 2× the depth slope (KV reads don't
parallelize across GPUs). `tensor` confirmed correct for this workload; layer
only wins shallow cold prefill (1011 vs 701 t/s ≤65k).

Draft (MTP) KV quant: `--cache-type-k-draft/-v-draft q8_0` costs ~2% step,
acceptance unchanged (0.63–0.72 both ways), frees ~0.45 GB VRAM and halves the
checkpoint linear term (3.9 → 2.1 KiB/token).

---

## 3. The stalls, reproduced on demand

61k-token conversation, 40 tokens mutated at position P (simulating Claude
Code's system-reminder / file re-injection churn), defaults vs
`-cms 4096 -ctxcp 40+`:

| edit at | default: reprocessed / wall | cms 4096 | delta |
| --- | --- | --- | --- |
| 49152 | 20,484 tok / 35.1 s | 14,336 / 25.3 s | **−30%** |
| 30720 | 30,720 / 51.0 s | 30,720 / 51.3 s | 0% (checkpoint sat exactly at the edit) |
| 15360 | 55,296 / 86.0 s | 47,104 / 74.9 s | **−13%** |
| 3072  | 61,440 / **94.0 s full recompute** | identical | 0% — below the first checkpoint, no setting helps |

Structural read: checkpoints only trim the rewind *slack* (≤ spacing); the
re-prefill of everything above the edit is unavoidable server-side. An edit at
3k in a 61k conversation is a full recompute under any flag combination — the
live `cached=0` events (7+ full 100–121k recomputes in 30 minutes, one 6-minute
stall) are this exact case at production depth. The only real fixes for those
are client-side (stop mutating early context) or compacting before depth
accumulates.

RAM: build-to-61k checkpoint overhead was 2.0 GB (defaults) vs 2.6 GB
(cms 4096) with f16 draft; q8 draft roughly halves the growth term.

---

## 4. Deployed config (models.ini, 2026-07-28)

```
ctx-size = 163840            # was 196608 — pays for f16 K; alloc size is speed-neutral
cache-type-k = f16           # was q8_0 — keys are the quality-sensitive half anyway
cache-type-v = q8_0
cache-type-k-draft = q8_0    # new — draft was f16: 768M VRAM + 3.9K/tok in every checkpoint
cache-type-v-draft = q8_0
checkpoint-min-step = 4096   # was 8192 — measured −13..30% on mid-conversation stalls
ctx-checkpoints = 40         # 40×4096 covers 163840 exactly; early ckpts never evicted
```

Measured on the deployed candidate (full 163k prefill survived, ~1.1–1.2 GB
free per GPU):

| depth | old step ms | new step ms | old→new decode trend |
| --- | --- | --- | --- |
| 94208  | 63.4 | 59.5 | −6% |
| 131072 | 74.3 | 66.8 | −10% |
| 163328 | 82.5 | ~72  | −13% |

Fallbacks, all measured: RAM pressure → `checkpoint-min-step 8192` +
`ctx-checkpoints 32` (lose ~15% on stalls). Need >164k ctx back → revert K to
q8_0 and ctx to 196608 (lose the depth-slope win). Max decode, short sessions →
f16/f16 @ 98304 (step 52.4 @94k, −17% vs old).

Operating guidance unchanged from the original doc and now better-grounded:
**compact or clear by ~60–90k.** Decode is still 67–73 t/s there, every stall
scales with depth-above-the-edit, and the worst case (early edit = full
recompute) costs depth ÷ ~500 t/s seconds no matter what the server does.

---

## 5. Machine facts that shaped this (verify before porting elsewhere)

- 2× RTX 5060 Ti 16 GB on PCIe **Gen3 x8 / Gen3 x4** (i5-9400F, 6 cores, 32 GB
  RAM). Decode under tensor split draws only ~120 W/GPU of 180 — sync-bound,
  which is why layer split's serialization is so costly here.
- GPU VRAM after deploy: 14.9/15.1 of 16.3 GB. The gemma lesson from this file's
  history applies: loads-fine ≠ survives-deep-prefill; the deploy candidate was
  stress-tested with a real 163k prefill before shipping.
- Router: manually launched `llama-server --models-preset .../models.ini` on
  :9337 (ptyxis terminal, not systemd). `RESTORE.sh` in the session scratchpad
  restarts it.

# MTP on Qwen3.6-27B — what is measured, and what is not

**The one-line version:** MTP is the single biggest speed lever on this rig
(**1.7–1.9×**, and it is quality-free), but **every MTP number we publish was
measured at `ctx 32768` while production runs `ctx-size = 196608`.** External
work says speculative gains compress at depth, so the headline 76.3 t/s is an
upper bound for a workload shape we do not actually run.

Companion to [`docs/tuning.md`](tuning.md), which records the winning values.
This file records what those values are *conditional on*.

---

## 1. What MTP is, and why it is not a quality tradeoff

Qwen3.6-27B-MTP ships **multi-token prediction** tensors — draft-free
speculative decoding. The model proposes several tokens ahead from its own
extra heads, then the full model **verifies** them in one pass and rejects any
that it would not itself have produced.

That verification step is what makes it **output-preserving**. MTP is not a
speed-for-quality trade; it is the same tokens, sooner. Nothing in this
document trades intelligence for throughput.

Two hard constraints: MTP requires `parallel = 1` (single slot), and it does
not support vision.

---

## 2. Measured: the MTP ladder

Server-measured, `bench/mtp-bench.sh` and `bench/nvfp4-bench.sh`.

**Q4_K_XL, q8_0 KV:**

| Config | Gen t/s |
| --- | --- |
| baseline (no MTP) | 37.2 |
| n-max 2 | 61.6 |
| n-max 3 | 66.0 |
| **n-max 4** | **70.6** |
| n-max 5 | 68.3 |
| n-max 6 | 61.5 |

**NVFP4 (production quant):**

| Config | Gen t/s |
| --- | --- |
| baseline (no MTP) | 40.5 |
| **n-max 3** | **76.3** |
| n-max 4 | 73.2 |
| n-max 5 | 69.0 |

Two things worth carrying forward: the peak **moved from n-max 4 to n-max 3**
when the quant changed, so draft depth is not portable across quants; and the
multiplier is **1.9×** on Q4_K_XL, **1.88×** on NVFP4 — MTP is worth more than
every other flag on this rig combined.

Independent confirmation from a different harness
([`bench/q27-ngram-sweep-results.txt`](../bench/q27-ngram-sweep-results.txt),
f16 KV, code prompt):

```
mtp-n3-control      tg = 70.73 t/s
no-spec-control     tg = 40.99 t/s     → 1.73×
```

---

## 3. ⚠️ The gap: every number above is a 32k number

Each of the five `q27-*` sweeps pins context at **32768**:

| Bench | Context | Headline |
| --- | --- | --- |
| [`q27-spec-sweep`](../bench/q27-spec-sweep-results.txt) | `ctx 32768` | n3 = 70.44 t/s |
| [`q27-sync-sweep`](../bench/q27-sync-sweep-results.txt) | `ctx 32768` | tensor-n3 = 68.12 t/s |
| [`q27-ngram-sweep`](../bench/q27-ngram-sweep-results.txt) | `ctx 32768` | mtp-n3 = 70.73 t/s |
| [`q27-latency-sweep`](../bench/q27-latency-sweep-results.txt) | `ctx 32768` | control = 70.20 t/s |
| [`q27-backend-sampling`](../bench/q27-backend-sampling-results.txt) | `ctx 32768` | tensor-n3 = 68.50 t/s |

`config/models.ini.example` runs the NVFP4 profile at **`ctx-size = 196608`** —
**6× deeper than anything we measured.**

An external spec-decode study (2026-07, Spec-Bench across quants) states the
caveat directly:

> Spec-Bench with short outputs, greedy, batch 1 is close to a **best case** for
> speculative decoding. Under concurrency the per-stream gains shrink and
> **longer contexts will compress them further**, so treat these as an **upper
> bound** for this workload shape, not a general speedup.

**We already proved this principle on our own hardware — on the other model.**
From [`docs/laguna-retune-2026-07-27.md`](laguna-retune-2026-07-27.md):

> Depth matters and should always be quoted: prefill at 62k is 410–515 t/s, not
> the 523–570 measured at 32k. Decode at 62k is 10.7–14.9, not 15–19.

That is a **~25% decode drop from 32k to 62k** on Laguna S. The lesson was
applied there and has not yet been applied to the 27B — where the number is
more visible, because 76.3 t/s is quoted in the repository README.

**Nothing here says 76.3 is wrong.** It says it is unqualified. Until the
depth sweep in §7 runs, quote it as *"76.3 t/s at 32k"*, the same way the
Laguna numbers are quoted.

---

## 4. Measured: ngram speculation does nothing

Also from [`q27-ngram-sweep`](../bench/q27-ngram-sweep-results.txt) — the
cleanest result in the whole harness:

| Config | Gen t/s | vs no-spec |
| --- | --- | --- |
| no-spec-control | 40.99 | — |
| `ngram-mod` | 41.01 | **+0.05%** |
| `ngram-simple` | 40.86 | −0.3% |
| `ngram-map-k` | 40.75 | −0.6% |
| `ngram-map-k4v` | 40.67 | −0.8% |
| `ngram-cache` | 31.92 | **−22%** |

Depth does not rescue it — `ngram-mod` at n4 / n8 / n16 gives 40.53 / 40.68 /
40.68, i.e. flat. And **stacking it onto MTP makes things worse**:
`mtp3+ngram-mod` = 68.73 vs `mtp-n3` alone = 70.73 (**−2.8%**).

This matches the external finding ("ngram ~1.03× regardless of quant, and
net-negative under concurrency") and our own Laguna-XS data
([`laguna-xs-tune-results.txt`](../bench/laguna-xs-tune-results.txt)): f16
KV + ngram 147.3 vs f16 KV alone 146.9, again inside noise.

### Consequence for the shipped config

`config/models.ini.example` currently carries two ngram recommendations that
this data does not support:

1. The header **SPEED PROFILE** suggests `spec-type = ngram-mod` for "snappier
   agent turns." It buys ~0%.
2. The **Laguna-XS profile** sets `spec-type = ngram-mod` and justifies f16 KV
   with *"draft-free speculative decoding conflicts with a quantized KV
   cache."* The rationale is inverted: the f16-vs-q8 gap (147 vs 135 t/s) is
   the **KV quant**, not ngram. Keeping f16 KV is defensible on its own merits;
   keeping ngram is not.

Neither is harmful, both are noise-level, and the XS one costs double KV memory
for a stated reason that does not hold.

---

## 5. Measured: everything else that was swept, and lost

Recorded so none of it gets re-litigated.

**Split mode** ([`q27-sync-sweep`](../bench/q27-sync-sweep-results.txt),
[`q27-backend-sampling`](../bench/q27-backend-sampling-results.txt)):

| Mode | Gen t/s | Note |
| --- | --- | --- |
| **tensor** | **68.1–68.5** | sampler runs on CPU |
| layer | 47.8 | sampler runs on GPU, still 30% slower |
| row | LOAD_FAILED | needs GPU P2P — unavailable over the PCH x4 slot |

Note the counterintuitive bit: **layer split moves draft sampling onto the
GPU and is still much slower.** Deeper drafts under layer split do raise mean
accepted length (2.96 → 3.43 at n8 → 3.58 at n16) but throughput falls anyway
(47.8 → 41.1 → 40.9). Acceptance is not the objective; wall-clock is.

**Draft depth / p-min** ([`q27-spec-sweep`](../bench/q27-spec-sweep-results.txt)):
`n3/p0.0` = 70.44 is the peak. Every deeper config loses — `n8/p0.0` 59.94,
`n8/p0.6` 44.33, `n8/p0.75` 31.57. `n16` fails to load at any p-min.

**Poll / latency knobs** ([`q27-latency-sweep`](../bench/q27-latency-sweep-results.txt)):
control 70.20, `poll-100` 68.56, `poll-0` 58.95. The default wins; `--poll 0`
costs 16%.

**PCIe** ([`q27-pcie-probe`](../bench/q27-pcie-probe.txt)): confirms Gen3 **x8**
on GPU0 and **x4** on GPU1 (the PCH-attached slot), both ~80–90% utilised under
load. Draft acceptance measured at 0.573, mean length 2.71. This is the
hardware reason row-split is unavailable and tensor-split is the ceiling.

---

## 6. External findings not yet reproduced here

Single-source, dated, and **not measured on this rig**. Listed because two of
them predict things our own harness would not currently detect.

| Finding | Source | Bearing on us |
| --- | --- | --- |
| **The heavier the quant, the more spec-decode buys you** — 10/10 configs rank Q8 > Q6 > Q4 by multiplier. Acceptance is quant-independent at matched depth; the base step slows with weight bytes, draft+verify overhead does not. | 2026-07 spec-decode study | Directly relevant to any Q5/Q6 A/B — the cost of a heavier quant is partly repaid by a larger MTP multiplier. |
| **llama.cpp's MTP path on UD-Q4 is "pathologically slow"** — Q4 MTP-3 slower in absolute t/s than Q6 MTP-3 at identical acceptance, which bandwidth cannot explain. Output correct, mechanism unknown. | same | Our `Q4_K_XL` + MTP row (70.6 t/s) is exactly this combination. That baseline may be measuring a bug, which would make heavier quants look better than our table implies. |
| **nvfp4 is the fastest quant/engine pair overall** | same | Independent confirmation of the current champion's format. |
| Longer context and concurrency both compress speculative gains | same | §3. |

A caution on comparing numbers across our own files: NVFP4 n-max 3 reads
**76.3 t/s** in [`mtp-bench`](../bench/mtp-bench-results.txt) but **75.1 t/s**
in [`nvfp4-tune-results.txt`](../bench/nvfp4-tune-results.txt), and 62.1 t/s in
the PCIe probe. Different harnesses, prompts and warmup handling. Quote the
file, never a bare number.

---

## 7. Open work

### Task 1 — Measure MTP at production depth ★

The gap in §3. Everything else here is bookkeeping; this is the one that could
change a shipped value.

1. Re-run the MTP ladder (n-max 2/3/4/5) at **32k, 96k, and 196k** on the NVFP4
   profile, using the depth-aware method from the Laguna retune: realistic
   filler (not a repeated sentence — it flatters speculation), one **discarded
   warmup** (the first generation after load builds CUDA graphs and was worth a
   60% discrepancy there), and ≥4 repeats.
2. Record decode **and** prefill at each depth, plus draft acceptance and mean
   accepted length.
3. **Done when:** `docs/tuning.md` and the README quote 27B throughput with a
   depth attached, and we know whether `n-max 3` is still the peak at 196k. If
   the optimum shifts with depth, `models.ini` should follow the depth we
   actually run at.

### Task 2 — Drop the ngram recommendations

§4. Remove `spec-type = ngram-mod` from the Laguna-XS profile and correct the
comment (keep f16 KV, fix the stated reason); revisit the header SPEED PROFILE.
No measurement needed — the data is already in the repo.

### Task 3 — Fold the depth caveat into the quant question

Any NVFP4 → Q5_K_XL / Q6_K comparison must sweep MTP `n-max` rather than
inheriting a value, because the peak moved between quants (n4 → n3) and the
external study says the multiplier grows with weight size. Comparing NVFP4 at
n3 against Q5_K_XL at n4 is not a matched test.

### Explicitly not worth re-testing

- ngram, in any variant or depth (§4)
- layer or row split (§5)
- draft depth above n-max 5 (§5)
- `--poll` tuning (§5)

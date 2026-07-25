# Laguna S 2.1 / XS 2.1 — status, configuration, and open work

**Purpose:** a self-contained handoff for anyone (human or agent) picking up
Laguna work on this rig. Everything below is either measured here (linked to
the raw file in [`bench/`](../bench/)) or attributed to a dated external
report. Nothing is a guess — where something is unverified, it says so.

**Status as of 2026-07-25:** Laguna S 2.1 is **not** the daily driver. It is a
**second-line escalation model** for hard debugging and long-context tool
loops. The daily driver remains Qwen3.6-27B-NVFP4-MTP (76 t/s, 196k ctx). The
model is 3 days old, its ecosystem is still being patched daily, and several
of its problems are packaging bugs rather than model quality.

---

## 1. What the model is

| | Laguna S 2.1 | Laguna XS 2.1 |
| --- | --- | --- |
| Architecture | MoE, 118B total / ~8B active | MoE, ~33B |
| Released | 2026-07-22 | 2026-07-02 |
| Context | 1M native (YaRN-extended from 8192) | — |
| Our quant | `unsloth/Laguna-S-2.1-UD-Q2_K_XL` (36.96 GiB) | `Q4_K_M` |
| Fits 32 GB VRAM? | No — expert layers offload to CPU | **Yes, fully resident** |

Poolside published the raw generations behind their benchmark claims at
<https://trajectories.poolside.ai/>, which makes "fabricated benchmarks" the
least likely explanation for the gap between their numbers and community
results. The more probable story is that they benchmarked full precision on
an internal harness, and what shipped broken was the *packaging* — chat
template, quant metadata, and third-party engine integrations.

### Where it earns its place

It is **not** a Qwen replacement. Its value is that it over-thinks:

- **Complex debugging.** It keeps going after the first root cause instead of
  stopping there. Independently reported by several users as finding bugs that
  Qwen, DeepSeek and Claude missed (2026-07-24, r/LocalLLaMA).
- **Long-context tool-call stability.** One user working at 200k–600k context
  reports Qwen3.6-27B collapses into repeated tool calls above ~150k while
  Laguna stays stable (2026-07-25, r/LocalLLaMA). **Unverified here** — we
  have not run a >150k agentic session on either model.

Its cost is time: 60k+ thinking tokens on a hard problem is normal, which at
our ~16 t/s is over an hour. Use it asynchronously, not interactively.

---

## 2. Why Q2_K_XL and not a higher quant

Counter-intuitive but measured twice, here and externally: for this MoE,
**keeping every expert and quantising harder beats keeping fewer bits of
model**.

Our head-to-head ([`bench/verifier-quality.txt`](../bench/verifier-quality.txt),
[`bench/q2kxl-tune.txt`](../bench/q2kxl-tune.txt)) — 6 planted bugs, blind:

| Candidate | Score | Wall | Decode | Prefill | Size |
| --- | --- | --- | --- | --- | --- |
| **UD-Q2_K_XL @ K15** | **6/6** | **432 s** | **16.68 t/s** | **547.9 t/s** | 36.96 GiB |
| UD-IQ3_XXS @ K13 | 6/6 | 521 s | 14.08 t/s | 462.7 t/s | 41.24 GiB |

Q2_K_XL wins on every axis. The smaller file lets `K` rise 13 → 15 (more
expert layers resident on GPU), which is where the speed comes from.
`IQ3_XXS` is the one unsloth rung off the pareto frontier; `IQ3_S` scores the
same while being larger.

**External corroboration (2026-07-24, A. Fateev):** pruning S 2.1 INT4 from
256 → 128 experts to fit 2×3090 scored **56/75** (REAP) or **45/75** (routing
mass), while keeping all 256 experts at **Q2 scored 63/75** — near the full
INT4 baseline of 64/75. HumanEval+ 138/164 (Q2) vs 119/164 (pruned).
Conclusion: **do not chase expert-pruned checkpoints.**

**Caveat on the quant floor.** The SLQ paper (arXiv 2605.02404, IST-DASLab)
puts *task-lossless* compression at ~3.3 bits and *distribution-lossless* at
5–6 bits. Q2 is below both. Our 6/6 result is a task-level measurement; a
long agentic chain can accumulate distribution error that a bug-finding suite
does not expose. Treat Q2_K_XL as validated for **short escalation tasks**,
not proven for **long autonomous runs**.

---

## 3. Production configuration

`K` = number of expert layers kept on GPU per half of the 48-layer stack. The
`-ot` regex sends the rest to CPU. Generator (from
[`bench/laguna-s-dflash.sh`](../bench/laguna-s-dflash.sh)):

```python
K = 15
cpu = [l for l in range(48) if not (24-K <= l <= 23 or 48-K <= l <= 47)]
regex = 'blk\\.(' + '|'.join(map(str, cpu)) + ')\\.ffn_.*_exps\\.=CPU'
```

`models.ini` section — **not yet committed to `models.ini.example`**, see task
2 in §6:

```ini
[unsloth/Laguna-S-2.1-GGUF]
model = ~/.lmstudio/models/unsloth/Laguna-S-2.1-GGUF/Laguna-S-2.1-UD-Q2_K_XL.gguf
# role: second-line escalation debugger. NOT the daily driver.
# K15 expert-offload regex must be passed via -ot (see above) — models.ini
# cannot express the generated regex, so this profile is launched by script.
ctx-size = 98304
gpu-layers = 99
flash-attn = on
cache-type-k = q8_0
cache-type-v = q8_0
ubatch-size = 2048
no-mmap = true
jinja = true
cache-ram = 0
```

Hard limits found by probing ([`bench/q2kxl-tune.txt`](../bench/q2kxl-tune.txt),
[`bench/laguna-s-ctx100k.txt`](../bench/laguna-s-ctx100k.txt)):

- `K17` — LOAD_FAILED. `K15` is the ceiling.
- `K15` + ctx 131072 or 196608 — LOAD_FAILED. 98304 is the ceiling at K15.
- Trading K for context: `K11-ctx98304` = 12.79 t/s, `K11-ctx131072` = 8.81 t/s.
  Not worth it; keep K15 @ 98304.

`--no-mmap` is not optional — prefill goes from 327 t/s (mmap, ub2048) to
451.9 t/s without it ([`bench/laguna-prefill.txt`](../bench/laguna-prefill.txt)).
llama.cpp warns about this directly when `-ot` overrides are combined with mmap.

### XS 2.1 profile

Fully VRAM-resident, so it is a completely different performance regime
([`bench/laguna-xs-tune-results.txt`](../bench/laguna-xs-tune-results.txt)):

| Config | Decode | Prefill |
| --- | --- | --- |
| f16 KV + ngram, ub2048 | **147.3 t/s** | 672.6 t/s |
| q8 KV, ub512 | 138.6 t/s | 535.2 t/s |
| ctx 262144 (q8 KV) | 132.8 t/s | 305.3 t/s |

Max context 262144; 393216 does not fit.

---

## 4. Rejected options — with the measurements

Do not re-litigate these without new evidence. Each was measured here.

### DFlash speculative decoding — rejected on both tiers

**S tier** ([`bench/laguna-dflash-results.txt`](../bench/laguna-dflash-results.txt)):

| Config | Decode | vs baseline |
| --- | --- | --- |
| K13 no-draft baseline | 12.02 t/s | — |
| K13 + DFlash n=7 | 12.35 t/s | +2.7% (noise) |
| K13 + DFlash n=15 | 7.20 t/s | −40% |
| K11 + DFlash n=7 | 10.88 t/s | worse than K13 baseline |

The draft *worked* — acceptance 0.587 (mean accepted length 5.11), better than
poolside's published ~3.1 on text. It still does not pay off, because the
target model verifies mostly **on CPU**: verifying N drafted tokens costs ~N×
instead of ~1×. The 2.23 GiB draft also forces K down, making the target
slower. This is a structural property of CPU-offloaded MoE, not a tuning miss.

**XS tier** — also rejected, and this is the important one because XS *is*
fully resident, so the CPU-verify argument does not apply
([`bench/laguna-xs-dflash-greedy.txt`](../bench/laguna-xs-dflash-greedy.txt),
tested with greedy sampling since poolside benchmarked greedy):

| Config | Decode | Acceptance |
| --- | --- | --- |
| greedy baseline | 134.9 t/s | — |
| greedy + DFlash n=3 | 135.1 t/s | 0.456 |
| greedy + DFlash n=7 | 98.7 t/s | 0.227 |

n=3 is inside noise; n=7 is a 27% regression. **DFlash is not a lever on this
hardware at either tier.**

Also note: on upstream llama.cpp the S draft fails to load outright — tensor
contract mismatch, expected 76 got 69
([`bench/laguna-s-tune-results.txt`](../bench/laguna-s-tune-results.txt)). It
only loads on poolside's `laguna` branch. Upstream support is not merged; the
PR author states it "will follow in a future PR."

### YaRN rope parameters — measured, no quality gain, 17% slower

The GGUF is missing `rope.scaling.orig_ctx_len`. We supplied it and re-ran the
quality suite ([`bench/ropefix.txt`](../bench/ropefix.txt),
[`bench/verifier-quality.txt`](../bench/verifier-quality.txt)):

| Arm | Score | Wall | Thinking chars (bug 1) |
| --- | --- | --- | --- |
| Q2_K_XL K15 (as shipped) | 6/6 | **432 s** | 15,159 |
| Q2_K_XL + YaRN fix | 6/6 | 507 s | 17,640 |

Same score, 17% slower, more thinking. Rejected.

**⚠️ Open question — this may not be the same fix the community is discussing.**
Our test supplied **`yarn-orig-ctx`**. A widely-reported fix (2026-07-25,
r/LocalLLaMA) is that the unsloth GGUFs ship
**`yarn-attn-factor = 1.4852030277252197`** where it should be **`1.0`**, and
that forcing 1.0 at runtime resolves looping. These are *different
parameters*. Two things follow:

1. Not passing YaRN flags does **not** disable the bad value — llama.cpp reads
   rope scaling from GGUF metadata unless overridden. If the bad attn factor
   is present, we have been running with it.
2. Our quality suite uses **short prompts** (~123-token prompt eval, see
   [`bench/ropefix.log`](../bench/ropefix.log)). Community looping reports
   start at **50k+ real context**. Our bench cannot see the failure mode it is
   supposed to fix.

Resolving this is task 1 in §6.

### Static reasoning budget — rejected

Capping thinking depth was rejected on the grounds that depth is the model's
to choose. Worth revisiting **only** for the escalation role, where a runaway
reasoning chain is the dominant failure: one external user reports good
results at Q2_K_XL with a 512-token reasoning budget (2026-07-25). Not tested
here.

---

## 5. Upstream status and known bugs

| Item | Status (2026-07-25) | Source |
| --- | --- | --- |
| Laguna arch in llama.cpp | **Merged 2026-07-22**, PR ggml-org#25165 | verified |
| Our llama.cpp pin | pinned to the PR branch (commit `5219577`) — can move to mainline | repo |
| DFlash upstream | **Not merged.** "Will follow in a future PR" | PR #25165 |
| Partial GPU expert offload | **Open bug** — non-monotonic perf degradation and crashes | PR #25165, 2026-07-23 |
| Chat template | Initial release broken; poolside published a corrected `chat_template.jinja` | community |
| `yarn-attn-factor` in unsloth GGUFs | Reported wrong (1.4852… vs 1.0) | community, unverified here |
| Reasoning loops | **Not fixed.** Reported at NVFP4, FP8 *and* Q5_K_M | HF discussion, r/LocalLLaMA |
| Tokenizer warning | `special_eos_id is not in special_eog_ids` on every load | our logs |

**The partial-expert-offload bug is the reason not to move off the pinned
build yet.** It is exactly the code path this model depends on here.

**Precision is not the explanatory variable.** Looping is reported at FP8 and
Q5_K_M, while Q4_K_XL and Q2_K_XL work for others once the corrected chat
template is used. Current best hypothesis: **chat template + rope metadata**,
not bit depth. This matters because template fixes are free and more VRAM is
not.

---

## 6. Open work

Ordered. Each task states what to run and what "done" looks like.

### Task 1 — Determine whether `yarn-attn-factor` is wrong in our GGUF ★ highest value

Free to test, and it is the leading candidate for the looping other people see.

1. Read the value actually baked into our file:
   ```sh
   llama.cpp/build/bin/llama-gguf ~/.lmstudio/models/unsloth/Laguna-S-2.1-GGUF/Laguna-S-2.1-UD-Q2_K_XL.gguf \
     | grep -i -E 'yarn|rope'
   ```
   Record `rope.scaling.attn_factor` and `rope.scaling.orig_ctx_len` verbatim.
2. If `attn_factor` ≈ 1.4852030277252197, re-run the quality suite with
   `--yarn-attn-factor 1.0` added, **at long context** — the existing suite's
   short prompts will not reproduce looping. Build a ≥50k-token prompt.
3. **Done when:** we can state, with a committed bench file, whether forcing
   1.0 changes long-context behaviour. Record it either way.

Do **not** simply adopt the community's full YaRN block
(`rope-scale 32`, `yarn-orig-ctx 8192`) as a bundle — we already measured that
combination as 17% slower with no quality gain. Isolate `attn_factor`.

### Task 2 — Commit the Laguna profile to `config/models.ini.example`

The tuned S and XS settings currently live only in commit `3bf7541`'s message
and this document. Add both sections with the `# role:` comments, and note the
`-ot` regex cannot be expressed in `models.ini` (see §3).

### Task 3 — Get poolside's corrected `chat_template.jinja` and A/B it

Several users report the corrected template is what separates working setups
from looping ones, at quants as low as Q4_K_XL. Fetch it from poolside's HF
repo, pass via `--chat-template-file`, re-run the quality suite. Check whether
it also clears the `special_eos_id` tokenizer warning.

### Task 4 — Verify the long-context claim that would actually justify this model

The one reported capability we cannot get from Qwen is stability past ~150k
context with heavy tool use. Run an agentic session at 150k+ on both
Qwen3.6-27B-NVFP4-MTP and Laguna S 2.1, and count repeated/looping tool calls.
Our max Laguna context is 98304 at K15, so this needs a K/context trade
(`K11-ctx131072` = 8.81 t/s) — slow, but the question is stability, not speed.
**If this reproduces, it is the strongest argument for keeping Laguna. If it
does not, the escalation role rests on debugging quality alone.**

### Task 5 — Re-check upstream in ~2 weeks

Move off the PR-branch pin to mainline once the partial-expert-offload bug is
fixed. Watch for the follow-up DFlash PR (it will not change our conclusion —
see §4 — but it changes what loads). Re-run
[`bench/upstream-verify.txt`](../bench/upstream-verify.txt)'s comparison after
any move.

### Explicitly not worth doing

- **Buying hardware for this model.** A GB10 box (DGX Spark / ASUS GX10,
  128 GB @ 273 GB/s) runs S 2.1 at ~41 t/s fully tuned — roughly half our
  Qwen daily driver's 76 t/s, for ~$3,000. Loop-free reports cluster on FP8,
  which implies two boxes. Looping is also reported *at* FP8, so the spend
  does not reliably buy the fix.
- **Expert-pruned checkpoints** (REAP or routing-mass). See §2.
- **Higher quants of the 27B daily driver.** NVFP4 is already above the
  task-lossless threshold; Q8 would cost 196k → ~8–16k context and half the
  speed, and lose MTP.

---

## 7. Measurement notes

Method matches [`docs/tuning.md`](tuning.md): probe through the production
path (`llama-server` with the real flags, requiring a real chat completion),
and do not change a value on a single run — ±1–2 t/s is noise.

Two numbers in `bench/` look contradictory and are not:

- **S decode ranges 12–20 t/s across files.** It is sensitive to K, context,
  mmap, and prompt length. `20.0 t/s` in [`ropefix.log`](../bench/ropefix.log)
  is a 123-token prompt with 230 graphs reused; `16.68 t/s` in
  [`q2kxl-tune.txt`](../bench/q2kxl-tune.txt) is the K-ladder condition. Quote
  the file, not a bare number.
- **XS decode is 147 t/s here but 102 t/s in
  [`upstream-verify.txt`](../bench/upstream-verify.txt).** Different builds and
  measurement conditions; the upstream-verify run is the apples-to-apples one
  for build comparisons.

Concurrency on S ([`bench/laguna-s-conc2.txt`](../bench/laguna-s-conc2.txt)):
aggregate throughput barely scales — N=1 11.6 t/s, N=8 16.7 t/s, with
per-stream collapsing to 2.4 t/s. Do not plan on batching this model.

---

## 8. External evidence log

Dated, so it can be aged out. None of this is verified on our hardware.

| Date | Claim | Source |
| --- | --- | --- |
| 2026-07-22 | Laguna arch merged into llama.cpp mainline | PR ggml-org#25165 |
| 2026-07-23 | Partial GPU expert offload crashes post-merge | PR #25165 comments |
| 2026-07-24 | Q2-all-experts (63/75) beats expert-pruned (56/75) at same size | A. Fateev, X |
| 2026-07-24 | Finds bugs Qwen/DeepSeek/Claude missed; specialist not generalist | r/LocalLLaMA |
| 2026-07-25 | unsloth GGUFs ship `yarn-attn-factor` 1.4852…; forcing 1.0 fixes looping | r/LocalLLaMA |
| 2026-07-25 | Q4_K_XL + corrected poolside template works on mainline | r/LocalLLaMA |
| 2026-07-25 | Works in OpenCode at `iq4_xs`, "no issues so far" | r/LocalLLaMA |
| 2026-07-25 | Qwen3.6-27B collapses >150k ctx; Laguna stable with heavy tool use | r/LocalLLaMA |
| 2026-07-25 | Looping still reported on FP8 and Q5_K_M after updates | HF discussion |
| 2026-07-25 | Tool calls degrade outside poolside's own CLI agent | r/LocalLLaMA |

The consistent community advice, and ours: **a model three days old is not a
model you tune against.** Re-evaluate in two weeks.

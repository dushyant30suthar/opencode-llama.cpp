# Qwen3.6-27B at production depth — 2026-07-28

Closes Task 1 of [`mtp-27b.md`](mtp-27b.md): *"every MTP number we publish was
measured at `ctx 32768` while production runs `ctx-size = 196608`."*

Measured **passively on a live agentic session** — Claude Code running on another
machine (192.168.1.4) against this box over the LAN, 1379 samples, depth 2k to
123k. Nothing was sent to the model; the collector polls `/slots` and tails the
server log. A real workload at real depth cannot be manufactured after the fact,
so it was recorded while it was happening.

---

## 1. The headline: MTP does **not** compress at depth

`mtp-27b.md` §3 warned that "longer contexts will compress speculative gains,"
citing an external study. **On this model that is wrong, and measurably so.**

| | repo's 32k bench | **live, 64–128k** |
| --- | --- | --- |
| draft acceptance | 0.573 | **0.847** |
| mean accepted length | 2.71 | **3.54** |
| samples | — | 148 |

MTP accepts **more** at depth, not less. So the throughput drop below is *not*
speculation degrading — it is the base decode step slowing. That distinction
matters: had MTP compressed, the fix would be re-sweeping `n-max`. It hasn't, so
there is nothing to tune there.

**Caveat, stated plainly:** depth is not the only variable. The 32k bench used a
code prompt; this is a live agent session full of tool output and structured
text, which is inherently more predictable and would raise acceptance on its own.
This shows MTP is *not the culprit* — not that depth *improves* it.

---

## 2. The depth curve

NVFP4, `ctx 196608`, q8_0 KV, `split-mode tensor`, `ub 2048`, MTP `n-max 3`,
`parallel 1`. Medians.

| depth | decode t/s | prefill t/s (sustained) |
| --- | --- | --- |
| ≤32k | **71.3** | 688 |
| 32–64k | 69.1 | 615 |
| 64–96k | 43.6 | 529 |
| 96–128k | 44.9 | 488 |

**Decode has a knee, not a slope.** Flat to 64k, then −37%, then flat again.
Whatever changes, changes between 64k and 96k.

**Prefill declines smoothly, −29% across the range.** It does not collapse.

The 32k decode figure (71.3) independently **validates** the repo's 76.3 t/s from
`mtp-bench-results.txt` — same ballpark, live workload, different harness.

### A measurement trap worth recording

An earlier pass of this analysis reported *"prefill collapses to 115.9 t/s at
96–128k."* **That was an artifact and is wrong.**

`prefill t/s` is computed as `Δn_prompt_tokens_processed / Δt`. On a cache-hit
turn only ~370 tokens are processed, so the sample divides a tiny numerator by
fixed per-request overhead and yields ~20 t/s. Binning those together with real
prefills dragged the median to 116.

Split by magnitude:

```
sustained prefill (>200 t/s): n=296, median 567 t/s
low samples      (≤200 t/s): n=142, median  20 t/s   ← not prefill rates
```

The table above filters to sustained samples. **Use ~500–570 t/s** for planning;
it matches direct observation of `processed` climbing 6,144 → 64,662.

---

## 3. The actual user-visible problem, and its real cause

Two consecutive turns in the same conversation at 117,649 tokens:

| turn | cached | processed | wait |
| --- | --- | --- | --- |
| A | 117,194 | 367 | instant |
| B | 52,670 | **64,979** | **~2 minutes** |

64,979 ÷ ~500 t/s ≈ 130 s. The arithmetic checks out.

### It is not (only) linear prefix-cache behaviour

**This model is hybrid.** Verified from GGUF metadata:

```
general.architecture            qwen35
qwen35.full_attention_interval  4
qwen35.ssm.state_size           128
qwen35.ssm.inner_size           6144
qwen35.block_count              65
```

~16 full-attention layers, ~49 **recurrent** (gated-deltanet). `llama-model.cpp`
puts this on `llama_memory_hybrid`.

**A recurrent state cannot be rewound to an arbitrary position.** So on
divergence the server does not resume at the longest common prefix — it falls
back to the nearest **context checkpoint at or before it**
(`server-context.cpp:3289-3324`), and `n_prompt_tokens_cache` is assigned *after*
that pull-back (line 3352).

Therefore **`cache = 52,670` is not where the client diverged.** It is the last
checkpoint before wherever it diverged. A divergence at 110k can still cost a
rewind to 52k.

Checkpoint spacing is a flag, and its default is coarse (`common/common.h:624`):

```c
int32_t n_ctx_checkpoints   = 32;    // max checkpoints per slot
int32_t checkpoint_min_step = 8192;  // minimum spacing
```

### Why the client diverges at all

llama.cpp's prefix cache is strictly linear. Claude Code is not append-only — it
injects `<system-reminder>` blocks, re-injects file contents when files change on
disk, and adds todo/task reminders. Any insertion at position N invalidates
everything after N. Claude Code targets Anthropic's `cache_control` breakpoints,
which tolerate mid-conversation edits; llama.cpp has no analogue, merged or
proposed.

---

## 4. Dead ends — confirmed in source, do not retry

| lever | verdict |
| --- | --- |
| `--cache-reuse` / `n_cache_reuse` | **Silently ignored on this model.** `LLM_ARCH_QWEN35` → `LLAMA_ROPE_TYPE_IMROPE` → `n_pos_per_embd()==4` → `get_can_shift()` false → `server-context.cpp:3168` logs "cache reuse is not supported - ignoring". And even where it works it handles *deletions* from the cache, never *insertions* — `A+B` vs `A+X+B` recovers nothing. |
| bounded `--cache-ram` (2–4 GB) | **Stores nothing.** One entry at this depth is full KV (~34 KiB/token ≈ 4.0 GiB at 118k) *plus* its checkpoints (~4.6 GiB) ≈ 8–9 GiB. `server-task.cpp:1679` skips any entry over the cap. Would need 10–18 GiB — the swap event already recorded in `models.ini`. |
| `--cache-idle-slots` | **Irrelevant to a single-slot server.** Disabled when `cache_ram == 0` anyway. It publishes a RAM copy of an *idle* slot; with one slot and one linear conversation that copy is always a prefix of the live slot and never preferred. Helps only two *divergent* prefixes alternating. |
| `/slots save\|restore` | Upstream issue #25913 (open): on hybrid models restore reports success, then the next request reports `cache_n = 0` and reprocesses everything, because checkpoints are not persisted with the snapshot. |
| `--parallel > 1` | Halves `n_ctx_seq` (kv_unified=false), doubles checkpoint RAM, and one client lands on the same slot by LCP similarity anyway. |

**Correction to `models.ini`:** the comment *"MTP requires `parallel = 1`"* is not
a llama.cpp constraint. `common_speculative_init(params, n_parallel)` and
`common_speculative_impl_draft_mtp(params, n_seq)` both take a sequence count.
It is our convention, not the engine's.

---

## 5. Operating recommendation

At ~500 t/s sustained prefill, worst case is a full recompute of the session:

| re-prefill budget | tokens recomputed | implied depth |
| --- | --- | --- |
| 30 s | 15k | 16k |
| 60 s | 30k | 30k |
| 120 s | 60k | 60k |
| **current 118k** | — | **~235 s worst case** |

16k is unusable for agentic work. **Compact or clear at 48–64k**: it caps the
worst case near 2 minutes instead of 4, and recovers decode throughput — 69.1 t/s
in the 32–64k bin versus 44.9 at 96–128k, **+54%**.

That is the recommendation that holds even if nothing in §6 lands.

---

## 6. Open work

### Task A — capture the divergence (free, do first)

Restart with `LLAMA_SERVER_SLOTS_DEBUG=1` and `-lv 1`. The server then logs
`restored context checkpoint (pos_min, pos_max, n_past)` (line 3316) and dumps
`old:`/`new:` tokens around the mismatch (3248-3287). Over ~20 turns that gives:

- the **true** longest common prefix versus the post-checkpoint `n_past` — i.e.
  whether checkpoint spacing or client churn dominates
- the literal text Claude Code mutates
- measured per-checkpoint RAM

Everything below is guesswork until this is captured.

### Task B — sweep `--checkpoint-min-step`

Default 8192. Halving it halves the worst-case rewind (~16 s → ~4 s at 500 t/s).
Cost is real: each checkpoint stores the full recurrent state, estimated
48 × 6144 × 128 × 4 B ≈ **144 MiB**, so 32 of them ≈ 4.6 GiB of host RAM already.
Lowering the step without raising `--n-ctx-checkpoints` *shrinks coverage*
(32 × 2048 = 65k instead of 262k). Measure real per-checkpoint size in Task A
before choosing. Candidate: `--checkpoint-min-step 4096 --n-ctx-checkpoints 40`.

### Task C — `196608 @ q8_0` vs `98304 @ f16`

Speculative for this arch, but our own Laguna bench measured f16 KV worth **+22%
decode / +15% prefill** over q8_0, because quantized KV pays a dequant on every
attention read. Roughly VRAM-neutral here (98304 @ f16 ≈ 8.1 GB vs 196608 @ q8_0
≈ 8.6 GB), and halving ctx also halves the worst-case reset. Different
architecture, so the +22% is not guaranteed to carry — needs an A/B.

### Task D — client-side hardening (cheap, low expected yield)

`CLAUDE_CODE_ATTRIBUTION_HEADER=0` is the known llama.cpp/Claude Code fix
(issue #19494) for a per-request hash in the first system block driving LCP to
zero. **Our data proves that is not happening here** — Turn A cached 99.6% — but
it is free hardening. `CLAUDE_CODE_DISABLE_AUTO_MEMORY=1` and
`CLAUDE_CODE_DISABLE_BACKGROUND_TASKS=1` remove minor churn. No documented switch
suppresses `system-reminder` injection or file re-injection.

---

## 7. Method

`scripts/collect-q27-production.py` (added with this doc). Read-only: polls
`/slots` every 2 s and tails the server log, classifying each sample as decode,
prefill, or a completed-request acceptance record. Never sends a request.

Two things that would have corrupted the result and were caught:

1. **The server log is shared across every model the router starts.** A naive
   grep for `draft acceptance` returned 660 records with `mean len` up to 32 —
   impossible at `n-max 3`. Those were other models (DFlash runs at n-max 32–64).
   Segmenting by PID cut it to the 148 belonging to this server.
2. **The prefill artifact in §2.** Filter to sustained samples before binning.

Raw data: `bench/q27-production-2026-07-28.jsonl`.

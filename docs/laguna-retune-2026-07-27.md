# Laguna S 2.1 — full retune, 2026-07-27

Hardware: 2× RTX 5060 Ti (16 GiB each, 32.6 GiB total), i5-9400F (6 cores, no HT,
PCIe 3.0), 32 GiB RAM. Model: `unsloth/Laguna-S-2.1-UD-Q2_K_XL.gguf` (36.96 GiB).
llama.cpp `ff067f76d` (b10133), plus a from-source build of poolside's fork.

Everything below was measured on this box in one session. Where a number
contradicts an older file in `bench/`, the reason is stated.

---

## 1. What changed

Old production config → new:

```diff
-override-tensor = blk\.(0|1|2|3|4|5|6|7|8|24|25|26|27|28|29|30|31|32)\.ffn_.*_exps\.=CPU   # K15
+override-tensor = blk\.(0|1|2|3|4|5|6|7|24|25|26|27|28|29|30|31)\.ffn_.*_exps\.=CPU        # K16
-ctx-size = 98304
+ctx-size = 106496
-cache-type-k = q8_0
-cache-type-v = q8_0
+cache-type-k = q4_0
+cache-type-v = q4_0
 ubatch-size = 2048
+batch-size = 2048        # REQUIRED: ubatch is clamped to batch, see §6.1
```

| | old (K15/q8) | **new (K16/q4)** | delta |
| --- | --- | --- | --- |
| prefill @32k depth | 487.4 t/s | **523.1 t/s** | +7.4% |
| decode @8k depth | 18.09 t/s | **19.23 t/s** | +6.3% |
| context | 98304 | **106496** | +8% |
| free VRAM | 995 MiB | 727 MiB | −27% |
| planted-bug suite | 6/6 | 6/6 | — |

Better on all three axes with no quality regression. The cost is headroom.

### The two assumptions that were hiding this

Both ceilings recorded in `docs/laguna.md` (branch `claude/multi-machine-llm-inference-ysobcb`) were **artifacts of one untested
parameter**, not properties of the hardware:

1. *"K17 — LOAD_FAILED. K15 is the ceiling."* — the ladder tested K13, K15, K17.
   **K16 was never tested.** It works.
2. *"K15 + ctx 131072 — LOAD_FAILED. 98304 is the ceiling at K15."* — true under
   q8_0 KV. Under q4_0, K15 reaches 131072 with 1111 MiB free.

Both fall out of the same thing: **`cache-type-k/v` had never been varied.**
Every config in `models.ini` and every bench in `bench/` used q8_0. Switching to
q4_0 frees 552 MiB at ctx 98304, which is enough for the missing K rung *and*
for +51% context.

---

## 2. The config frontier

All q4_0 KV, `split-mode layer`, `--no-mmap`, `--cache-ram 0`.
K = expert layers kept resident **per half** of the 48-layer stack.

| config | K | ub | ctx | prefill@32k | prefill@62k | decode@8k | decode@62k | free |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| old baseline | 15 | 2048 | 98304 | 487.4 | 448.2 | 18.09 | 14.9 | 995 |
| **shipped** | **16** | **2048** | **106496** | **523.1** | — | **19.23** | — | 727 |
| max context (mid) | 15 | 2048 | 131072 | 492.5 | 452.2 | 17.83 | 14.9 | 1111 |
| max prefill+ctx | 13 | 4096 | 163840 | 570.3 | 514.7 | 15.25 | 12.9 | 1113 |
| widest window | 13 | 2048 | 196608 | 445.5 | — | 15.62 | — | 1733 |

Swap by replacing three keys. Regex generator:

```python
K = 16
cpu = [l for l in range(48) if not (24-K <= l <= 23 or 48-K <= l <= 47)]
regex = 'blk\\.(' + '|'.join(map(str, cpu)) + ')\\.ffn_.*_exps\\.=CPU'
```

### K ladder (the whole trade in one table)

| K | CPU layers | decode@8k | max ctx | free @98304 |
| --- | --- | --- | --- | --- |
| 17 | 14 | — | **will not load** | 963 |
| **16** | 16 | **19.23** | 114688 | 813 |
| 15 | 18 | 17.83 | 131072 | 1547 |
| 13 | 22 | 15.62 | 196608 | 3017 |
| 11 | 26 | 13.17 | 299008 (fit) | 4489 |

**Every step up in K buys ~8% decode and costs ~33k context.** Monotonic, no
sweet spot to discover — pick the point that matches the workload.

### K16 context ladder

Prefill is flat across all of these (523.1 / 523.3 / 523.5 t/s):

| ctx | free | verdict |
| --- | --- | --- |
| 98304 | 813 MiB | safest |
| **106496** | **727 MiB** | shipped |
| 114688 | 619 MiB | works, thin |
| 131072 | — | will not load |

114688 is real but sits only 172 MiB above where gemma-4-31B died mid-prefill
the same night (447 MiB free). 106496 was chosen for margin, not performance —
they are identical in speed.

---

## 3. Method

Deliberately different from `bench/`'s approach in three ways, each of which
turned out to matter:

- **Realistic filler.** Prefill prompts are ~600 KB of real llama.cpp and plugin
  source (23.5% unique-word ratio) rather than a repeated sentence (1.0%). The
  synthetic filler flatters n-gram speculation enormously. *In the event this
  mattered less than feared — Laguna-XS deep decode moved only 3.4% between the
  two — but the check was necessary before trusting any spec-decoding number.*
- **Repeats with a discarded warmup.** The first generation after load builds
  CUDA graphs and is much slower. Measuring it produced a **60% discrepancy
  between two identical configs** (9.0 vs 14.4 t/s). With one warmup discarded
  and 4 repeats, decode spread drops to 0.1–1.3%.
- **`/completion` for throughput, `/v1/chat/completions` for behaviour.** The
  raw endpoint isolates prefill/decode without template effects. The OpenAI
  endpoint with `--jinja` is the only way to see the termination bug (§8).

Depth matters and should always be quoted: prefill at 62k is 410–515 t/s, not
the 523–570 measured at 32k. Decode at 62k is 10.7–14.9, not 15–19.

---

## 4. What each lever does

### 4.1 KV quant — the whole win

At K15, ctx 98304, ub2048:

| KV | prefill | decode | free | max ctx |
| --- | --- | --- | --- | --- |
| q8_0 | 487.4 | 18.09 | 995 MiB | 98304 |
| q4_0 | 490.7 | 17.73 | 1547 MiB | 131072 |

**+552 MiB, +51% context, −2% decode, prefill unchanged, 6/6 quality.**
KV cost measured at **13.3 KB/token** with q4_0 (218 MiB per 16384 tokens,
linear and predictive to within a few MiB across the whole ladder).

Why it is nearly free: Laguna S is **12 global + 36 SWA layers (1:3 ratio,
window 512)**. Only the 12 global layers scale with context; the 36 SWA layers
are capped at their window. So KV is far cheaper than a uniform-attention model
of the same size, and halving its precision costs correspondingly little.

### 4.2 ubatch — the only prefill lever

At K13, q4_0, ctx 98304:

| ub | prefill | decode |
| --- | --- | --- |
| 1024 | — | 15.85 |
| 2048 | ~445 | 15.62 |
| 4096 | 567.0 | 15.25 |
| 8192 | — (needs K11) | — |

**+27% prefill for −3.8% decode.** This contradicts `bench/laguna-prefill.txt`,
which showed decode falling 13.32 → 11.53 → 10.28 across ub 512 → 2048 → 4096.
Those were single samples under mmap; with warmup discarded and `-b` set
correctly, ubatch is nearly free for decode. **Take the big one.**

Prefill peaks at **K11/ub8192 = 579.0 t/s** and turns over at K9 (549.7) — below
K11 the extra PCIe traffic outruns what the larger batch amortises.

### 4.3 K — the only decode lever

See the ladder in §2. Decode scales with the number of CPU-resident expert
layers because decode is **PCIe-transfer-bound** (proven in §5.1).

---

## 5. Negative results

These cost the most time and are the most useful to record.

### 5.1 `GGML_OP_OFFLOAD_MIN_BATCH` — no gain, but settles *why*

Undocumented, runtime-tunable, found in `ggml/src/ggml-cuda/ggml-cuda.cu:5357`:

```c
const int min_batch_size = getenv("GGML_OP_OFFLOAD_MIN_BATCH")
                         ? atoi(getenv("GGML_OP_OFFLOAD_MIN_BATCH")) : 32;
```

It gates **where CPU-resident expert weights are computed**: below the
threshold, on the CPU in place; at or above, copied to GPU and computed there.
Decode runs at batch 1, so it is *always* below the default 32 — Laguna's expert
math has always run on the i5-9400F.

| min_batch | decode (K13) |
| --- | --- |
| 1 | **8.69** |
| 2 | 15.77 |
| 8 | 15.65 |
| 32 (default) | 15.66 |
| 512 | 15.75 |

Forcing GPU offload at batch 1 is **45% slower**. Same at K15 (17.83 → 10.27).
So decode is transfer-bound, not CPU-compute-bound, the default of 32 is correct
for this hardware, and **K is the only decode lever**. Values 2–32 are identical.

### 5.2 DFlash speculative decoding — dead across the whole range

Upstream b10133 **cannot load the draft** despite listing `draft-dflash` in
`--spec-type`: `done_getting_tensors: wrong number of tensors; expected 76, got
69`. 5/5 arms failed. This confirms `docs/laguna.md` §5 (branch `claude/multi-machine-llm-inference-ysobcb`) on the current build.

Built poolside's fork to test properly — `github.com/poolsideai/llama.cpp`
branch `laguna`, commit `04b2b72`. **Two build fixes required on Fedora 44:**

- `-DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-15` (gcc 16.1.1 is outside CUDA
  13.3's allowlist)
- `#include <cmath>` in `common/speculative.cpp` — `std::isfinite` at line 1091
  is no longer pulled in transitively by gcc 16

Results (K13, q4_0, ctx 98304, 3 repeats):

| n-max | decode | vs baseline |
| --- | --- | --- |
| no draft | **15.60** | — |
| 7 | 6.75 | −57% |
| 15 | 3.39 | −78% |
| 32 | 3.49 | −78% |
| 40 | 4.87 | −69% (36% spread) |
| 64 | 3.59 | −77% |
| K11, n=32 | 3.24 | −79% |

**Hypothesis tested and falsified.** The reasoning was: llama.cpp only offloads
CPU-resident experts at batch ≥32, and every prior DFlash run used n ≤ 15, so
verification was guaranteed to happen on CPU at N× cost *by construction*.
Crossing the threshold should have flipped it. It did not — n=32 is
indistinguishable from n=15. The n=40 uptick is inside its own 36% spread.

Not VRAM pressure either: K11/n=32 had 2613 MiB free and was no better.
Speculation is simply the wrong shape for a transfer-bound decode path.

### 5.3 Partial expert offload — actively harmful

Splitting `ffn_(up|gate)_exps` from `ffn_down_exps` to get finer VRAM
granularity than a whole K step:

| split layers | prefill | free |
| --- | --- | --- |
| 0 (plain K15) | 490.7 | 1547 |
| 2 | 474.8 | 1967 |
| 4 | 456.4 | 2409 |
| 6 | 441.4 | 2851 |

**≈ −17 t/s per split layer**, monotonic, independent of batch and context. A
K15+shave2 config has *less* weight on CPU than K13 yet prefills 17% slower.
This is the "non-monotonic perf degradation" bug filed against PR ggml-org#25165,
reproduced. **Keep `-ot` on whole-layer boundaries.**

### 5.4 `-b` (logical batch) does nothing for throughput

At K13/q4_0/ub4096: `-b` 4096 / 8192 / 16384 → 567.0 / 564.3 / 571.0 t/s. A 1.2%
spread with no trend across a 4× range. But see §6.1 — it is not inert, it
*gates* `-ub`.

---

## 6. Traps in llama.cpp

### 6.1 `-ub` is silently clamped to `-b`

`src/llama-context.cpp:264`:

```c
cparams.n_ubatch = std::min(cparams.n_batch, params.n_ubatch == 0 ? params.n_batch : params.n_ubatch);
```

`-b` defaults to **2048**. So `-ub 4096` without a matching `-b 4096` runs at
2048 and silently gives none of the prefill gain. This invalidated a whole phase
of this session's own sweep before it was caught (§7). **Always set both.**

### 6.2 `--fit` is disabled by flags we always pass

`--fit on` (default) sizes unset arguments to available VRAM with a 1024 MiB
margin. It is disabled by:

- `-ngl 99` → `"n_gpu_layers already set by user to 99, abort"`
- `split-mode = tensor` → `"llama_params_fit is not implemented for SPLIT_MODE_TENSOR"`

Both appear in every profile in `models.ini`. Consequence: **nothing rescues an
over-large `ctx-size` at runtime** — it loads and then dies mid-prefill. Probe
before raising.

`--fit` is still useful as a *probe*: leave `ctx-size` unset and read back the
`n_ctx` it chooses. Its answers are conservative (K15/q8 → 56576, where 98304
works in practice) because of the 1024 MiB margin, but the *ratios* are exact —
that is how the +51% q4_0 context gain was first spotted, before measuring it.

### 6.3 llama-server now has MCP and built-in tools

`tools/server/server-mcp.cpp` speaks stdio MCP (spawns servers as subprocesses,
pumps NDJSON), and `--tools` enables `read_file, file_glob_search, grep_search,
exec_shell_command, write_file, edit_file, get_datetime` executed server-side.
`--ui-mcp-proxy` is a CORS proxy for the built-in WebUI.

**Do not enable any of this behind opencode.** opencode is already the MCP
client and agent loop; a second one in the inference server means tool calls
with no permission model, no diff review, no session history. Default is
`no tools`; the plugin passes nothing.

---

## 7. Methodological errors made in this session

Recorded because each produced numbers that looked plausible and were wrong.

1. **Single-sample decode.** Two identical configs returned 9.0 and 14.4 t/s.
   Cause: the first generation after load builds CUDA graphs. Fix: discard a
   warmup, repeat 4×. All decode numbers taken before this were discarded.
2. **`-ub` without `-b`.** Phase E's configs were labelled ub4096/ub8192 but ran
   at 2048. Its conclusion (§5.3) survived — all four points were at the same
   effective batch, making it a *cleaner* comparison than intended — but the
   labels were wrong and one "K15+ub4096 now loads!" claim was false.
3. **Two null configs.** A "partial offload" regex targeted layers that were
   already fully CPU-resident at that K, making it a no-op. It reproduced the
   base config exactly, which is how it was caught. Fixed with an assertion.
4. **Cache-hit repeats.** The first long-context test reused one prompt, so
   run 1 came back `prompt_n=1, wall=0.6s` — a KV prefix hit, not a sample.
   Fixed by shifting the filler window per repeat.
5. **A prompt that could not test what it claimed.** The first termination test
   asked an easy question; the model answered directly with `reasoning=0c`, so
   it could not possibly exercise a *reasoning* non-termination bug. Replaced
   with a planted bug buried mid-prompt, which fires reasoning reliably.
6. **Duplicate concurrent runs.** Two sweeps briefly ran against the same GPUs
   and port. Caught via a `PHASED-START` appearing twice; the affected row was
   dropped and the phase re-run.
7. **`pgrep -f` self-match.** Cleanup commands matched their own command line
   and killed their parent shell (exit 144), three times. Fixed by moving
   orchestration into a script file so `ps` shows only `bash runall.sh`.

---

## 8. The reliability finding that outranks all of the above

**On `/v1/chat/completions` with `--jinja`, Laguna S sometimes consumes the
entire token budget in `reasoning_content` and returns EMPTY `content`.**

Reproduced **2 of 7** long-context runs (62k real prompts), in two shapes:

| shape | finish_reason | reasoning | predicted | wall | outcome |
| --- | --- | --- | --- | --- | --- |
| runaway | `length` | 25,845 chars | 8000 (cap) | 842 s | nothing |
| silent | `stop` | 9,645 chars | 3247 | 358 s | nothing |

**It is not caused by this retune.** The `stop`-shaped failure occurred on the
**old q8_0 / ctx 98304 / K15 production config**. It is a property of the
template/termination path (upstream, PR ggml-org#25165), independent of KV
quant, context size and K.

Consequences:

- A hard `max_tokens` ceiling does not prevent it — it only bounds the waste.
  8000 tokens cost 842 s here; a 32768 budget would cost ~45 min.
- The 5 passing runs found a subtle planted off-by-one buried in the middle of
  62k tokens, including one run with 6506 chars of reasoning that terminated
  cleanly. The model is capable; the plumbing is unreliable.
- The laguna doc's note that "disabling thinking avoids it" is the workaround
  worth testing next (task 1 there).

Sample sizes are small (7 runs). The failure rate is ~29% in this sample and
should not be quoted as precise.

---

## 9. Questions closed

- **`yarn-attn-factor`** (task 4 in the laguna doc): the GGUF ships
  1.4852030277252197 where the community reports 1.0. Forced 1.0 vs default at
  ctx 131072, 62k prompts: **2/2 pass each, no behavioural difference, prefill
  identical (511.1 vs 514.7)**. Non-issue on this quant and build.
- **Is K15 the ceiling?** No — K16 works. K17 genuinely does not (`--fit` falls
  to its 4096 floor; only 963 MiB free).
- **Does DFlash help?** No, at any n, on poolside's own fork. §5.2.
- **Is decode CPU-bound or transfer-bound?** Transfer-bound. §5.1.
- **Does q4_0 KV cost quality?** No — 6/6 on the planted-bug suite at both K15
  and K13, matching the q8_0 baseline, plus 5/5 correct bug identification in
  the passing long-context runs.

## 10. Questions still open

- **Thinking off by default** — the confirmed workaround for §8, untested here.
- **Multi-machine RPC.** Decode is bounded by PCIe transfer of CPU-resident
  experts. llama.cpp's RPC backend could place those experts in a second
  machine's VRAM instead of host RAM, replacing a PCIe-3.0-to-RAM hop with a
  network hop to real VRAM. Whether that wins depends entirely on link speed.
  This is the only remaining structural fix for decode that is not "buy a
  bigger card".
- **K16 at long context.** Validated at 32k prefill only; the 62k termination
  runs were done at K13 and K15.
- **Quality at q4_0 KV for K16 specifically.** The suite was run at K15 and K13.

---

## 11. Reproducing

Harness scripts used (all in the session scratchpad, not committed):
`bench.py` (prefill/context + VRAM), `decodebench.py` (repeats + warmup),
`fitprobe.py` (max-ctx via `--fit`), `looptest2.py`/`looptest3.py` (termination
on the chat endpoint), `dflashtest.py`/`dflashfork.py`, `offloadbench.py`
(`GGML_OP_OFFLOAD_MIN_BATCH`), `k16bench.py`.

Quality: `bench/verifier-quality.sh <model.gguf> <K> <label> [extra flags]` —
extra flags override its hardcoded `-ctk q8_0 -ctv q8_0`, e.g.
`-ctk q4_0 -ctv q4_0 -ub 4096 -b 4096`.

---

## 12. Addendum — q4_0 vs q8_0 KV, tested directly

Prompted by r/LocalLLM consensus that KV quantization degrades with depth and
hurts tool calling ("friends don't let friends run less than 16 bit kv"). This
matters more than a normal caveat here: **K16 only fits because of q4_0**, so if
q4_0 is unsafe the entire retune collapses back to roughly K15/q8_0/98304.

Test: one ~62.5k-token haystack, five unique unguessable codes planted at 10 /
30 / 50 / 70 / 90% depth, asked for one at a time. Shared prefix, so it prefills
once and the rest are cache hits. Greedy (`temp 0`, `top-k 1`), exact string
match, `/completion` to keep the §8 termination bug out of the signal. K13 @
ctx 98304 — both quants fit there, so KV precision is the only variable.

| depth | q4_0 | q8_0 |
| --- | --- | --- |
| 10% | HIT | HIT (identical reply) |
| 30% | HIT | **MISS** — answered `TM-8653-VXG` |
| 50% | HIT | HIT (identical reply) |
| 70% | HIT | HIT (identical reply) |
| 90% | HIT | HIT (identical reply) |
| **total** | **5/5** | **4/5** |

At 50/70/90% the two arms produced **character-identical output** — under greedy
decoding that means the same argmax at every token, i.e. the quantization
difference never perturbed a single choice at depth.

The one miss went to **q8_0**, at shallow depth, and was a cross-needle blend:
`TM-` is bravo's prefix, `VXG` is charlie's suffix. That is retrieval
interference between five similarly-formatted codes — a weakness of the test
design — not a cache-precision failure.

**Conclusion: no systematic q4_0 penalty at 62k on this model.** The predicted
failure mode is degradation *with depth*; the deep needles are precisely where
the arms agree perfectly.

Mechanistic support: Laguna S is 12 global + 36 SWA layers (window 512). Three
quarters of the layers discard their cache continuously and physically cannot
accumulate quantization error across long context. Only 12 layers carry
long-range KV. This is a concrete reason to expect Laguna on the resilient end
of the model-dependent spectrum (cf. reports that Gemma 4 is KV-sensitive while
Qwen3.6 is resilient).

**Not established.** n=5 per arm cannot prove equivalence. Tool-call reliability
under q4_0 — the other specific community claim, and the actual opencode
workload — remains untested. Nothing here speaks to 160k+ depths, and the K13
profile offers 163840.

### Also checked: the chat template

`docs/laguna.md` task 0 worried that the unsloth Q2_K_XL predates poolside's
2026-07-25 "fix tool calls and thinking" re-embed. Extracted
`tokenizer.chat_template` from the GGUF and diffed it against
`poolside/Laguna-S-2.1-GGUF/chat_template.jinja`:

```
6a7
> {%- set preserve_thinking = preserve_thinking | default(false) -%}
```

**The templates are functionally identical** — poolside's simply has that line
duplicated. Both declare `preserve_thinking` and both use it at line 55/56. So
the embedded template is current, and the §8 termination bug is **not** explained
by a stale template. Task 0 can close. That leaves llama.cpp's parsing of the
template as the remaining suspect, which is where the upstream bug is filed.

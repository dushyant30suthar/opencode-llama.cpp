# Tuning — Intel iGPU (Arc 140T / Arrow Lake-H)

The Intel counterpart to [tuning.md](tuning.md). Same rule: nothing here is a
guess. Every number was measured on one machine, and where a number was later
found to be wrong it is corrected in place with the reason, because a retracted
measurement is more useful than a confident one.

The short version: **this hardware is not a slow desktop, it is a different
shape.** Generation is capped by memory bandwidth, prefill is capped by an
upstream software defect, and the one thing it does unusually well — holding
speed at long context — is invisible to every benchmark people normally run.

## Contents

- [The machine](#the-machine)
- [Which backend](#which-backend)
- [Which model](#which-model)
- [The speed ceiling, and where it actually comes from](#the-speed-ceiling-and-where-it-actually-comes-from)
- [The one real strength: flat speed at long context](#the-one-real-strength-flat-speed-at-long-context)
- [Context: the single most important setting](#context-the-single-most-important-setting-on-this-hardware)
- [Dead ends, with reasons](#dead-ends-with-reasons)
- [Memory behaviour](#memory-behaviour)
- [The NPU](#the-npu)
- [Two machines are not one machine](#two-machines-are-not-one-machine)
- [Remaining tests and trials](#remaining-tests-and-trials)
- [Method note](#method-note)

---

## The machine

| | |
| --- | --- |
| GPU | Intel Arc 140T, Arrow Lake-H, Xe-LPG+, 8 Xe cores @ 2.25 GHz |
| Matrix units | **Present.** `GPU_HW_MATMUL` in `OPTIMIZATION_CAPABILITIES`; `DEVICE_GOPS` fp16 36.9 / fp32 4.6 TFLOPS, int8 73.7 TOPS |
| CPU | Core Ultra 7 255H, 16 threads |
| Memory | 32 GB LPDDR5X-8400, **134 GB/s** peak, shared with the CPU |
| GPU-addressable | OpenVINO / SYCL ~28.5 GiB · Vulkan ~23.1 GiB (Mesa caps lower) |
| Kernel driver | `xe` (Mesa warns it is experimental on this platform) |
| NPU | present, `/dev/accel0`, `intel_vpu` |

**On XMX — correct an assumption before it costs you a day.** llama.cpp's Vulkan
backend prints `matrix cores: none` on this GPU, and it is easy to read that as
a hardware fact. It is not. It is a statement about *Mesa's Vulkan path*. The
silicon has matrix engines; the 8× fp16-to-fp32 ratio proves it (vector-only
hardware gives 2×). This distinction explains why OpenVINO's prefill is several
times Vulkan's on identical weights: it reaches the matrix units through Intel's
own kernels, and the Vulkan path cannot.

There is no dedicated VRAM. "GPU memory" is system RAM, so a model competes with
your browser and IDE. Budget ~26–28 GB on a quiet machine, ~22–24 GB in normal
use.

## Which backend

Measured on Qwen2.5-Coder-7B Q4_K_M / int4, clean machine, single stream.

| Stack | Prefill | Generation |
| --- | --- | --- |
| llama.cpp + **Vulkan** (GGUF) | 307 t/s | **19.4 t/s** |
| llama.cpp + OpenVINO backend (GGUF) | **1523 t/s** | 11.8 t/s |
| **OpenVINO GenAI** (IR int4) | TTFT 423 ms | 17.8 t/s |
| llama.cpp + SYCL | 240 t/s | 11.2 t/s |
| CPU only | 201 t/s | — |
| *ipex-llm (archived Jan 2026)* | *279 t/s* | *10.4 t/s* |

And on the model that matters, a 30B-A3B MoE:

| Stack | Prefill | Generation |
| --- | --- | --- |
| llama.cpp + Vulkan (fa **off**) | 250 t/s | 26.2 t/s |
| OpenVINO GenAI (IR int4) | — | 30.3 t/s |
| **OVMS** (same IR, over HTTP) | — | **32.5 t/s** |
| llama.cpp + OpenVINO backend | — | **crashes** (`GGML_ASSERT` in `get_rows`, MoE unsupported) |
| *ipex-llm* | *107 t/s* | *16.6 t/s* |

**Conclusions:**

- **OVMS wins**, and the HTTP layer costs nothing — it beats the in-process
  Python library, presumably from paged attention and prefix caching.
- **SYCL is a dead end.** Upstream SYCL performs about the same as the archived
  ipex-llm it was supposed to replace, and it needs `source setvars.sh` before
  every run, which a directly-spawned server will not inherit.
- **The llama.cpp OpenVINO backend is not the OpenVINO path.** It is a preview
  bolt-on: enormous prefill, poor generation, and it cannot run MoE at all. Its
  weak generation is an integration artifact, *not* the hardware — the native
  path on the same weights is 60% faster. Do not judge OpenVINO by it.
- **Vulkan is still the right pick if you want GGUF**, and it needs no
  environment sourcing at all.

### If you use Vulkan, turn flash attention off

The opposite of the desktop. Measured on the 30B MoE:

| Config | pp2048 @ d0 | tg @ d0 | pp2048 @ d8192 | tg @ d8192 |
| --- | --- | --- | --- | --- |
| **fa off** | 204 | **24.7** | **83** | **16.1** |
| fa on | 194 | 23.6 | 60 | 11.2 |

At 8k context, turning FA *off* buys +38% prefill and +44% generation.
`ubatch-size = 2048` is best for prefill (228 vs 197 at 512).

## Which model

The single most important selection rule on this hardware:

> **Sparse beats small.** A bigger MoE is faster than a smaller dense model,
> because generation cost tracks *bytes read per token*, not parameter count.

| Model | Size | Generation |
| --- | --- | --- |
| Qwen3.6-35B-A3B (MoE, ~3B active) | 19.7 GB | **29.4 t/s** |
| Qwen3-Coder-30B-A3B (MoE) | 16 GB | 32.5 t/s |
| Qwen3.6-27B (**dense**) | 15.7 GB | **6.3 t/s** |

The dense 27B is *smaller on disk* and **4.7× slower**. It streams all ~14 GB of
weights per token; the 35B streams only its active experts.

A second, subtler rule falls out of the overhead analysis below:

> **Prefer fewer layers.** Per-pass overhead scales with layer count — measured
> 0.86 ms/layer on the 64-layer 27B, 0.53 ms/layer on the 40-layer 35B.

Note both Qwen3.6 checkpoints ship as **vision-language exports**
(`Qwen3_5ForConditionalGeneration`). They need `VLMPipeline`, not
`LLMPipeline` — the latter constructs fine and then dies inside `generate()`
with `Port for tensor name input_ids was not found`. This packaging has
consequences well beyond vision; see [dead ends](#dead-ends-with-reasons).

## The speed ceiling, and where it actually comes from

Generation speed is `bytes read per token ÷ memory bandwidth`, until it isn't.
Decomposing measured per-token time against the 134 GB/s floor:

| Model | ms/token | memory bus | **non-bus (compute + overhead)** |
| --- | --- | --- | --- |
| Qwen3.6-27B dense (~14 GB/tok) | 159 ms | 104 ms (66%) | 55 ms (34%) |
| Qwen3.6-35B-A3B MoE (~1.7 GB/tok) | 34 ms | 12.7 ms (37%) | **21.3 ms (63%)** |

**The dense model is genuinely at the wall.** Two independent cross-checks say
we are ahead of comparable systems, not behind:

- An Arc Pro B70 (608 GB/s) does 22 t/s on this model. Scaled to 134 GB/s that
  predicts **4.85 t/s**. We measure 6.28.
- A Qwen3-8B on *this same iGPU* does 12.1 t/s. Scaled by weight ratio that
  predicts **4.1 t/s**. We measure 6.28.

At 56%+ of peak bandwidth the kernels are fine. Best-in-class would be ~72%,
so total remaining headroom is ≤28% and nobody has demonstrated it. **Stop
tuning the dense model.**

**The MoE is a different story, and this corrects an earlier claim in this
repo's history that it was also bandwidth-bound.** Only 37% of its per-token
time is memory. The other 63% is per-pass overhead — which is exactly why every
byte-shrinking flag did nothing for it: they target bytes, and bytes are not the
constraint. This is a known open upstream bug,
[openvino#36270](https://github.com/openvinotoolkit/openvino/issues/36270),
reporting ~78 unfused ops per layer × 40 layers ≈ 3,120 kernel launches per
forward pass — about 21 ms at typical launch cost, matching the measurement
almost exactly.

**So the MoE has roughly 2× waiting on an upstream fusion fix.** We already run
the newest OpenVINO (2026.3.1) and get 29–32 t/s against that issue's ~10 t/s,
so the configuration is not the problem. Re-test when a release ships with MoE
fusion.

### Why prefill is 8× faster than decode on identical weights

Prefill reaches ~230 t/s while decode gets ~29 t/s reading the same weights. The
difference is not bandwidth — it is that prefill puts **many tokens through one
pass**, amortising the fixed per-pass cost, while decode puts one. Anything that
raises tokens-per-pass multiplies throughput. That is the entire theory of
speculative decoding, and on this stack every route to it is closed (below).

## The one real strength: flat speed at long context

Measured on the 35B MoE through OVMS:

| Context | TTFT | Generation |
| --- | --- | --- |
| 586 | 3.9 s | **29.07 t/s** |
| 9,086 | 39.3 s | **27.36 t/s** |
| 19,306 | 85.4 s | **25.30 t/s** |
| 39,406 | 110 s | **0 — KV cache exhausted** |

**Generation falls only ~13% across a 33× increase in context.** A conventional
transformer degrades far more, because KV traffic grows linearly with context.
Qwen3.6 is a **hybrid linear-attention** model: only 16 of 64 layers hold a real
KV cache; the other 48 carry constant-size recurrent state that neither grows
with context nor crosses the memory bus per token.

Prefill, by contrast, scales linearly — 3.9 → 39 → 85 s. So the cost of a large
context is paid **once, at load**, not per token. With OVMS prefix caching a
re-sent 36k-token prompt returned in **0.3 s**.

**Use this.** The workload this machine is unusually good at is *load a large
codebase once, then iterate inside it*. That is the opposite of the "small
context, fast tokens" shape people usually optimise a laptop for.

Prefill rates, measured by streaming TTFT with output verified:

| Prompt tokens | TTFT | Prefill |
| --- | --- | --- |
| 2,075 | 12.8 s | 162 t/s |
| 8,755 | 52.3 s | 167 t/s |
| 17,995 | 117.3 s | 153 t/s |
| 36,795 | 158.8 s | 232 t/s |

## Context: the single most important setting on this hardware

**`cache_interval_multiplier`. Default 8. Set it to 128.** Nothing else in this
document comes close to mattering as much.

Hybrid models — Qwen3.6, Qwen3.5 — checkpoint their **entire fp32 recurrent
state** every `kv_block_size × multiplier` tokens so prefix caching can resume.
At the default of 8 those snapshots dwarf the actual KV cache. Measured on
Qwen3.6-35B-A3B by reading OVMS's own `cache usage: N%` at a fixed 8,434-token
prompt:

| multiplier | KiB/token | usable context @ 4 GB cache |
| --- | --- | --- |
| 4 | 477 | ~9k |
| **8 — the default** | **267** | **~15k** |
| 64 | 47 | ~89k |
| **128** | **33** | **~127k** |

**An 8× reduction from one number.** The floor (~33 KiB/token) lands right in the
range llama.cpp achieves on the same model, so the architecture was never the
problem.

Verified end to end at `cache_size: 4, cache_interval_multiplier: 128`, needle
planted at the end of each prompt to prove the whole thing was read:

| Prompt | Time | Result |
| --- | --- | --- |
| 19,255 tok | 11.5 s | answered, needle found *(returned EMPTY at the default)* |
| 44,215 tok | 19.9 s | answered, needle found |
| 90,615 tok | 371.6 s | answered, needle found |

**It fixes prefill as well as memory.** A 44k prompt at the default writes
~23 GB of snapshots (344 checkpoints × ~67 MiB); at 128 it writes ~1.4 GB. That
is why 44k now prefills in 20 seconds when 19k used to take 85. Earlier
revisions of this document reported prefill at 150–230 tok/s and framed it as a
hardware limit — that was this bug, not the iGPU.

This is [openvino.genai #4050](https://github.com/openvinotoolkit/openvino.genai/pull/4050),
merged 2026-07-01 for **2026.3**; the multiplier shipped in 2026.2.1 as the
stopgap. OVMS documents it: *"Using prefix caching with new Linear Attention
models such as Qwen3.5/Qwen3.6 consumes exceeding amount of memory."*

> **Do not set it below 8.** Values under the default *increase* checkpoint
> frequency: 4 measured 477 KiB/token and 1 failed to serve at all. Both were
> mistaken for "the knob does nothing" here before the default was known.

### 128 is the 35B's number, not a universal one — derive it per model

The table above was measured on the 35B. Copying it to the 27B silently costs
you more than half your context, because the checkpoint size is a property of
the *architecture*, and the two models are not close. Both numbers come
straight out of `config.json`:

```
checkpoint bytes = linear_layers × linear_num_value_heads
                                 × linear_key_head_dim
                                 × linear_value_head_dim × 4        (fp32)
                 + linear_layers × linear_conv_kernel_dim
                                 × (2·k_heads·k_dim + v_heads·v_dim) × 4

KV KiB/token     = full_attn_layers × num_key_value_heads × head_dim × 2 × bytes
                                                          (bytes = 1 at u8)

cache KiB/token  = checkpoint / (kv_block_size × multiplier) + KV
```

| | Qwen3.6-27B | Qwen3.6-35B-A3B |
| --- | --- | --- |
| layers (full attn / linear) | 64 (16 / 48) | 40 (10 / 30) |
| `linear_num_value_heads` | 48 | 32 |
| **one checkpoint** | **151.5 MiB** | **63.8 MiB** |
| **KV per token** | **32.0 KiB** | **10.0 KiB** |

The 27B's checkpoint is **2.4× larger** and its KV is **3.2× larger**, so the
same multiplier buys it far less. Predicted usable context at `cache_size: 4`
(3.9 GiB), with measurements alongside:

| multiplier | 27B KiB/tok | predicted ctx | measured |
| --- | --- | --- | --- |
| 128 | 107.8 | ~38k | 44k answered, 90,617 **EMPTY** |
| 512 | 50.9 | ~80k | 90,617 **EMPTY** (cache peaked 90.9%) |
| 1024 | 41.5 | ~99k | **not yet tested** |
| 2048 | 36.7 | ~111k | not yet tested |

Every measurement so far is consistent with the model: 90,617 exceeds both the
38k and 80k predictions, and failed both times.

**The asymptote matters more than the multiplier.** As the multiplier grows the
checkpoint term vanishes and KV alone sets the floor — 32 KiB/token on the 27B.
At `cache_size: 4` that is a hard ceiling of **~128k tokens no matter how high
you set the multiplier**. Reaching 120k on the 27B therefore needs a multiplier
around 4096 *and* leaves almost no margin; it wants `cache_size: 5`. The same
arithmetic gives the 35B a ~409k ceiling, which is another reason it is the
better long-context model on this box, quite apart from being 4.7× faster.

### Over-context still fails silently

Whatever you set, a request past what the cache holds returns **HTTP 200 with
`finish_reason: "stop"`, empty content and zero completion tokens** — no error,
nothing in the log. Advertise a context you have actually served, and detect the
empty turn client-side: streaming, it arrives as one frame with the `delta` key
**absent** (not `{}`), then `[DONE]`, and `usage` never appears in a stream at
all. Test for a non-empty string after trimming, and count `reasoning_content`
as output or every thinking turn looks like a failure.

## Dead ends, with reasons

Everything below was tried and measured. Each is recorded with *why*, so it is
not retried.

| Lever | Result |
| --- | --- |
| `CACHE_DIR` (compiled-blob cache) | **No gain.** Cold load 24 s, warm load 29 s. Cost 16 GB of disk. Reverted. |
| `DYNAMIC_QUANTIZATION_GROUP_SIZE` | **Disabled (0) is best.** 32 → 117 t/s prefill, 64 → 128, 128 → 199, 0 → **231**. |
| `KV_CACHE_PRECISION: u4` | No speed change (29.57 vs 29.44) **and it altered the output** (1,862 tokens vs 1,250, different text). Downside-only on a hybrid linear-attention model. |
| Draft-model speculative decoding | **Impossible.** OpenVINO asserts *"Speculative decoding is not supported for models with embeddings"*, and every Qwen3.6 OV export is an embedder model. Also **no Qwen3.6 draft model exists** in any format — Qwen published only the 27B and 35B, and older Qwen3 drafts are tokenizer-incompatible (vocab 248320 vs 151936). |
| MTP (the desktop's 76 t/s trick) | **Absent from the export.** `config.json` declares `mtp_num_hidden_layers: 1` but the head is not in the OV graph. Not fixable by re-exporting. |
| `prompt_lookup` | **Broken with reasoning models.** It engages (76.9% acceptance) and emits *correct* tokens, then terminates after ~4. Candidates are drafted from prompt+history; once one contains a stop token it validates and ends generation — and a reasoning model always emits `</think>`. Baseline produced 6,934 chars; prompt lookup produced `"Here's a"`. |
| Concurrency (`max_num_seqs > 1`) | **Actively harmful on MoE.** 1 stream → 25.6 t/s aggregate; 2 streams → **13.2**. Two sequences route to different experts, so each pass must load the *union* of their expert sets. Single-stream is optimal. |
| Smaller int4 group size (64 vs 128) | Intel's docs are explicit: smaller group = larger footprint and **slower** inference. 128 is already the fast setting. |
| MXFP4 / NF4 | CPU-only, and documented as *"not faster than INT8_ASYM"*. Not a GPU speed path. |
| Vulkan cooperative matrix | **3× slower prefill.** llama.cpp gates Xe-LPG+ coopmat to the Windows driver; patching it to allow Mesa does engage `KHR_coopmat` (Mesa advertises an 8×8×16 fp16→fp32 shape) but prefill fell from 285 to 91 t/s. Upstream's restriction is correct, not an oversight. |
| `PERFORMANCE_HINT`, `NUM_STREAMS`, `INFERENCE_PRECISION_HINT` | Already optimal by default. No-ops. |
| `MAX_PROMPT_LEN` | NPU-only. Silently ignored on GPU. |
| `max_num_batched_tokens: 4096` | **Harmful here, despite being Intel's own recommendation.** That guidance assumes a discrete card with dedicated VRAM; on shared memory the larger prefill activation buffers come out of the same 32 GB as the model and cache. Pushed 3.3 GB into swap: a 90k prefill ran 12+ minutes at 98% CPU idle and ~6 MB/s swap-in — not a crash, just thrash. **2048 is the working value**; the OVMS default of 256 is too low. |
| Sparse attention | Skip. It applies to only the 16 full-attention layers of 64, and the kernel is gated to Xe2 — the Arc 140T reports `xe_hpg`. |
| Gemma 4 26B-A4B on GPU | **4 GiB OOM at 28k context.** OVMS 2026.3.0's continuous-batching VLM_CB path (openvino#36737) resets `sliding_window` to 0 via the SDPA→PagedAttention transform, allocating a full 28k-context KV buffer (~4 GiB) that exceeds the Xe driver's 4 GiB−10KB cap. **Fix:** `pipeline_type: VLM` in graph.pbtxt switches to the legacy stateful executor, bypassing the broken path entirely. See [tuning-intel.md#dead-ends] and [REPORT.md](../ovms-audit/REPORT.md). |

### The pattern behind four of these

Draft speculation, MTP, prompt-lookup-via-chat, and prompt-lookup-via-raw-
completions all fail for **one root cause: Qwen3.6 ships only as a
vision-language export.** That single property makes it an *embedder model*
(killing speculation) and a *VLM servable* (OVMS then allows only
`/v3/chat/completions`, `/v3/responses`, `/v3/tokenize` — the raw completions
endpoint returns *"Wrong endpoint. VLM Servable allowed only on…"*, and the chat
template injects the stop tokens that poison prompt lookup).

A **text-only export** would be neither, which would reopen all three at once.
`qwen3_5_text` and `qwen3_5_moe_text` are registered in optimum-intel's
exporter, and the checkpoint carries a separable `text_config`.

> **Correction.** This previously read *"no text-only Qwen3.6 OpenVINO build
> exists publicly."* One has since been produced locally for the 35B —
> `Qwen3.6-35B-A3B-textonly`, 18 GB, `architectures: ["Qwen3_5MoeForCausalLM"]`,
> no vision sub-models. It has **never been served**; see
> [remaining tests](#1-serve-the-text-only-export--highest-value).

The VLM IR also already ships the two pieces that chain into a text-only
pipeline:

```
openvino_text_embeddings_model : input [?,?]            -> inputs_embeds [?,?,N]
openvino_language_model        : inputs_embeds [?,?,N]  -> logits [?,?,vocab]
```

This is the highest-value unexplored lever on this hardware — worth ~2× if it
lands, since it is the only route to amortising per-pass overhead.

## Memory behaviour

**There is no GPU memory leak.** An earlier version of this document said there
was; it was wrong, and the "fix" it recommended (reboot) was wasted effort.
Measured, engine loaded then killed:

| Point | unaccounted | MemAvailable |
| --- | --- | --- |
| baseline | 3.5 G | 24.9 G |
| model loaded | 19.6 G | 9.0 G |
| **t+2s after SIGTERM** | **2.8 G** | **26.2 G** |

Released completely, in about two seconds, returning slightly *better* than
baseline. Repeated start/stop cycles do not accumulate.

**What actually produces a sudden collapse to ~5 tok/s is the model plus its KV
cache exceeding RAM.** Same prompt, same everything else:

| Footprint | MemAvailable after load | Generation |
| --- | --- | --- |
| `cache_size: 6` → 24.6 GiB | 3.0 G | **5.14 tok/s** |
| `cache_size: 4` → ~21 GiB | 8.9 G | **28.36 tok/s** |
| `cache_size: 2` → 19.1 GiB | 9.0 G | 28.52 tok/s |

A 5.5× swing from one config value. On a 30 GiB machine the 19.7 GB MoE wants
`cache_size: 4` at most: 6 leaves ~5 GB for the OS, the desktop (gnome-shell
alone holds ~1.2 GiB of GPU buffers) and your editor, and it spills.

### Diagnosing this correctly

Three traps, all of which produced wrong conclusions here before the numbers
above were measured:

- **Attribute GPU memory with `drm-total-(system|gtt|vram|stolen):`** from
  `/proc/<pid>/fdinfo/*`. Matching bare `drm-total-` also catches
  `drm-total-cycles-*`, which are cycle counters, and produces absurdities like
  "109809 GiB".
- **"Unaccounted" memory is not automatically a leak.** `MemTotal −
  MemAvailable − anon − cached − slab` legitimately includes *live* GPU
  allocations, because GPU buffers never appear in process RSS — OVMS showed
  0.4 GB RSS while holding 19 GiB. A large unaccounted figure with a model
  loaded is expected.
- **Check whether it is actually swapping** before blaming memory at all:
  `pswpin` + `pswpout` deltas from `/proc/vmstat`. A few pages per second is
  idle; ~50k in three seconds is thrashing.

Separately and still true: OpenVINO's Python pipelines expose no teardown —
`LLMPipeline`/`VLMPipeline` have no `close()`, and `release_memory()` lives on
`ov.CompiledModel`, which they never surface. So a Python process must exit to
release its model. That is a one-model-per-process constraint, not a leak, and
OVMS handles it properly.

### Measurement hygiene — one trap produced a fully bogus result

**Never benchmark on the port the plugin uses.** A hand-launched server and the
opencode plugin's server will both take 8100, and the loser dies silently while
the benchmark keeps posting to a URL that now answers from a *different model*.
This happened here: a 90,617-token probe aimed at the 27B was answered by the
35B after the plugin started it mid-run, and was briefly recorded as a
breakthrough. The 35B had always handled 90k. Give manual runs their own port,
or drive everything through the plugin.

Two cheap checks that would have caught it immediately:

- **Confirm which model answered**, not just that something did. `/v3/models`
  returns the served id; log it beside every result.
- **Match the server log to the measurement window.** Each OVMS process writes
  `Mediapipe: <name> state changed to: AVAILABLE` with its pid and timestamp.
  If your server's log has no entries during the window, it was not your server.

Related: **`ps` RSS is the wrong meter on an iGPU.** Weights are allocated
through the DRM path into shared system RAM and never appear in process RSS —
OVMS reads ~1.4 GB while holding 15 GB. Use `MemAvailable`, and use `vmstat`'s
`si`/`so` columns to tell *computing* from *paging*: during healthy GPU work
the CPU sits ~99% idle with the GPU at 2250 MHz, which looks identical to a
stall until you check swap-in.

## The NPU

It works, and it is not the answer.

| Qwen2.5-Coder-1.5B | Generation | Power |
| --- | --- | --- |
| NPU (Intel AI Boost) | 15.8 t/s | ~1–3 W |
| GPU (Arc 140T) | **55.1 t/s** | ~30 W |

The GPU is 3.5× faster on the same model. The NPU's edge is watts, not speed —
it is an always-on background-AI engine, not a coding engine. It is also
stateless-only, Q4_0-primary, fixed prefill chunks, needs static shapes
(`MAX_PROMPT_LEN` / `MIN_RESPONSE_LEN` at pipeline construction) and an
NPU-friendly quantisation (`--sym --group-size -1`; the GPU-oriented group-128
asymmetric IR trips the compiler with *"Found N duplicated names"*).

**Driver pairing is exact.** OpenVINO 2026.2 requires NPU user-mode driver
**v1.33.0**; Fedora packages 1.32.0 (which targets 2026.0) and the mismatch
surfaces as `Unsupported configuration key: NPU_MAX_TILES`. The Ubuntu tarball's
`libze_intel_npu.so.1.33.0` loads fine on Fedora (needs GLIBC 2.38, Fedora has
2.43) via an `LD_LIBRARY_PATH` override — but both the UMD *and* the
driver-compiler must be overridden together, or it silently falls back.

## Two machines are not one machine

Pooling this laptop with the CUDA desktop to run a model that fits neither does
not work, for three independent reasons:

1. **Capacity.** Poolside Laguna S 2.1 is 118B/8B-active, **75 GB at Q4**.
   Pooled GPU memory is 32 (desktop) + ~28 (laptop) = ~60 GB. It still does not
   fit.
2. **Architecture.** llama.cpp RPC is *pipeline-parallel*: layers are split into
   per-machine blocks and each token flows through them sequentially with a
   network hop between. A remote node adds **capacity, never speed**.
3. **The weak node dominates.** Every token would wait on the laptop's slower
   memory plus a TCP round-trip. One published setup fell from 20 t/s to 2 t/s
   purely by moving that link from wired Ethernet to Wi-Fi.

Different engines cannot split a model at all — RPC ships ggml tensor ops, so
both ends must be llama.cpp. llama.cpp-on-one-box plus OVMS-on-the-other is two
independent servers, not a cluster.

**Do instead:** run Laguna XS 2.1 (33B/3B-active, 20.3 GB at Q4) on the desktop
alone, or the full S 2.1 there with `--n-cpu-moe` streaming experts from system
RAM. Leave the laptop as its own machine.

### The model files are not interchangeable either

A recurring source of confusion, worth stating flatly:

| | desktop (RTX) | this laptop (Arc 140T) |
| --- | --- | --- |
| format | **NVFP4** | **OpenVINO IR int4** |
| files | `.safetensors` | `openvino_language_model.xml` + `.bin` |
| why | Blackwell FP4 tensor cores | the Intel GPU plugin implements **only i4/u4** |

Both are 4-bit; neither loads on the other machine. Verified in the laptop's
IR: 992 tensors at `element_type="u4"` (the quantised weight matrices), with
`f16` for embeddings and scales and `u8` zero-points. OpenVINO's GPU plugin has
zero NVFP4 kernels, and `nf4`/`i2`/`u2` have zero kernel hits too — so int4/u4
is not a compromise, it is the only 4-bit path this silicon has.

The `NVFP4 ctx …` commits in this repository's history are **desktop** work.
Model paths do not live under the servable directory; resolve them from
`models_path:` in `~/.local/state/ovms/servables/<name>/graph.pbtxt`.

## Remaining tests and trials

Open work, most valuable first. Each says what to run and what would count as
success, so it can be picked up cold.

### 1. Serve the text-only export — highest value

`~/.lmstudio/models-ov/Qwen3.6-35B-A3B-textonly` **already exists on disk**
(18 GB) and declares `architectures: ["Qwen3_5MoeForCausalLM"]`,
`model_type: qwen3_5_moe_text`, with no `openvino_vision_embeddings_*` files.
An earlier revision of this document said no text-only Qwen3.6 build existed;
that is now wrong — one was produced locally.

**It has never been served.** There is no servable for it under
`~/.local/state/ovms/servables/`, and it appears in no OVMS log.

This is the lever that [the pattern behind four dead ends](#the-pattern-behind-four-of-these)
identifies as reopening everything at once: not an embedder model (speculative
decoding becomes legal), not a VLM servable (`/v3/completions` becomes
available, and prompt lookup escapes the chat template's stop tokens), and
~1.7 GB smaller. Worth ~2× if per-pass overhead can then be amortised.

**Run:** create a servable pointing at it, load it, confirm it comes up as a
plain LLM rather than a VLM servable, then re-run the decode benchmark and
retry `prompt_lookup`. **Success:** `/v3/completions` returns 200 instead of
*"Wrong endpoint. VLM Servable allowed only on…"*.

**Then build the 27B equivalent.** optimum-intel registers `qwen3_5_text` for
the dense variant; the 35B export proves the exporter path works.

### 2. Finish the 27B multiplier sweep — cheap, unfinished

`cache_interval_multiplier` 1024 and 2048 on the 27B, predicted ~99k and ~111k
(table [above](#128-is-the-35bs-number-not-a-universal-one--derive-it-per-model)).
**This was attempted and the result had to be thrown away** — see the port
collision trap below. Currently the 27B's verified ceiling is still 44k.

**Run:** `cache_interval_multiplier: 1024`, `cache_size: 4`,
`max_num_batched_tokens: 2048`, `max_num_seqs: 1`; probe 90,617 tokens with a
needle at the end. Budget ~17 min per probe — a 90k prefill is slow (below).
**Success:** non-empty completion with the needle returned.

### 3. `cache_size: 5` on the 27B, if 120k is the goal

The 32 KiB/token KV floor caps the 27B at ~128k with a 4 GB cache, so 120k has
essentially no margin there. 5 GB moves the ceiling to ~160k. **Risk:** 4 GB
already leaves ~3 GB free during a 90k run, and `cache_size: 6` historically
collapsed throughput to 5 tok/s by spilling. Watch `MemAvailable` and
`vmstat` swap-in, not RSS.

### 4. `PagedCausalConv1D` is reference-only on the GPU

The remaining decode gap on hybrid models. The op has no optimised GPU kernel
and is unchanged in master. Nothing to tune — worth filing upstream with the
per-token numbers from this document, and re-testing when it lands.

### 5. Make `server.ini` actually apply

In the [opencode-localhost](https://github.com/dushyant30suthar/opencode-localhost)
plugin (branch `openvino-backend`), `cache-size`, `cache-interval` and
`context` are parsed and then ignored — `ovms-serve` takes no such flags, and
they live in the servable's `graph.pbtxt`, which the plugin does not write.
**Until this is fixed none of the tuning in this document is reachable from
opencode**; it has to be applied by hand-editing `graph.pbtxt`.

### 6. Parked

- **`KV_CACHE_PRECISION: f16` vs `u8`.** Costs about half the context; the
  quality difference has not been measured and is not cheap to measure.
- **OpenVINO 2026.3.** Two things this document is blocked on land there: MoE
  operator fusion ([openvino#36270](https://github.com/openvinotoolkit/openvino/issues/36270),
  worth ~2× on MoE decode) and the checkpoint fix
  ([genai#4050](https://github.com/openvinotoolkit/openvino.genai/pull/4050),
  which should make `cache_interval_multiplier` unnecessary). Re-measure the
  whole document on release.

### Prefill degrades with prompt length

Not previously recorded. On the 27B, measured with output verified:

| Prompt | Prefill |
| --- | --- |
| short (≤20k) | 158–292 t/s |
| 90,617 tok | **93–97 t/s** |

A 90k prompt takes **~16 minutes** before the first token. Prefix caching makes
every *subsequent* turn nearly free, but the first hit at long context is
expensive enough to shape how the machine is usable. This is the quadratic cost
of the 16 full-attention layers appearing at length; the GPU is pinned at
2250 MHz throughout, so it is not throttling.

## Method note

Two findings in this document did not come from benchmarking, and would not
have.

The inference system was modelled as a graph — stages as actors, each flag as an
operator on the term it transforms — and analysed with
[endiagram](https://endiagram.com)'s structural tools (`structure`, `invariant`,
`live`, `reachable`, `equivalent`). It independently:

- flagged `promptNgrams` as an **isolated uncovered siphon** — a set that, once
  empty, stays empty — predicting prompt lookup's permanent starvation mode
  before it had been attributed;
- placed every speed lever inside a depletable set containing the memory bus,
  i.e. showed the ceiling is *topological*, not a tuning failure;
- and, comparing prefill against decode, returned **Tree vs Disconnected** with
  the entire difference being one node: `spreadOverheadAcrossTokens`. In prefill
  the fixed overhead flows into an amortiser and is divided across the batch. In
  decode it is produced and flows nowhere. That single missing edge *is* the
  8× gap.

That last result is what redirected the investigation from "shrink the bytes"
(where everything was already optimal) to "raise tokens per pass" (where the
remaining 2× actually lives).

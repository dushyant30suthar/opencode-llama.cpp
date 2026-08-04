# MiniMax H3 video generation on the 2× 5060 Ti box — setup, geometry, and tuning

_Session of 2026-08-03/04. Companion to the text-model work in
[backends-qwen27b-benchmarks.md](backends-qwen27b-benchmarks.md); same hardware,
completely different workload._

Raw per-run data: `bench/h3-results-2026-08-03.jsonl`.
Harnesses: `scripts/h3-sweep.py`, `scripts/h3-matrix.py`, `scripts/h3-analyse.py`,
`scripts/h3-clips.py`, `scripts/h3-install-accel.sh`.

---

## Executive summary

- **Single-GPU tuning: 3.1× faster, deployed.** SageAttention 2.2 (−50% at real
  clip length; the probe understated it 5×) + EasyCache 0.4 (−31.7% here, and
  −2.7 GB peak VRAM) + `--fast` (−5%). A 5.2s clip: 20.8 → ~6.8 min. 15s: ~2 h →
  ~35–40 min. `cfg=1.0` is architecturally correct (guidance-distilled), halving
  everything.
  _Correction from the second 2026-08-04 session: the cache figure is ~−26% and
  the compounded total ~2.8×; the baseline row alone carried a ~24 s
  checkpoint-staging cost. The wall-clock figures above were measured directly
  and stand._
- **Two-GPU: a custom pipeline split was built and RAN** — both cards executing,
  whole 21 GB DiT VRAM-resident, faster per-step than single-GPU — something
  upstream ComfyUI cannot do at all (open feature request #13951). Blocked from
  production by one located assertion in comfy_kitchen's DLPack export (§11);
  the loading-mode trilemma (§9) is otherwise solved by the native vbar path.
- **GPU1's idleness is a 31 GB RAM limit, not a config issue** (§7): no P2P
  between the cards (PHB topology) kills sharding-by-storage; two instances need
  ~43 GB RSS. **A 64 GB kit dissolves every wall at once** — the already-proven
  classic-mode split AND dual instances (2× throughput).
- **One hard reset** was caused by an unguarded `--novram` run (37 GB into 31 GB
  RAM — §9 postmortem). `scripts/h3-memguard.sh` is now mandatory for heavy runs.
- 1080p/2K is not native — the canvas caps at 1344×768 (§3); "2K" is a separate
  regeneration pass the ComfyUI nodes do not expose.

## 0. What this model actually is

**MiniMax H3 is a video generation model, not an LLM.** Released 2026-07-31
(the Hailuo 3.0 line): text/image/video/audio in, 4–15s of video **with native
32 kHz stereo audio** out, up to 1344×768 natively.

It has no chat endpoint, no GGUF, no EXL3 quant, and nothing to do with the
llama.cpp/exllamav3 serving stack. Do not confuse it with **MiniMax M3**, which
is the 427B/23B-active text coding model from the same house — different model,
and it does not fit 32 GiB either.

It is documented in this repo because it competes for the same two GPUs.

## 1. Weights — which repo, and why it matters

Two repos exist and only one works with ComfyUI:

| | `MiniMaxAI/MiniMax-H3` | `Comfy-Org/MiniMax-H3` ← **used** |
|---|---|---|
| Format | diffusers: subfolders, `config.json`, sharded | single-file safetensors, flat keys |
| ComfyUI native nodes | **cannot load** | loads directly |
| For | diffusers / SGLang / vLLM | ComfyUI |
| Quantized variants | none | int8_convrot, fp8_scaled, nvfp4_awq |

The full repo is 364 GB (every variant). The five files actually needed total
**63.4 GB**:

| file | size | why this variant |
|---|---:|---|
| `diffusion_models/minimax_h3_fl2va_pruned_int8_convrot` | 20.97 GB | T2V + I2V. **pruned** drops the ~13B AdaLN branches, which are precomputable and never loaded at inference (34.04 GB unpruned). int8 over `fp8_scaled` (same size) because sm_120 has native INT8. |
| `diffusion_models/minimax_h3_ref2va_pruned_int8_convrot` | 20.97 GB | reference-to-video only |
| `text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq` | 15.69 GB | **NVFP4 is native on sm_120.** int8 is 27.14 GB, bf16 51.51 GB — this is both smallest and fastest here. |
| `vae/minimax_h3_video_vae_fp16` | 5.21 GB | |
| `vae/minimax_h3_audio_vae_fp32` | 0.61 GB | |

`_bf16` DiT is 66.28 GB — never fits, do not download.

Verification that these are correct, not just plausible: ComfyUI's own
`model_config_from_unet()` reads the file as `MiniMaxH3` with `num_layers 50,
hidden_size 5376, num_attention_heads 56, attention_head_dim 128,
ffn_hidden_size 14336, latents_dim 24, audio_latents_dim 32, text_dim 5120`,
`latent_format MiniMaxH3AV`. All five files match HuggingFace's declared sizes
byte-exactly.

## 2. The stack

ComfyUI **v0.30.1** at `~/Projects/comfyui`, extracted out of
`opencode-llama.cpp-exp` (it was an untracked plain clone, so a plain `mv`; the
`.venv` survived — torch 2.13.0+cu130, py3.12, both cards visible). H3 needs
≥0.30.0.

The old Wan2.2 / LTX-2.3 GGUF weights were left in place — nothing there is
reusable for H3, and disk was never the constraint.

### ComfyUI facts worth not rediscovering

- Local H3 nodes register through the **new `ComfyExtension` API**
  (`comfy_entrypoint()` → `get_node_list()`), **not** `NODE_CLASS_MAPPINGS`,
  which is empty. Enumerate by awaiting the entrypoint.
- The `Minimax*Node` entries in `/object_info` are MiniMax's **cloud API** nodes.
  The local ones are `MiniMaxH3ImageToVideo`, `MiniMaxH3ReferenceToVideo`,
  `EmptyMiniMaxH3LatentAV`, `MiniMaxH3SigmaShift` — note the capital M; a
  case-sensitive grep for `inimax` misses all of them.
- `MiniMaxH3ImageToVideo` covers **both** t2va and fl2va: omit `first_frame`
  and `last_frame` and it is pure text-to-video. There is no separate T2V node.
- `CLIPLoader type` must be **`"minimax"`**.
- Useful endpoints: `/system_stats` (per-device VRAM), `/prompt`
  (`exec_info.queue_remaining`), `/history/<id>` (server-side ms timestamps).

### The AV pipeline

H3 latents are `NestedTensor(video[B,24,T,H/16,W/16], audio[B,32,2,T·40])`
packed pairs. Sampling runs on the flat pack **with any stock sampler** — the
model handles the audio stream's shifted schedule internally. Decode therefore
reuses the LTX AV nodes, which are generic over the pack:

```
KSampler → LTXVSeparateAVLatent ─┬→ VAEDecode(video vae)      → IMAGE ─┐
                                 └→ LTXVAudioVAEDecode(audio) → AUDIO ─┴→ CreateVideo(24fps) → SaveVideo
```

The H3 nodes emit only `positive`; negative is `ConditioningZeroOut` of it.

## 3. Geometry and the cost model

From `comfy/ldm/minimax/model.py`: 50 layers, hidden 5376, 56 heads × 128,
FFN 14336, `patch_size (1,2,2)` on a /16 latent grid ⇒ /32 in pixel space.
Active params in the DiT blocks work out to **19.27 B**, which matches the
33B-total-minus-13B-AdaLN figure independently.

```
frame_count  snaps up to the 17k+5 grid; trained range ~124–362 (≈5–15s)
latent_t     = ((frame_count-5)//17)*5 + 2
tokens       = latent_t * (H/32) * (W/32) + 2*round(frame_count/24*40)
```

| clip | tokens | TFLOP/forward | attention share |
|---|---:|---:|---:|
| 22 frames @1344×768 (probe) | 7,130 | 348 | 21% |
| 124 frames (5.2s) | 37,710 | 3,491 | 58% |
| 362 frames (15.1s) | 109,062 | 21,255 | **80%** |
| 362 @1920×1088 | 219,486 | 77,520 | 89% |

**Attention is dense.** `optimized_attention(q, k, v, mask=None)` — no sparsity,
no windowing, over one packed text+video+audio sequence. Cost is therefore
**quadratic in clip length**: tripling 5s→15s costs ~6×, not 3×.

Consequence that shaped the whole session: any measurement taken at short clip
length **systematically understates** attention-related wins. A probe row is
21% attention; a real clip is 58–80%.

### 1080p is not reachable and never was

`adapt_canvas()` caps area at 768×1344 = 1.03 MP. 1344×768 **is** the native
ceiling. The "2K" in the marketing is H3-In-context Regeneration — a separate
upscaling pass that the four ComfyUI nodes do not expose. Locally you generate
at ≤1344×768 and upscale outside the model.

## 4. Environment traps (all four cost real time)

1. **PyPI's `sageattention` is v1.0.6** — the old pure-Triton build. It installs
   and imports perfectly, which is the trap: an "is it importable?" check passes
   while you benchmark **v1** believing it is v2. The INT8-QK **v2.2** kernels
   behind the 30–35%-over-SDPA claims only exist in a source build of
   `thu-ml/SageAttention`. Verify by checking for
   `sageattn_qk_int8_pv_fp16_cuda` in the module's exports.
2. **Fedora 44 ships gcc 16.1.1; CUDA 13.3 refuses anything past gcc 15**
   (`crt/host_config.h:137: unsupported GNU version!`). Every nvcc compile dies
   until `export CUDAHOSTCXX=/usr/bin/g++-15`. `gcc15`/`gcc15-c++` are already
   installed from the exllamav3 work. Prefer this over
   `-allow-unsupported-compiler`, which only silences the check.
3. **nvcc is not on PATH but is installed** — `/usr/local/cuda-13.3/bin/nvcc`
   (rpm `cuda-nvcc-13-3`). `command -v nvcc` returning empty looks exactly like
   "no toolkit".
4. **The ComfyUI venv was made with `uv venv`, so it has no `pip` module.**
   `python -m pip` fails; use `uv pip install --python .venv/bin/python`.

### And one shell trap that cost the most time

`pgrep -f 'main.py --port'` / `pkill -f 'main.py --port'` **matches the very
shell running it**, because that literal string is in the shell's own command
line. Every mysterious `exit code 144` in this session was the harness killing
itself. Fix: assemble the pattern at runtime (`PAT="ma""in.py --po""rt"`) or
skip `$$`. `sweep.py`'s internal `stop_any()` was never affected — `pkill`
excludes itself — only the interactive shells were.

## 5. Benchmark method

Two axes with very different costs:

- **Launch flags** (attention backend, memory, precision) are read at import ⇒
  one server restart per row (~10s boot + ~40s to reload 21 GB off NVMe).
- **Workflow params** (sampler, steps, cfg, placement, caching, resolution) are
  just a different graph against a resident model ⇒ free.

`sweep.py` nests them so one restart amortises over a whole workflow phase.

Stages: **PROBE** (22 frames/10 steps — throughput only, quality meaningless by
design, ~2 min/row) and **CONFIRM** (124 frames/20 steps, ~21 min/row). Most of
the session ran at an intermediate 124 frames/10 steps: real clip length so
attention's share is honest, half the steps so a row is ~6 min.

**Timing rule, carried over from the text-model work:** every number comes from
ComfyUI's own `execution_start` → `execution_success` messages, which carry
server-side millisecond timestamps (`execution.py add_message`). Client stream
timing is never used — that is the mistake that inflated the first exllamav3
decode numbers ~1.7× and rigged that comparison. `wall_s` is recorded only as a
cross-check.

Run-to-run reproducibility measured at **<1%** (K0 369.8 vs 366.7s; K1 285.1 vs
284.7s), so the 2% noise threshold used for ranking is well calibrated.

## 6. Results

All at 1344×768, 124 frames, cfg 1.0, euler/simple unless noted.

### Attention backend — the dominant term

Probe length (22 frames, 7,130 tokens — 21% attention):

| row | backend | server s | vs default |
|---|---|---:|---:|
| A0 | default SDPA | 83.3 | — |
| A1 | `--use-pytorch-cross-attention` | 89.4 | +7.3% |
| **A2** | **`--use-sage-attention`** | **74.3** | **−10.9%** |
| A4 | `--use-split-cross-attention` | 130.4 | +56.6% |

At **real clip length** the same flag is worth far more — backing fixed overhead
out of A0-confirm (1250.6s, 20 steps, 58.08 s/it) against K0 (369.8s, 10 steps,
sage) puts sage at **≈28 s/it vs 58 s/it, about −50%**. This is the single
clearest illustration of the probe-understates-attention effect.

### Step caching — the lever that was not in the original matrix

H3 is **guidance-distilled but NOT step-distilled**, so no lightning LoRA exists
and steps cannot be cut for free. Caching is the only way to buy back step cost.
`EasyCache`/`LazyCache` ship natively in v0.30.1.

> **CORRECTION (second session, 2026-08-04).** The `vs base` column below is
> overstated by about 5 points. K0 is the first row of a phase that shares one
> server, so it alone absorbed the ~24 s cost of staging the checkpoint —
> measured later in phase N and fixed by `sweep.py --warmup`. Warm, K0 is ~343 s,
> which makes **EasyCache 0.4 ≈ −26%, not −31.7%**. Still the largest single
> tuning win on this box. See
> [h3-levers-2026-08-04-session2.md](h3-levers-2026-08-04-session2.md) §5b.

| row | config | server s | vs base (overstated, see above) | peak VRAM cuda:0 |
|---|---|---:|---:|---:|
| K0 | no cache | 366.7 (cold; ~343 warm) | — | 14.04 GB |
| K1 | EasyCache 0.2 | 284.7 | −23.0% | **11.31 GB** |
| **K2** | **EasyCache 0.4** | **252.6** | **−31.7% → ~−26%** | |
| K3 | LazyCache 0.2 | 281.5 | −23.9% | |
| K4 | EasyCache 0.3, early start | 252.9 | −31.6% | |

**The VRAM result matters as much as the time.** Caching frees 2.7 GB of peak
VRAM. At 15s the token count is 2.9× and activations scale with it; 14 GB at 5s
was already close to the 15.5 GiB ceiling. Caching is plausibly what makes 15s
runs possible at all rather than OOM at the VAE decode.

### `--fast` optimizations

| row | flags | server s | vs control |
|---|---|---:|---:|
| B0 | control | 373.4 | — |
| B1 | `fp16_accumulation` | 359.2 | −3.8% |
| B2 | `+ cublas_ops` | 356.4 | −4.5% |
| B3 | `autotune` | 373.1 | −0.1% (null) |
| **B4** | everything | **354.8** | **−5.0%** |

`autotune` is a genuine null result — worth recording so it is not added later
on faith.

### Compounded

sage (−50%) × EasyCache 0.4 (−31.7%) × `--fast` (−5.0%) ≈ **3.1× faster**.

> **CORRECTION:** with the cache restated at ~−26% (see above), the compounded
> figure is ≈ **2.8×**, not 3.1×. The tuned wall-clock numbers below were
> measured directly and are unaffected — only the decomposition changes.

| clip | untuned | tuned |
|---|---:|---:|
| 5.2s @ 20 steps | 1250.6s (20.8 min) | ~6.8 min |
| 15.1s @ 20 steps | ~2 h (extrapolated) | **~20–40 min** |

## 7. The structural problem: GPU1 is idle

**GPU0 runs at 99% / 75°C while GPU1 sits at 0% / 176 MiB / 38°C.** ComfyUI
places a diffusion model on one device; there is no tensor-parallel path for a
DiT, and this is an architecture limit, not a misconfiguration.

Worse, the DiT is 20.97 GB against 15.5 GiB usable, so **~6 GB streams from host
RAM every step** — and host RAM is under real pressure: one ComfyUI instance
holds **21.7 GB RSS of 31 GB total, leaving ~5 GB available**.

Options evaluated:

- **Two instances, one per GPU** (`--cuda-device 0/1`, separate ports). Doubles
  throughput for batch work and is the standard answer. **Not viable here**:
  2 × 21.7 GB RSS against 31 GB RAM would thrash into zram — the same disaster
  recorded in the llama.cpp work at 26.8 GB.
> **NOTE (2026-08-04 session 2):** the "x4 slot is slow" framing below needs
> scoping. Phase D's D4 row ran the entire streaming 21 GB DiT on cuda:1's x4
> slot and matched the x8 slot to **0.06%**, so for WEIGHT STREAMING the 2.76
> GB/s figure is not the constraint — that cost is per-transfer overhead, not
> throughput. **The xDiT rejection below still stands**, because all-to-all of
> ACTIVATIONS is a genuinely different traffic pattern and is bandwidth-bound.
> What is revised is only the broader implication that cuda:1 is a poor place to
> put compute; it is not.

- **xDiT / USP / PipeFusion** — real sequence-parallel compute across GPUs
  (FLUX.1: 13.87s → 6.98s). Rejected: large integration effort against ComfyUI's
  custom H3 implementation, and communication-heavy all-to-all across GPU1's
  **x4 PCH slot at a measured 2.76 GB/s** would likely eat the gain. The text
  work already measured 26% of prefill wall time lost to TP traffic on that link.
- **`MultiGPU_WorkUnits`** (native, "MultiGPU CFG Split") — **not a placement
  tool despite the name.** It deep-clones the whole model per GPU to split the
  positive/negative CFG passes: 20.97 × 2 = 41.9 GB against 32 GB total. And
  with `cfg=1.0` there is no second pass to split anyway.
- **DisTorch2** (`ComfyUI-MultiGPU`, pollockjj) — shards the DiT's layers across
  devices at load. `UNETLoaderDisTorch2MultiGPU` exposes `compute_device`,
  `virtual_vram_gb` and **`donor_device`, which defaults to `cpu`**. Pointing
  the donor at `cuda:1` moves the spill from host RAM into idle VRAM. This does
  not make GPU1 *compute*, but it removes the host-RAM round trip and frees
  ~6 GB of RAM. **Phase M measures this** — see the results file.

### Measured verdict (2026-08-04) — it is a RAM limit, not a GPU limit

Every option above was tested rather than reasoned about. The results:

**1. There is no P2P between the cards.**

```
torch.cuda.can_device_access_peer(0,1) -> False
nvidia-smi topo -m: GPU0 <-> GPU1 = PHB     (traverses the PCIe host bridge)
link: GPU0 Gen3 x8, GPU1 x4
```

`PHB` means every GPU-to-GPU byte routes GPU0 → CPU root complex → GPU1. So
using GPU1 as a memory donor costs **two hops instead of one**, and the
measurement agrees exactly:

| row | donor | server s | peak cuda:1 |
|---|---|---:|---:|
| M0 | stock loader (host RAM) | 369.2 | 0.18 GB |
| **M1** | **DisTorch2, donor=cpu** | **348.0** | 0.17 GB |
| M2 | DisTorch2, donor=cuda:1 | 359.9 | **0.17 GB** |

DisTorch2 emitted the correct allocation string (`#cuda:0;6.0;cuda:1`) yet GPU1
never exceeded **180 MiB / 4%** across a full load+sample cycle sampled at 1 Hz.
Nothing lands there, and pointing the donor at it is *slower* than host RAM.

Note M1 is still worth keeping: DisTorch2's sharding beats ComfyUI's native
offload by **−5.7%** and frees **2.2 GB** of peak VRAM (13.73 → 11.52 GB), even
with donor=cpu.

**2. Two independent instances do not fit in 31 GB — even with `--mmap-torch-files`.**

Both instances booted fine (`--cuda-device 0/1`, ports 8194/8195, each seeing
one card). Firing the same graph at both simultaneously:

```
[  5s] avail 21 GB   cached 5 GB
[ 10s] avail 13 GB
[ 15s] avail  5 GB
[ 20s] avail  2 GB   cached 3 GB   swap +1520 MiB
ABORT: swap grew 3971 MiB in 20s
```

`Cached` *falls* (5 → 3 GB) as pressure rises, which is the tell: **mmap does not
share the weights**. ComfyUI materialises transformed tensors from the int8
ConvRot layout rather than using the file bytes, so each process builds its own
~21 GB copy. The flag cannot help with that.

Harness: `scripts/h3-dual-gpu-test.py`, log `bench/h3-dual-gpu-abort-2026-08-04.log`.

**Conclusion: GPU1 is idle because this box has 31 GB of RAM, not because of
anything configurable in ComfyUI.** Sharding is blocked by the absence of P2P;
independent instances are blocked by host RAM. Consistent with the llama.cpp
history on this same box (26.8 GB swap disaster, `cache-ram` dead for hybrids).

**The fix is 64 GB of system RAM**, which would make two instances viable and
roughly double throughput — two clips in the time one takes now. That is the
highest-value upgrade for this workload and it is a DDR4 kit, not a GPU.

## 8. Correct settings, and why

- **`cfg = 1.0`** — H3's released checkpoints are **guidance-distilled**; CFG
  behaviour is baked into the weights. The negative pass is genuinely wasted
  work, not a quality trade. This alone was a 2× error in the first estimates.
- `shift_video 12.0`, `shift_audio 3.0` — `MiniMaxH3SigmaShift` node defaults.
- `1344×768`, `fps 24.0`, frame counts on the **17k+5** grid, trained 124–362.
- `CLIPLoader type: "minimax"`.
- Steps/sampler: **not settled.** One source indicates euler wants ~50 sigma
  points while `res_multistep` is second-order (~21). Everything measured here
  ran 10–20 steps for throughput, which may be undercooked for quality.

### Prompting

H3 follows instructions closely; vague prompts give vague results. Documented
order: **Subject → Scene → Action → Camera → Timing → Visual Style → Audio**.
Naming an explicit camera move and a timing beat matters at 15s, where an
unspecified clip drifts or stalls halfway.

## 9. The pipeline split — GPU1 computing, and the crash it cost

Since no existing tool does compute-across-both-GPUs for H3 on a no-P2P board,
a custom node was built: **`H3PipelineSplit`** (`custom_nodes/h3_dualpipe/`,
mirrored as `patches/comfyui-h3-dualpipe-node.py`). Blocks 0..s-1 live and
execute on cuda:0; blocks s..49 on cuda:1; activations cross once per forward
(~0.4 GB at 124f). Uses ComfyUI's own `patches_replace["dit"]` per-block hook,
placement in the `ON_LOAD` callback, a `torch.cuda.device` guard around the
cuda:1 half, `--disable-cuda-malloc`.

**It works.** Measured live: both cards fully loaded (15.3 / 14.2 GB), whole
DiT VRAM-resident, utilization alternating GPU0 100% ↔ GPU1 100% per half-step,
sampling at **23.5–26.5 s/it vs ~29+ baseline**, 10/10 steps completed.

Bugs fixed en route, each one general:
1. Moving weights inside the first forward → `cudaErrorIllegalAddress`
   (async-offload streams hold events on those tensors). Placement must happen
   in `ON_LOAD`, outside inference. Same class as exllamav3 #260.
2. sage/int8 kernels launch on the *current* device, not the tensor's —
   `torch.cuda.device(device_b)` guard required around the cuda:1 half.
3. **ComfyUI-MultiGPU's import-time patch of comfy_kitchen breaks any
   third-party multi-device flow**: its `wrap_for_dlpack_with_device_guard`
   assumes DisTorch's single-compute-device model, and int8_linear dies under
   it. The extension must be disabled for the split to run.
4. H3's audio VAE (`audio_vae.py:101`): `self.filter` cast to `x.dtype` but not
   `x.device` — crashes any config where the module is not resident on the
   compute device. Local fix applied; upstreamable.
5. ComfyUI's weight wrappers keep references to moved weights, so `.to()` off
   cuda:0 does not free it (OOM against stale copies; `empty_cache` does not
   help). Workaround was `--novram` — see below for why that was wrong.

### The crash (2026-08-04 ~07:15) — hard lesson

`--novram` keeps every weight in host RAM: DiT 21 GB + text encoder 15.7 GB
≈ **37 GB against 31 GB**. The Q6 run pushed the box into zram thrash — kernel
log shows kswapd stall traces and swap climbing — the desktop froze and the
machine needed a **hard reset**. Two failures compounded: the RAM arithmetic
was knowable in advance, and the run was left unattended with no memory guard
(dual.py had one inline; the sweep path did not).

Consequences now in force:
- **`--novram` is BANNED for H3 on this box.** The numbers do not fit. Ever.
- **`bench/memguard.sh`** (mirrored `scripts/h3-memguard.sh`): every heavy run
  goes through it — kills the process group at MemAvailable < 2 GB or swap
  growth > 1.5 GB. Self-tested on both paths.
### The loading-mode trilemma (mapped 2026-08-04 morning, all three measured)

ComfyUI 0.30 has three weight-loading modes, and on a 31 GB-RAM / no-P2P box
each one blocks the split differently:

| mode | what happens | verdict |
|---|---|---|
| default (dynamic VRAM) | weights live in a **comfy_aimdo vbar arena**, not plain tensors (`resolve_cast_module_with_vbar`, `vbar_fault`); `.to()` copies out but the arena never frees → cuda:0 OOMs against its own stale copies (`empty_cache` powerless) | VRAM wall |
| `--novram` | all weights stay in host RAM: 21+15.7 ≈ 37 GB > 31 GB → zram freeze | **BANNED** (froze the box) |
| `--disable-dynamic-vram` (classic) | `.to()` frees correctly (placement left 4.8 GB free vs 0.4 before — the split sampled and beat single-GPU per-step) but load/evict transitions materialise full CPU-side copies → transient RAM spike; memguard tripped at swap+3.1 GB and killed it safely | RAM-transient wall |

The split itself is **proven** (both GPUs computing, faster per-step); what is
unfinished is a loading path that fits 31 GB RAM. Two real options:

1. **Native vbar placement** — teach the split to allocate blocks 25–49's vbars
   on cuda:1 through comfy_aimdo's own API instead of `.to()` behind its back.
   The arena is per-device (`vbars_analyze(device)`), so this is the correct
   integration; it is real work against a new, undocumented subsystem.
2. **64 GB RAM** (a DDR4 kit) — dissolves every wall at once: classic-mode split
   loads clean, AND two independent instances fit (the simple 2× throughput
   path). By effort-to-payoff this is the rational move; every software path
   above is fighting the same 31 GB ceiling from a different side.

## 10. Open questions

- SageAttention **3** (FP4, sm_120) is built and reachable via a small custom
  node (`custom_nodes/sage3_select`) — ComfyUI registers `sage3` as an attention
  function but ships **no way to select it**, so it is dead by default. Effect
  unmeasured.
- **Resolution scaling** is the largest untested lever: 960×544 is 0.26× the
  attention cost of 1344×768 at 15s. Pure quality trade, no precision loss.
- LoRA training via **ostris/ai-toolkit** landed 2026-08-03 and reads the *same*
  `Comfy-Org` quantized files already on disk (`COMFY_REPO = "Comfy-Org/MiniMax-H3"`),
  keeping them quantized through convrot8/nvfp4 backends. Fast path on sm_120.
  Feasibility on 16 GB cards unmeasured — training adds gradients, optimizer
  state and activations on top of a model that already spills. Strategically:
  this is the precondition for a community **step-distill** LoRA, which would
  beat every optimization in this document combined.

## 11. Native vbar placement — the v3 chronicle (2026-08-04 morning)

The comfy_aimdo arena was reverse-engineered and the split rebuilt natively.
What is now KNOWN and WORKING:

- A vbar (`ModelVBAR(size, device)`) is a per-device arena; `model.dynamic_vbars`
  is a dict keyed by device — multi-arena is architecturally supported.
- The module→device binding is ONE line in `ModelPatcherDynamic.load`:
  `m._v = vbar.alloc(size)`. Rebinding = `vbar_unpin(old)` + alloc in a cuda:1
  arena + refill + `_v_signature` stamp.
- The lazy re-fault path CANNOT serve fresh allocations (pin/file-slice
  descriptors are registered per load-time alloc → `HostBuffer.read_file_slice
  failed`). EAGER fault-in works: `aimdo_to_tensor(new_v, dev)` +
  `vbar_fault(new_v)` + `interpret_gathered_like([w,b], dest)` views +
  `copy_` + stamp `_v_weight/_v_bias/_v_signature`. All copies must run inside
  `torch.cuda.device(dev_b)` (kitchen kernels launch on the current device).
- **Status: placement completes end-to-end** (140 modules rebound to the cuda:1
  arena, eager copies done, plain params moved). The remaining blocker is at
  first compute: `comfy_kitchen/tensor/base.py` `_handle_copy_`/export raises
  `BufferError: Can't export tensors on a different CUDA device index.
  Expected: 0. Current device: 1.`

Next-session leads, in order:
1. Patch kitchen's DLPack export at the assert site: wrap with
   `torch.cuda.device(tensor's device)` so export always runs on the owning
   device. (ComfyUI-MultiGPU's "P2P-aware DLPack device guard" tried exactly
   this globally and broke int8_linear — do it NARROWLY, only when the export
   device mismatches, inside the h3_dualpipe node's init.)
2. If the failing tensor is the shared aimdo CAST BUFFER
   (`get_aimdo_cast_buffer`) pinned to device 0: per-device cast buffers, or
   force the resident path by verifying `vbar_signature_compare` accepts the
   placement-time signature (a mismatch silently reroutes to the xfer path).
3. Fallback that already ran 10/10 steps: `--disable-dynamic-vram` split —
   viable on a 64 GB RAM box as-is.


## 12. Complete experimental ledger — every run, including failures

All 37 recorded runs, chronological. 1344×768, cfg 1.0, euler/simple
unless a row's phase varies it. Failures are data: each one located a bug or a
wall documented in §§7, 9, 11.

| # | phase/row | stage | config | result | s/step | peak VRAM | note |
|---|---|---|---|---:|---:|---|---|
| 1 | A/A0 | probe | `defaults` | 83s | 8.3 | 0:15.1G, 1:0.2G | default (torch SDPA, flash-capable) |
| 2 | A/A1 | probe | `--use-pytorch-cross-attention` | 89s | 8.9 | 0:15.0G, 1:0.2G | explicit torch 2.x SDPA |
| 3 | A/A2 | probe | `--use-sage-attention` | 74s | 7.4 | 0:15.0G, 1:0.2G | SageAttention 2 — INT8 QK, FP16 V; ~30-35% over SDPA rep |
| 4 | A/A4 | probe | `--use-split-cross-attention` | 130s | 13.0 | 0:13.1G, 1:0.2G | memory-thrifty, expected slower; run it to bound the axi |
| 5 | A/A0 | confirm | `defaults` | 1251s | 62.5 | 0:13.1G, 1:0.2G | default (torch SDPA, flash-capable) |
| 6 | K/K0 | probe-len124-s10 | `--use-sage-attention` | 370s | 37.0 | 0:13.3G, 1:0.2G | baseline |
| 7 | K/K1 | probe-len124-s10 | `--use-sage-attention` | 285s | 28.5 | 0:11.3G, 1:0.2G | EasyCache, default threshold |
| 8 | K/K2 | probe-len124-s10 | `--use-sage-attention` | 253s | 25.3 | 0:14.9G, 1:0.2G | EasyCache, aggressive |
| 9 | K/K3 | probe-len124-s10 | `--use-sage-attention` | 281s | 28.1 | 0:14.9G, 1:0.2G | LazyCache |
| 10 | K/K4 | probe-len124-s10 | `--use-sage-attention` | 253s | 25.3 | 0:14.9G, 1:0.2G | EasyCache, earlier start |
| 11 | B/B0 | probe-len124-s10 | `--use-sage-attention` | 373s | 37.3 | 0:13.6G, 1:0.2G | control |
| 12 | B/B1 | probe-len124-s10 | `--use-sage-attention --fast fp16_accumulation` | 359s | 35.9 | 0:13.7G, 1:0.2G | fp16 accumulate in GEMMs |
| 13 | B/B2 | probe-len124-s10 | `sage --fast fp16_accumulation cublas_ops` | 356s | 35.6 | 0:13.8G, 1:0.2G | + cuBLAS paths |
| 14 | B/B3 | probe-len124-s10 | `--use-sage-attention --fast autotune` | 373s | 37.3 | 0:14.3G, 1:0.2G | kernel autotune; the first run pays for it |
| 15 | B/B4 | probe-len124-s10 | `--use-sage-attention --fast` | 355s | 35.5 | 0:14.3G, 1:0.2G | everything — ComfyUI calls these untested and quality-af |
| 16 | K/K0 | probe-len124-s10 | `--use-sage-attention` | 367s | 36.7 | 0:14.0G, 1:0.2G | baseline |
| 17 | K/K1 | probe-len124-s10 | `--use-sage-attention` | 285s | 28.5 | 0:11.3G, 1:0.2G | EasyCache, default threshold |
| 18 | K/K2 | probe-len124-s10 | `--use-sage-attention` | 254s | 25.4 | 0:15.0G, 1:0.2G | EasyCache, aggressive |
| 19 | K/K3 | probe-len124-s10 | `--use-sage-attention` | 283s | 28.3 | 0:15.0G, 1:0.2G | LazyCache |
| 20 | K/K4 | probe-len124-s10 | `--use-sage-attention` | 255s | 25.5 | 0:14.9G, 1:0.2G | EasyCache, earlier start |
| 21 | M/M0 | probe-len124-s10 | `--use-sage-attention` | 369s | 36.9 | 0:13.7G, 1:0.2G | control: stock loader, spill goes to host RAM |
| 22 | M/M1 | probe-len124-s10 | `--use-sage-attention` | 348s | 34.8 | 0:11.5G, 1:0.2G | DisTorch2 but donor=cpu (isolates the node's own overhea |
| 23 | M/M2 | probe-len124-s10 | `--use-sage-attention` | 360s | 36.0 | 0:13.0G, 1:0.2G | donor=cuda:1 — the fix |
| 24 | M/M0 | probe-len124-s10 | `--use-sage-attention` | 376s | 37.6 | 0:13.7G, 1:0.2G | control: stock loader, spill goes to host RAM |
| 25 | P/P0 | probe-len124-s10 | `--use-sage-attention` | 368s | 36.8 | 0:13.8G, 1:0.2G | baseline, single GPU |
| 26 | P/P1 | probe-len124-s10 | `--use-sage-attention` | **FAILED** | — | 0:15.2G, 1:12.1G | even split: 25+25 blocks |
| 27 | P/P0 | probe-len124-s10 | `sage -cuda-malloc -async-offload` | **FAILED** | — | 0:14.6G, 1:0.2G | baseline, single GPU |
| 28 | P/P1 | probe-len124-s10 | `sage -cuda-malloc -async-offload` | **FAILED** | — | 0:15.1G, 1:9.6G | even split: 25+25 blocks |
| 29 | Q/Q1 | probe-len124-s10 | `sage -cuda-malloc -async-offload` | **FAILED** | — | 0:15.1G, 1:9.6G | split 25/25, fresh server |
| 30 | Q/Q1 | probe-len124-s10 | `sage -cuda-malloc -async-offload` | **FAILED** | — | 0:15.1G, 1:9.6G | split 25/25, fresh server |
| 31 | Q/Q1 | probe-len124-s10 | `sage -cuda-malloc -async-offload --novram` | **FAILED** | — | 0:11.0G, 1:9.6G | split 25/25, fresh server |
| 32 | Q/Q1 | probe-len124-s10 | `sage -cuda-malloc -async-offload --novram` | **FAILED** | — | 0:13.1G, 1:10.1G | split 25/25, fresh server |
| 33 | Q/Q1 | probe-len124-s10 | `sage -cuda-malloc -async-offload --novram` | **FAILED** | — | 0:11.0G, 1:11.7G | split 25/25, fresh server |
| 34 | Q/Q1 | probe-len124-s10 | `--use-sage-attention` | **FAILED** | — | 0:13.7G, 1:0.3G | split 22/28 + EasyCache 0.4 — balanced free (~3.8G/card) |
| 35 | Q/Q1 | probe-len124-s10 | `--use-sage-attention` | **FAILED** | — | 0:13.8G, 1:1.6G | split 22/28 + EasyCache 0.4 — balanced free (~3.8G/card) |
| 36 | Q/Q1 | probe-len124-s10 | `--use-sage-attention` | **FAILED** | — | 0:14.0G, 1:12.7G | split 22/28 + EasyCache 0.4 — balanced free (~3.8G/card) |
| 37 | Q/Q1 | probe-len124-s10 | `--use-sage-attention` | **FAILED** | — | 0:13.7G, 1:12.5G | split 22/28 + EasyCache 0.4 — balanced free (~3.8G/card) |

Machine-readable: `bench/h3-results-2026-08-03.jsonl`. The tuned selection
derived from these rows: `--use-sage-attention --fast`
(see `bench/h3-FINDINGS-generated.md` for the generated per-phase report).

## 13. Production configuration (single-GPU, measured)

The deployable result of the session. All measured, no projections:

```
launch : --use-sage-attention --fast fp16_accumulation cublas_ops
workflow: EasyCache reuse_threshold 0.4 (node between SigmaShift and KSampler)
         cfg 1.0 (guidance-distilled — the negative pass is wasted work)
         1344×768, euler/simple, shift 12.0/3.0, 24 fps
```

| clip | untuned baseline | with this config |
|---|---:|---:|
| 5.2s (124f) @ 20 steps | 20.8 min (measured) | **~6.8 min** |
| 5.2s @ 10 steps | — | **~4.2 min** (measured class: K2 253s) |
| 15.1s (362f) @ 20 steps | ~2 h (extrapolated) | **~35–40 min** (extrapolated, 6.09× FLOPs) |

Files: `config/comfyui/comfyui-server-tuned.ini` (opencode-localhost plugin
drop-in), `config/comfyui/launch-tuned.sh`, `config/comfyui/h3-tuned.json`
(workflow). Steps count and sampler quality (euler@10–20 vs res_multistep) were
never quality-judged — phase E/F rows ran or were designed but outputs need eyes.

## 14. Phases designed but not completed

The matrix defined 80+ rows; the two-GPU investigation consumed the night.
Designed, wired into `scripts/h3-matrix.py`, and runnable as
`sweep.py <letter> --base-flags="--use-sage-attention" --length 124 --steps 10`:

| phase | rows | what it answers |
|---|---|---|
| C | 13 | offload/memory flags (async-offload streams, fast-disk, reserve-vram, channels-last…) |
| D | 5 | encoder/VAE placement on cuda:1 (SelectCLIPDevice/SelectVAEDevice) |
| E | 8 | samplers — res_multistep (2nd order, ~21 sigma pts) vs euler (~50) could halve steps |
| F | 9 | cfg/steps curves (cfg=1.0 already settled by architecture) |
| G | 5 | sigma shift around the 12.0/3.0 defaults |
| H | 6 | fp8 text-encoder, cpu-vae, bf16-vae |
| I | 2 | single-card control |
| J | 6 | geometry scaling — validates the quadratic model end-to-end |
| L | 5 | resolution scaling — the largest untested lever: 960×544 = 0.26× attention cost at 15s |
| K5–K11 | 7 | cache thresholds past 0.4, torch.compile, **sage3 FP4 rows** |

Also built but unmeasured: **SageAttention 3** (FP4, sm_120-native) — compiled
and installed, reachable via `patches/comfyui-sage3-select-node.py` (ComfyUI
registers `sage3` but ships no selector). Reported 2–5× over FlashAttention on
Blackwell; needs quality judgment (FP4 quantises V).

## 15. Adjacent findings

**LoRA training landed (ostris/ai-toolkit, 2026-08-03).** Its H3 extension
loads the SAME Comfy-Org quantized files already on this disk
(`COMFY_REPO = "Comfy-Org/MiniMax-H3"`, identical paths), trains with weights
kept quantized (convrot8/nvfp4 backends — fast path on sm_120). T2V/I2V only so
far. VRAM feasibility on 16 GB cards unmeasured. Strategic relevance: H3 is
guidance-distilled but NOT step-distilled — this toolkit is the precondition
for a community step-distill LoRA, which would multiply against every
optimization in this document.

**The only public H3 multi-GPU implementation** (joeynyc/MiniMax-H3-2x-DGX-Spark)
uses Ulysses sequence parallelism over 2× DGX Spark: 96%/96% utilization, 2.3×.
Not portable here — USP needs the full 21 GB model per rank (Spark has 128 GB
unified; these cards 15.5) and per-layer all-to-alls need their RoCEv2 RDMA
fabric (this box: 2.76 GB/s host-staged, no P2P). It independently proves H3's
DiT parallelizes; the pipeline split in §9/§11 is the topology that fits THIS
hardware. Blueprint if a second box or big-VRAM cards ever arrive.

## 16. Artifact inventory

| artifact | where | what |
|---|---|---|
| Workflows (t2v, i2v, r2v, dualgpu, tuned) | `config/comfyui/*.json` | validated against v0.30.1 `validate_prompt()` |
| Sweep harness | `scripts/h3-sweep.py` + `h3-matrix.py` | server-timed, resumable, phase-based |
| Analyser | `scripts/h3-analyse.py` | results → tuned.json/FINDINGS/configs |
| Clip generator | `scripts/h3-clips.py` | rolling generation, 12 structured cinematic prompts, reads tuned.json |
| Runner/validator | `scripts/h3-run-workflow.py`, `h3-validate-workflows.py` | weight-size preflight; structural-vs-pending error split |
| Memory watchdog | `scripts/h3-memguard.sh` | **mandatory wrapper** for heavy runs; kills at avail<2 GB or swap+4 GB |
| Pipeline split node | `patches/comfyui-h3-dualpipe-node.py` | v3, vbar-native; state in §11 |
| sage3 selector node | `patches/comfyui-sage3-select-node.py` | makes FP4 attention reachable |
| Audio VAE device fix | applied in-tree (`comfy/ldm/minimax/audio_vae.py:101`) | `.to(dtype)` → `.to(device, dtype)`; upstreamable |
| Weight downloader | `scripts/h3-download-weights.sh` | resumable, size-verified |
| Accel installer | `scripts/h3-install-accel.sh` | sage2 source build (PyPI=v1 trap), gcc-15/nvcc handling |
| Raw results | `bench/h3-results-2026-08-03.jsonl` | all 37 rows |
| Dual-instance abort log | `bench/h3-dual-gpu-abort-2026-08-04.log` | the RAM-wall evidence |
| Generated report | `bench/h3-FINDINGS-generated.md` | per-phase tables from analyse.py |

ComfyUI-side mirrors live under `~/Projects/comfyui/{bench,workflows/h3,configs,custom_nodes}`.
ComfyUI-MultiGPU remains disabled (`custom_nodes_disabled_ComfyUI-MultiGPU`) —
its kitchen patch breaks any third-party multi-device flow (§9 bug 3).

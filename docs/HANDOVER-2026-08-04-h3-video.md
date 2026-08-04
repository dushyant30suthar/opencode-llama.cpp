# HANDOVER — MiniMax H3 video on the 2× RTX 5060 Ti box

_SUPERSEDED by [HANDOVER-2026-08-04b-h3-video.md](HANDOVER-2026-08-04b-h3-video.md) (second session of the same day): resolution measured at 2.4-3.2x, the two-GPU split unblocked, a validated cost model, and the EasyCache figure below corrected from ~31% to ~26%. Kept for its chronology._

_Written 2026-08-04. Self-contained: assume the reader (human or model) has no
session context. Companion narrative with all measurements:
[minimax-h3-video-2026-08-03.md](minimax-h3-video-2026-08-03.md). Raw data:
`bench/h3-results-2026-08-03.jsonl` (37 runs)._

## Where things stand in one paragraph

MiniMax H3 (video+audio generation, 33B DiT) runs on this box under ComfyUI
v0.30.1 at `~/Projects/comfyui`. Single-GPU generation is tuned and deployed:
**3.1× faster than baseline** (5.2s clip: 20.8 → ~6.8 min at 20 steps; ~4.2 min
at 10 steps; 15s clips ~35–40 min extrapolated). A custom two-GPU pipeline
split was built and RAN (both cards computing, whole DiT VRAM-resident, faster
per-step) but is blocked from production by one located assertion; details in
T2. The machine froze once during this work (unguarded `--novram` run) — the
ground rules below exist because of that.

## Ground rules — read before touching anything

1. **Every heavy run goes through the watchdog.** No exceptions:
   `~/Projects/comfyui/bench/memguard.sh -- <command>`. It kills the process
   group at MemAvailable < 2 GB or swap growth > 4 GB. A dead benchmark costs
   nothing; the alternative was a hard reset.
2. **`--novram` is BANNED for H3 on this box.** DiT 21 GB + text encoder
   15.7 GB ≈ 37 GB in host RAM against 31 GB total. It froze the machine on
   2026-08-04 (~07:15, kswapd stall traces in `journalctl -b -1`).
3. **Never `pgrep -f` / `pkill -f` a literal that appears in your own command
   line** (e.g. `'main.py --port'`, `'sweep.py'`) — it matches the shell running
   it and kills you (every mysterious exit-144 in the logs). Assemble patterns
   at runtime: `PAT="ma""in.py --po""rt"`, and skip `$$` in kill loops.
4. **ComfyUI-MultiGPU stays disabled** (renamed
   `custom_nodes_disabled_ComfyUI-MultiGPU`). Its import-time patch of
   comfy_kitchen breaks any third-party multi-device flow (int8_linear illegal
   access).
5. **No AI attribution in commits or PRs to this repo** — standing user
   instruction; author everything as the user.
6. CUDA builds need `CUDAHOSTCXX=/usr/bin/g++-15` (Fedora's gcc 16 is rejected
   by CUDA 13.3) and nvcc at `/usr/local/cuda/bin/nvcc` (not on PATH). The venv
   has no pip: use `uv pip install --python .venv/bin/python`.
7. The user's other production stack (llama.cpp router :9337, TabbyAPI :5000)
   shares these GPUs. Check `nvidia-smi` before taking the cards.

## Did generation time go down? Yes — the numbers

| clip | before (untuned) | now (measured config) | how |
|---|---:|---:|---|
| 5.2s, 20 steps | 20.8 min (measured) | **~6.8 min** | sage2 + EasyCache 0.4 + `--fast` + cfg 1.0 |
| 5.2s, 10 steps | — | **~4.2 min** (measured: 253s row K2) | same |
| 15.1s, 20 steps | ~2 h (extrapolated) | **~35–40 min** (extrapolated, 6.09× FLOPs) | same |

The wins, decomposed (all server-measured): SageAttention 2.2 −50% at real clip
length; EasyCache 0.4 −31.7% (and −2.7 GB peak VRAM — likely what makes 15s
possible at all); `--fast` −5%; `cfg=1.0` halves everything vs cfg>1 because
H3 is guidance-distilled (the negative pass is architecturally wasted work).

**Deployed artifacts** (copy `.ini` to
`~/.config/opencode/providers/comfyui/server.ini` to use from the opencode
panel):
- `config/comfyui/comfyui-server-tuned.ini`
- `config/comfyui/launch-tuned.sh <port>` — direct launch
- `config/comfyui/h3-tuned.json` — the winning workflow
- `~/Projects/comfyui/bench/tuned.json` — machine-readable; `h3-clips.py` reads it

---

# TODOs, in priority order

## T1 — Generate real clips; judge quality knobs by eye  _(effort: low, value: immediate)_

The throughput work is done but NOBODY HAS LOOKED at quality trade-offs yet.
Every benchmark ran the same test prompt at 10–20 steps.

```bash
cd ~/Projects/comfyui
bench/memguard.sh -- .venv/bin/python bench/clips.py --count 4          # 5.2s clips, tuned config
bench/memguard.sh -- .venv/bin/python bench/clips.py --count 1 --length 362   # one 15s clip
```

- `bench/clips.py` cycles 12 structured cinematic prompts (H3's documented
  order: Subject → Scene → Action → Camera → Timing → Style → Audio; vague
  prompts measurably produce vague results). Outputs land in
  `output/video/`, manifest in `bench/clips.jsonl`.
- Then judge: **10 vs 20 vs 30 steps** at fixed seed; **euler vs
  res_multistep** (2nd order, ~21 sigma points vs euler's ~50 — if 10 steps of
  res_multistep ≈ 20 of euler, that's a free 2×); **EasyCache 0.4 vs 0.2 vs
  off** for artifacts. Matrix phases E/F/G are wired for the timing side:
  `bench/sweep.py E F G --base-flags="--use-sage-attention" --length 124 --steps 10`.
- The 15s run also VERIFIES the 35–40 min extrapolation and the VRAM headroom
  claim (watch for OOM at the VAE decode; remedy: `--reserve-vram 2.0`).

## T2 — Finish the two-GPU pipeline split  _(effort: medium, value: the big one)_

Full chronicle in the main doc §9/§11. State: the split (blocks 0..21 on
cuda:0, 22..49 on cuda:1) **works end-to-end at the placement level** under
stock dynamic-VRAM: native rebinding into a per-device `comfy_aimdo` arena, 140
modules, eager fault-in, signatures stamped. It ran 10/10 sampling steps in an
earlier (`--disable-dynamic-vram`) variant and was FASTER per step than single
GPU. One blocker remains at first compute in the vbar-native variant:

```
BufferError: Can't export tensors on a different CUDA device index. Expected: 0. Current device: 1.
  at comfy_kitchen/tensor/base.py (~line 354/463, __torch_dispatch__/_handle_copy_)
```

Node: `~/Projects/comfyui/custom_nodes/h3_dualpipe/__init__.py` (mirrored:
`patches/comfyui-h3-dualpipe-node.py`). Test command (~6 min/row):

```bash
cd ~/Projects/comfyui
bench/memguard.sh -- .venv/bin/python bench/sweep.py Q \
  "--base-flags=--use-sage-attention" --length 124 --steps 10
# server log: bench/sweep-server.log (truncate before runs: `: > bench/sweep-server.log`)
```

Leads, in order:
1. **Patch the export narrowly.** In the node's init, wrap the kitchen export
   site so it runs under `torch.cuda.device(<owning tensor's device>)` ONLY
   when devices mismatch. (ComfyUI-MultiGPU proved the concept but its global
   patch broke int8_linear — be surgical.)
2. **Check the shared cast buffer.** `comfy/ops.py get_cast_buffer` →
   `comfy.model_management.get_aimdo_cast_buffer(offload_stream, device)` may
   hand back a device-0 buffer for a cuda:1 request. If so: per-device buffers.
3. **Verify the eager signature actually matches.** If
   `vbar_signature_compare(forward_sig, stamped_sig)` fails, the resident fast
   path is skipped and the (broken for fresh allocs) transfer path runs. Add a
   one-line log in `cast_modules_with_vbar` to confirm `resident=True` for
   rebound modules.

Success criterion: Q1 completes; both GPUs alternate ~100%; server_s at
124f/10steps ≤ 340s (single-GPU same-config: ~370s). Then re-balance
split_index for VRAM headroom and run the 15s test — full residency should
shine most there.

Expectation management: this buys ~10–15% latency plus VRAM headroom, and is
the foundation for micro-batching (two clips in flight = both GPUs busy
simultaneously ≈ 2× throughput). It does NOT make one clip 2× faster —
sequential dependency forbids that without PipeFusion-style stale-KV patch
pipelining (research-grade; see main doc §7).

## T3 — Buy 64 GB RAM  _(effort: money only, value: dissolves every wall)_

Not a software task, but it obsoletes half of T2's difficulty and is the only
path to simple 2× throughput:
- Two independent ComfyUI instances (one per GPU) become viable — measured to
  need ~43 GB RSS; the abort log is `bench/h3-dual-gpu-abort-2026-08-04.log`.
  Test harness ready: `scripts/h3-dual-gpu-test.py` (ports 8194/8195).
- The `--disable-dynamic-vram` split variant (already ran 10/10 steps) loads
  safely — its only failure was transient load-time RAM spikes.
- ai-toolkit LoRA training feasibility improves.
Board is B360 (DDR4). After install: rerun `h3-dual-gpu-test.py`; success =
2 clips in ~1.1× single-clip wall time.

## T4 — Run the unfinished matrix phases  _(effort: ~2h unattended, value: singles and one big maybe)_

~55 designed rows never ran (the GPU1 investigation took the night). All wired
in `~/Projects/comfyui/bench/matrix.py`; run each as:

```bash
bench/memguard.sh -- .venv/bin/python bench/sweep.py <PHASE> \
  "--base-flags=--use-sage-attention" --length 124 --steps 10
```

Priority within this: **L (resolution scaling)** — 960×544 is 0.26× the
attention cost of 1344×768 at 15s, the largest untested lever (~2.5–3× at 15s
if quality holds; pair with an upscale node for a "draft profile");
**K5–K9** (cache thresholds past 0.4, torch.compile); **C** (13 offload/memory
flags); **D/H/I**; **J** last (validates the quadratic model, expensive).
After any batch: `.venv/bin/python bench/analyse.py` regenerates
tuned.json/FINDINGS/configs automatically.

## T5 — Measure SageAttention 3 (FP4)  _(effort: low, value: possibly large, quality risk)_

Built and installed (`sageattn3`), selector node ready
(`custom_nodes/sage3_select`, mirrored `patches/comfyui-sage3-select-node.py`)
— ComfyUI registers the `sage3` attention function but ships no way to select
it. Rows are wired: K10 (sage3 alone), K11 (sage3 + EasyCache). Reported 2–5×
over FlashAttention on Blackwell, but FP4 quantises V — judge output, not just
the clock. If quality holds, it stacks with everything above.

## T6 — Upstream the fixes  _(effort: low, value: goodwill + un-fork the box)_

1. `comfy/ldm/minimax/audio_vae.py:101` — `self.filter...to(x.dtype)` lacks
   `device=`; crashes any config where the module isn't on the compute device.
   Fix is applied locally; one-line PR to Comfy-Org/ComfyUI. **Note: local tree
   is on the v0.30.1 tag with this uncommitted change — any `git pull` needs
   care.**
2. The kitchen DLPack device assert (T2) — file an issue with the repro.
3. `h3_dualpipe` node — publishable as a community node once T2 lands.
Remember rule 5: no AI attribution.

## T7 — Watch list  _(no action, check weekly)_

- **Step-distill LoRA for H3** — ai-toolkit gained H3 training 2026-08-03
  using the exact Comfy-Org files on this disk. The moment someone publishes a
  lightning/distill LoRA, 10 steps → ~4: bigger than every optimization here
  combined. Watch ostris/ai-toolkit, Civitai, r/StableDiffusion.
- **diffusers H3** merged (PR #14355) → xDiT/xfuser support may follow (only
  matters with better interconnect or a second box — see main doc §15).
- **ComfyUI updates**: v0.30.1 is pinned. Anything touching comfy_aimdo,
  comfy_kitchen, or `nodes_minimax_h3.py` invalidates T2 assumptions and the
  local audio-VAE fix. Re-validate before upgrading.
- Official MiniMax sampler guidance / the H3 technical report (unreleased as of
  2026-08-04): would settle steps/shift questions properly.

## T8 — Housekeeping  _(effort: minutes)_

- The repo files from this session are untracked — review and commit (user
  authors; see rule 5): `docs/`, `scripts/h3-*`, `patches/`, `config/comfyui/`,
  `bench/h3-*`.
- `~/Projects/comfyui` has uncommitted in-tree changes: audio_vae fix + the
  benchmark/bench dirs. Its `.git` points at upstream ComfyUI — do NOT push;
  consider a local branch to protect the fix from `git pull`.
- The r2v (reference-to-video) workflow validates but has never generated —
  needs a real reference image in `input/` and one test run.
- `bench/drive3.sh` chains the full remaining matrix if unattended overnight
  running is wanted again — under memguard, and only with the user's consent
  given the freeze history.

## Quick reference — the environment in 10 lines

```
ComfyUI:   ~/Projects/comfyui  (v0.30.1 tag + local audio_vae fix), .venv: py3.12, torch 2.13.0+cu130
Weights:   ~/Projects/comfyui/models/{diffusion_models,text_encoders,vae}/minimax_h3_* (63.4 GB, byte-verified)
GPUs:      2× RTX 5060 Ti 16 GB (sm_120), PCIe Gen3 x8 + x4, NO P2P (PHB topology, 2.76 GB/s host-staged)
RAM:       31 GB + 48 GB zram — THE binding constraint for everything two-GPU
Bench:     ~/Projects/comfyui/bench/{sweep.py,matrix.py,analyse.py,clips.py,memguard.sh,results.jsonl}
Nodes:     custom_nodes/{h3_dualpipe,sage3_select}; ComfyUI-MultiGPU disabled
Sweep use: bench/memguard.sh -- .venv/bin/python bench/sweep.py <PHASE> "--base-flags=..." --length 124 --steps 10
Timing:    ONLY server-side (execution_start→execution_success in /history); never client stream timing
Ports:     8188 panel-owned; 8191 sweep; 8192 clips; 8194/8195 dual-instance test
Docs:      this file + minimax-h3-video-2026-08-03.md (§ numbers referenced throughout)
```

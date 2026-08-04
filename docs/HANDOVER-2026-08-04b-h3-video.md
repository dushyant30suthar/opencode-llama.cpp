# HANDOVER — MiniMax H3 video, second session of 2026-08-04

_Supersedes [HANDOVER-2026-08-04-h3-video.md](HANDOVER-2026-08-04-h3-video.md).
Self-contained: assume no session context. Measurements and reasoning:
[h3-levers-2026-08-04-session2.md](h3-levers-2026-08-04-session2.md). Raw data:
`~/Projects/comfyui/bench/results.jsonl`._

## READ THIS FIRST (added 20:15)

The single most important thing learned today arrived last: **the per-GPU limit
on clip length × resolution was an undocumented 3.64 GiB SageAttention scratch
buffer, not the 16 GB card.** Processing attention in head groups
(`custom_nodes/sage_chunked`, use **4** groups) removes it — exactly, since heads
are independent — and native 1344×768 goes from a probed 13.0 s to **15.1 s, the
model's own trained maximum**. Measured: 15.083 s rendered in 29.6 min, 0
fallbacks, 0 OOM. Full account in §10 of the companion doc.

Long clips need three things stacked, not one: `--chunk 4`,
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (a *fragmentation* fix — it
does nothing against true exhaustion, which is why an earlier test recorded it
as useless), and `--tiled 32 --lowmem` for the decode. `bench/gen.py` carries all
four flags with the measurements in their help text.

## Where things stand in one paragraph

The tuned single-GPU config is unchanged and still correct
(`--use-sage-attention --fast`, EasyCache 0.4, cfg 1.0, euler/simple). What
changed is what we know. **Resolution is the dominant lever** — 960×544 is 2.4×
faster than native and 832×480 is 3.2×, which turns a 15 s clip from ~35 min
into roughly 9–12 min; it is quality-gated and unjudged. The **two-GPU split now
runs** (both cards computing, no BufferError) but is not yet a latency win. A
**cost model** (`bench/costmodel.py`) predicts every single-GPU row within 2.6 s
and is the right first stop before running anything. Three measurement defects
were found and fixed, one of which revises a published number: the EasyCache win
is ~26%, not ~31%.

## Ground rules — unchanged, read before touching anything

1. **Every heavy run goes through the watchdog:**
   `~/Projects/comfyui/bench/memguard.sh -- <command>`. Its rules require BOTH
   low memory AND rising swap, and that is deliberate — neither variable means
   anything alone on this box. The VAE decode legitimately runs MemAvailable
   down to 0.5-2 GB on every single clip, and zram (compressed RAM) grows by
   gigabytes while memory is plentiful. Five kills, four of them wrong, all came
   from trusting any INSTANTANEOUS reading. Normal operation here produces all
   of them several times a minute: MemAvailable 515-2205 MB and swap +3.4 GB are
   just ComfyUI staging a 15 GB encoder plus a 20 GB DiT into 31 GB of RAM, and
   zram churn makes the swap *rate* oscillate by 2 GB per 8s with 14 GB free. It
   therefore keys on DURATION and TOTAL: floor 150 MB, starvation below 400 MB
   sustained 30s, total swap growth 6 GB. It also type-checks both readings
   before comparing — fork fails under the very pressure it watches for, an
   empty reading becomes `0` in bash arithmetic, and that killed a healthy run
   on a measurement never taken. Eight kills, seven wrong, to get here. Do not
   add a level or rate threshold, and do not remove the type check, without
   re-reading §6 of the companion doc. Its rules key on **swap
   growth**, not on MemAvailable: this workload's VAE decode legitimately runs
   the box down to 0.5-2 GB available on every clip, and three healthy runs were
   killed before that was understood. Do not "tighten" it back to a memory
   threshold without re-reading §6 of the companion doc.
2. **`--novram` is BANNED for H3 on this box.** 21 GB DiT + 15.7 GB encoder
   against 31 GB RAM. It froze the machine on 2026-08-04 and cost a hard reset.
3. **Never `pgrep -f` / `pkill -f` a literal that appears in your own command
   line.** Assemble patterns at runtime: `PAT="ma""in.py --po""rt"`.
4. **ComfyUI-MultiGPU stays disabled** (`custom_nodes_disabled_ComfyUI-MultiGPU`).
5. **No AI attribution in commits or PRs** — standing user instruction; author
   everything as the user.
6. CUDA builds need `CUDAHOSTCXX=/usr/bin/g++-15`; nvcc at `/usr/local/cuda/bin/nvcc`.
   The venv has no pip: `uv pip install --python .venv/bin/python`.
7. The user's production stacks (llama.cpp router :9337, TabbyAPI :5000) share
   these GPUs. Check `nvidia-smi` first.
8. **Work stays on branch `h3-2gpu-v0.30.1`.** Never commit to master. The live
   deployed file is `~/.config/opencode/providers/comfyui/server.ini`; the
   repo's `configs/` is NOT it, and copying across is a deliberate step the user
   takes. Benchmarks use ports 8191/8192/8193, never the panel's 8188.

## Start here: the cost model

```
server_s = 84.0 + real_steps x step(tokens)
step(n)  = 0.54 x 28.3 x (n/37710)^2  +  0.46 x 28.3 x (n/37710)
```

`.venv/bin/python bench/costmodel.py [PHASES]` prints measured vs predicted for
every recorded row. It matches every single-GPU row within 2.6 s, including
sage3's end-to-end time to 0.8 s from a kernel-only microbenchmark. **Predict
before you measure** — if a proposed change cannot beat the model by more than
noise, the row is not worth 6 minutes. When a row *does* miss the model by a lot,
that residual is usually the finding (see §7b of the companion doc: the L rows'
saturating −50 s residual is how the offload tax got measured).

**What the model cannot see: allocation failure.** It happily extrapolates a
kernel's benefit into a regime where the kernel cannot allocate. That is exactly
how sage3's "~15% at 15 s" projection was produced, and it was wrong — sage3
OOMs there and comfy silently falls back to PyTorch. Any projection into a
larger geometry needs the backend confirmed to *run* at that geometry first;
`bench/attn-micro.py --tokens N` does this in about a minute.

Two known systematic effects to subtract before believing a number:

- **~24 s first-row penalty.** The first generation on a fresh server stages the
  checkpoint. `sweep.py` runs a whole workflow phase on one server, so it all
  lands on row 1. Use `sweep.py --warmup` for any phase whose rows are compared
  against each other. Launch phases (A/B/C/H/I) restart per row and are immune.
- **EasyCache skip counts** (10 steps): 0.2→3, 0.3/0.4/0.5→4, 0.6→5. Thresholds
  0.3–0.5 all skip 4, which is why 0.4 is saturated.
- **Three rows per server, maximum.** Host RAM accumulates across prompts inside
  one ComfyUI server and the third execution reliably thrashes on 31 GB — phase
  N was killed at row 3 on both attempts, by correct watchdog fires. `sweep.py`
  shares one server across a whole workflow phase, so rows 3+ are the ones that
  die and **whether a phase's conclusion survives depends on the order its rows
  are listed in**. D, L and `bench/quality.py` now run one server per row; do
  the same for any new multi-row phase whose later rows matter. Launch phases
  (A/B/C/H/I) restart per row already and are immune.

## Levers: what is settled

| lever | verdict |
|---|---|
| **Resolution** | **2.4× at 960×544, 3.2× at 832×480. Quality-gated, unjudged.** |
| EasyCache | Saturated at 0.4. 0.5, full-trajectory, LazyCache all identical. 0.6 buys 10% with quality risk. |
| torch.compile | Impossible. Dynamo cannot trace comfy_kitchen's DLPack kernels (FakeTensor has no data pointer). Do not retry. |
| SageAttention 3 | **Not adoptable on this box.** Alone at 5.2 s it is −8.3% end-to-end. But it **OOMs at 15 s and falls back to PyTorch silently** (2.9× slower than sage2), and paired with the deployed EasyCache it drives the box into swap — killed twice, once at MemAvailable=125 MiB with +3.1 GB swap. The only config it improves is one nobody runs. |
| Attention backend | sage2 is the launch-flag winner; sage3 is per-workflow via `custom_nodes/sage3_select`. |
| `--fast` | Won phase B, already deployed. |
| Offload tax | **Measured: ~5.0 s of the 28.3 s step (18%) is host-RAM streaming.** |

## TODOs, in priority order

### T1 — Judge the quality-gated levers _(low effort, unlocks the biggest win)_

Three levers worth 8%–240% are all blocked on the same look. `bench/quality.py`
renders one prompt at one seed under reference / deployed / cache 0.6 / sage3 /
960×544 / 10-step, on a single server, into `output/video/quality-*.mp4`.

**960×544 is the one that matters.** At 2.4× it is a different production
profile: iterate at draft resolution, re-render keepers native.
`workflows/h3/h3-draft-960x544.json` is ready to use. If quality holds, this is
the largest single improvement available and everything else is a rounding
error next to it.

### T2 — Finish the two-GPU split _(medium effort, now has a target)_

It **runs**: 10/10 steps, both cards alternating (gpu0 47% / gpu1 42% busy), no
BufferError, whole DiT resident across 11.0 + 15.65 GB. Fixed in
`custom_nodes/h3_dualpipe`: skip the vbar prefetcher for split models (it staged
cuda:1 blocks with a cuda:0 device label — the mislabel that caused the
BufferError), move the 112 plain params/buffers the rebinding missed, and stop
keying the activation cache on `id(rope)`.

Two problems remain, and they are different problems. **(a)** At split 22/28 it
is ~20% *slower* per clip: cuda:1 sits at 15.65 GB of 15.5 GiB usable and its
arena thrashes. **(b)** It has never completed a full clip — all three attempts
sampled 10/10 and then died at the VAE decode, the last one on a TRUE watchdog
positive (MemAvailable=704 MiB **and** swap +3577 MiB, sustained). The decode
needs ~5 GB for the video VAE on a cuda:0 already holding 11 GB, and the node's
eager fault-in **pins** the cuda:1 arena permanently, so there is no evictable
headroom anywhere and the pressure lands on a 31 GB host. The split's binding
constraint is host RAM at decode time, not VRAM during sampling.

**Phase R** (cuts at 25/28/31/34, queued in `bench/drive5.sh`) addresses both by
putting fewer blocks on cuda:1; note the direction is counter-intuitive — *more*
blocks on cuda:0, because cuda:0 also carries the embeddings, token refiner and
final layer.

**Do not "fix" this by unpinning the arena.** The pinning is load-bearing:
comfy's lazy re-fault cannot serve a fresh allocation (its pin/file-slice
descriptors were registered against the original load-time allocs, so a re-fault
falls through to `HostBuffer.read_file_slice` and dies). Unpinning converts a
watchdog kill into a hard crash. The real fix is to give the rebound allocations
a working re-fault path first — that is the item worth doing before any further
split tuning.

Success criterion is now quantitative: a balanced split should recover the
measured ~50 s offload tax per 10-step clip at native resolution. If R finds a
cut that does that and nothing better, the split is worth ~15% plus VRAM
headroom; if not, the thesis is wrong and it should be kept only for headroom
and micro-batching.

### T3 — Phase D, device placement — MEASURED, no effect _(closed)_

D0 (all default) 355.8 s vs D1 (encoder+VAE on cuda:1) 353.0 s: **0.8%, noise**.
The placement worked — 4 GB of peak footprint moved off cuda:0 — but cuda:0's
peak FELL (14.31 -> 10.26 GB) rather than rising to use the freed room, so
**ComfyUI's dynamic-VRAM allocator does not expand the DiT into space other
models vacate**. The models are time-multiplexed on cuda:0, not competing, so
there was no contention to relieve. Keep D1 for the 4 GB of headroom it does
provide; do not expect time from it.

The live follow-up is that the freed VRAM is *unclaimed, not unusable*: pair D1
placement with `--reserve-vram 0.5` (phase C's C6) and the allocator may take
it, which is the only route left to the ~50 s offload tax that placement alone
cannot touch. That combination is in no phase — add the row.

### T4 — The 15 s clip has still never been rendered _(queued)_

Every 15 s number is a projection. The model says 34.5 min native / ~9–12 min at
960×544. Queued at 10 steps to halve the cost while still answering the VRAM
question. If the VAE decode OOMs, try `--reserve-vram 2.0` (costs resident
weights) or `sweep.py --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
(may cost nothing — untested, and not a ComfyUI flag, which is why it was never
in the matrix).

### T5 — Remaining matrix phases _(queued: H, C; unrun: E, F, G, I, J)_

**Phase N settled which of these matters.** Differencing the full graph against
a sampler-only one (N0 103.1 s vs N1 29.8 s, same flags) puts **73.3 s of the
~84 s fixed cost in the VAE decode + mux** — N1 still does the model load, the
text encode and a full sampling step in under 30 s. So the load/evict
transitions are almost nothing.

- **H (VAE/encoder precision) is the lever**: `--bf16-vae`,
  `--fp16-intermediates`, `--cpu-vae` act on the term that exists.
- **C (offload/memory flags) acts on a term that barely exists** — run it for
  completeness, not in hope. C12 (`--high-ram`) is excluded as unsafe; see
  `bench/drive5.sh`.
- E/F/G are quality-affecting and belong with T1. J validates geometry scaling,
  which `costmodel.py` now does for free.

The same finding is why **T3 (phase D) is ranked where it is**: the 5 GB VAE is
both the largest block of non-sampling time and the peak of memory pressure, and
it is currently decoding on the card holding the 21 GB DiT while the other card
idles.

### T6 — An nvfp4 DiT _(lead, not attempted)_

Both DiT files on disk are `int8_convrot`. Non-attention work is 46% of a step
and is almost entirely these quantised linears; sm_120 has native nvfp4
tensor-core support, and the text encoder on this disk is already `nvfp4_awq`,
so the publisher does ship nvfp4 builds. This attacks the half of the step no
flag can touch. It is a ~21 GB download and its own quality question — probably
the largest unexplored item after T1.

### T7 — Upstream _(low effort)_

1. `comfy/ldm/minimax/audio_vae.py:101` — `.to(x.dtype)` lacks `device=`.
   One-line PR. **The local tree is v0.30.1 + this uncommitted change.**
2. The kitchen DLPack device assert — file an issue with the `aimdo_to_tensor`
   mislabel repro.
3. `torch.compile` × comfy_kitchen — worth an issue: wrapping the kernels as
   custom ops with fake implementations would make compile viable for every
   quantised Comfy model, not just H3.
4. `h3_dualpipe` and `sage3_select` are publishable community nodes.

### T8 — Housekeeping

- **Nothing from either session is committed.** Review and commit as the user
  (rule 5): `bench/`, `custom_nodes/`, `workflows/h3/`, `configs/`, plus the
  docs in this repo.
- The r2v workflow validates but has never generated a frame.
- `bench/drive5.sh` is the current queue; `bench/chain5.sh` shows the
  wait-then-exec pattern for chaining batches.

## Quick reference

```
ComfyUI:  ~/Projects/comfyui, branch h3-2gpu-v0.30.1 (v0.30.1 + audio_vae fix)
GPUs:     2x RTX 5060 Ti 16 GB (sm_120), Gen3 x8 + x4, NO P2P
RAM:      31 GB + 48 GB zram — the binding constraint for everything two-GPU
Bench:    bench/{sweep,matrix,analyse,costmodel,attn-micro,quality,clips}.py
          bench/{memguard,drive4,drive5,chain5}.sh
Nodes:    custom_nodes/{h3_dualpipe,sage3_select}; ComfyUI-MultiGPU disabled
Sweep:    bench/memguard.sh -- .venv/bin/python bench/sweep.py <PHASE> \
            "--base-flags=--use-sage-attention --fast" --warmup --length 124 --steps 10
          extras: --rows R1,R2  --env K=V  --warmup
Timing:   server-side only (execution_start->execution_success); never client stream
Ports:    8188 panel, 8191 sweep, 8192 clips, 8193 quality
```

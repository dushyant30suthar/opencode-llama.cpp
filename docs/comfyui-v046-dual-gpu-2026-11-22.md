# ComfyUI v0.46.2, w6a8 H3 and one server per GPU — 2026-11-22

Box: 2× RTX 5060 Ti 16 GB (sm_120, no P2P), 31 GB RAM, i5-9400F.
Everything below was measured on this box today unless marked otherwise.

## 1. The recurring OOM was our node, not the hardware

`[H3PipelineSplit] cuda:1 vbar OOM while rebinding block N` (every OOM in the log).

- `h3_dualpipe`'s ON_LOAD callback re-ran `place()` on every model load.
- ComfyUI keeps an existing `_v` on reload (`model_patcher.py`: `not hasattr(m, "_v")`), so
  each reload bump-allocated a second copy of blocks s..49 into the cuda:1 `ModelVBAR`
  (bump-only allocator, never reuses space) → OOM on the 2nd sampler / 2nd prompt.
- Symptoms: success/error alternating; multi-shot prompts dying at shot 2; failing block
  number moving around; the error's "lower split_index" hint was backwards.
- Fix (`patches/h3_dualpipe-idempotent.patch`, node in `patches/comfyui-h3-dualpipe-node.py`):
  re-fault the SAME allocation like `ops.cast_modules_with_vbar`, trim cuda:1's torch pool
  first, hint says "raise split_index". Verified: second run rebinds 0 modules, no OOM.
- Upstream: no fix needed there (the bug was ours). Related open PRs #16174, #16542.

## 2. Upgrade v0.30.1 → v0.46.2

- comfyui branch `h3-v0.46.2`; rollback = `h3-2gpu-v0.30.1` + `.upgrade-backup/venv-freeze-v0.30.1-20261122.txt`.
- comfy-aimdo 0.6.0, comfy-kitchen 0.2.41, frontend 1.59.6. torch 2.13.0+cu130 / sage 2.2.0 /
  triton 3.7.1 deliberately unchanged (torch 2.14 reported up to 4× slower on H3, #16150).
- Local audio-VAE fix dropped: upstream now uses `comfy.ops.cast_to_input`.
- Pinned host memory default fell from 28.8 GB to 15.6 GB. Allocator is torch-native now, so
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` actually applies (it did nothing under
  v0.30.1's cudaMallocAsync).

## 3. Model choices (old int8 DiTs and fp16 VAE deleted)

| | result |
|---|---|
| w6a8 vs int8 DiT (fl2va, 960×544×124, 10 steps) | 9.5 vs 11.2 s/step; visually same |
| ref2va int8 r2v | memguard-killed (host RAM); w6a8 completed, 11.8 s/step |
| int8 vs fp16 video VAE | indistinguishable; 2.6 vs 4.9 GB |
| turbo LoRA `fl2v_turbo_8step_v1.0` @ 8 steps | 122 s vs 278 s (20 steps); clean but skipped the leap/catch beat → drafts only |
| turbo LoRA @ 4 steps | 108 s; artifacts (stray poles, duplicate frisbee) → not usable |
| Qwen-Image 2.1 Turbo int8 | 5.0 s / 1024² image warm, text rendering exact |

Current models: `minimax_h3_{fl2va,ref2va}_pruned_w6a8`, `qwen3vl_32b_minimax_h3_nvfp4_awq`,
`minimax_h3_video_vae_int8_convrot`, `minimax_h3_audio_vae_fp32`, H3 turbo LoRAs,
`qwen_image_2.1_turbo_int8_convrot`, `qwen3vl_8b_int8_convrot`,
`qwen3.5_9b_qwen_image_2.1_pe_t2i.int8_convrot` (optional enhancer), `qwen_image_2.1_vae_bf16`.

## 4. Host RAM is the limit, not VRAM

Trace (w6a8, no --lowmem): the 15 GB text encoder lands as *anonymous* RSS (file RSS stayed
1.9 GB) and stays after encoding (`0 models unloaded`); DiT staging adds ~7 GB → swap.
`--disable-smart-memory` drops the encoder after use. Idle servers still hold ~6.5 GB each
(CUDA ctx, torch, pinned buffers; half of it swapped).

## 5. One server per GPU — ~2× throughput

`config/systemd/comfyui-gpu@.service`: `comfyui-gpu@0` cuda:0 :8188, `comfyui-gpu@1` cuda:1 :8189,
`--fast-disk --disable-smart-memory --fp16-intermediates --enable-cors-header`,
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. The opencode panel watches :8188 via `remote`.

| test | result |
|---|---|
| two w6a8 t2v drafts at once | 2 clips in 179 s, 9.5 s/step each, no swap |
| H3 on one GPU + Qwen on the other | both fine; Qwen 8–10 s/image while H3 runs (5 s alone) |
| Qwen on both | 6 images in 38 s |
| full size at once: t2v gpu0 + r2v (3 refs) gpu1, 1344×768×362, turbo 8 steps | 23 and 25 min, both OK; RAM floor 1.6 GB |

- Full-size r2v first OOMed at step 0 with 1.69 GiB "reserved but unallocated" →
  fragmentation; `expandable_segments` fixed it. `--reserve-vram 2` changed nothing; not set.
- Two full-size jobs is the RAM ceiling. Cheapest relief: stagger starts ~1 min (both 15 GB
  encoders otherwise load together); same DiT on both GPUs shares page cache, fl2va+ref2va don't.
- `H3PipelineSplit` cannot run on these servers (each sees one GPU).

## 6. One dashboard: ComfyUI-Distributed v1.7.1 (commit 09c7a1c)

- Stock ComfyUI has exactly one prompt worker (`main.py`).
- Master :8188, worker `gpu1` = :8189 (`config/comfyui/distributed/gpu_config.json`, lives in the
  node folder, not in git). Workflow needs `DistributedCollector(load_balance=true)`; queue from
  the UI or `POST /distributed/queue` with `enabled_worker_ids: ["gpu1"]`. Plain `/prompt` = master only.
- The browser pre-flights the worker URL itself, so a remote user must tunnel :8189 to local :8189:
  `ssh -N -L 8081:127.0.0.1:8188 -L 8189:127.0.0.1:8189 user@prajna`.
- Measured: 4 Qwen images in 31 s, both cards at 100%.

## 7. Housekeeping

- Deleted: 41 GB diffusers Qwen-Image copy, LTX/Wan files (11 GB; Wan TI2V-5B can't do identity,
  the LTX GGUF was truncated), int8 DiTs (39 GB), fp16 video VAE.
- `~/Videos/Furnstack` (103 GB) moved to `/mnt/sda1/Videos/Furnstack` (NTFS, verified: 194,648
  files, 110,170,421,124 bytes, 65 symlinks); auto-mounted via fstab `nofail`.
- Character consistency = H3 r2v with cropped refs + `--ref-size max`; LoRA training is the next
  lever (comfyui `OPEN-LEVERS.md` #5).

## 8. Newer turbo LoRAs (lightx2v, tested same day)

Same prompts/seeds, 960×544×124, w6a8, int8 VAE, res_multistep/simple, run two at a time.

| run | LoRA | steps | shift_video | time | result |
|---|---|---:|---:|---:|---|
| R0 | none (r2v, 3 refs) | 20 | 12 | 294 s | reference |
| **R1** | **ref2v 8-step v1.1 768p** | 8 | 12 | **119 s** | **≈ R0**: faces, outfits, staging, action |
| R2 | ref2v 8-step v1.1 768p | 8 | 6 | 154 s | close; framing differs |
| T1 | fl2v 8-step v1.2 768p | 8 | 12 | 146 s* | clean, better colour than v1.0, still skips the leap/catch |
| T2 | fl2v 8-step v1.2 768p | 8 | 6 | 109 s | artifacts (duplicate frisbee) |

\*first job on its server, includes model load.

Decisions:
- Keep `shift_video` 12. The reported shift-6 training value never helped and added artifacts.
- r2v: `minimax_h3_ref2v_turbo_8step_v1.1_768p_comfyui_bf16` at 8 steps is the default (2.5× faster, no visible loss).
- t2v/i2v: `minimax_h3_fl2v_turbo_8step_v1.2_768p_comfyui_bf16` at 8 steps for drafts only;
  finals = no LoRA, 20 steps. Don't run a turbo LoRA at 20 steps (distilled for 8; untested, not expected to help).
- Deleted the superseded `fl2v_turbo_8step_v1.0` and `ref2v_turbo_4step_v0.1` (the built-in
  ComfyUI H3 templates still name these; pick the new files there).
- Parked: 4B text encoder + ClipProj (`Merserk/qwen3vl-4b-int4-convrot`, `NicoLab28/ClipProj-MiniMax-H3`),
  2.6 + 0.5 GB instead of 15.7 GB: the biggest RAM relief for dual-GPU, prompt adherence unmeasured.

## 9. Images, styles and the agent-facing catalog (same day, later)

- Qwen-Image 2.1 base (`qwen_image_2.1_int8_convrot`) added for LoRAs; Turbo stays for LoRA-free drafts.
- `renderartist/technically-color-qwen` is for the ORIGINAL Qwen-Image (60 blocks × 3072) and cannot load on 2.1
  (32 blocks × 4096); deleted. No Technicolor LoRA exists for 2.1 (HF + Civitai "Qwen 2.1" searched).
- Qwen 2.1 film LoRAs (Civitai, need an API key for some): `80sfy` (author: 25 steps, cfg 3, negative "blurry"),
  `GrainScape UltraReal` (author: 0.7). Diffusion smooths grain away; visible grain = ffmpeg after generation.
- Edit mode (input image) keeps the subject; for new people use text-to-image with the look described.
- H3 style LoRAs (Civitai, API key): 1980s Fantasy / Horror / SciFi Movies (fl2va), Nostalgic Cinema (`r3tr0sks`),
  Minimax H3 Cinematic V2 (0.85). Fantasy and Nostalgic tested: i2v from a Qwen 80s frame, 704×1280×124, 20 steps,
  ~9 min per clip, both GPUs at once; characters and look hold.
- H3 width must be a multiple of 32 (720 → shape error; 704 works).
- Agent-facing files served by ComfyUI from `input/`: `agent-guide.md`, `model-catalog.md`, `model-catalog.json`
  (`pairs_with` per LoRA, `takes_loras` per model). Copies in `config/comfyui/`.

Test harnesses: `scripts/comfyui-2026-11-22/` (dual.py, mix.py, lora.py, lora2.py, full.py, watch.sh).

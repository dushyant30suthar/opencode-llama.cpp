# Agent guide: prajna ComfyUI (read this first)

Fetch this file: `GET http://localhost:8081/view?filename=agent-guide.md&type=input`
Model choices: `GET http://localhost:8081/view?filename=model-catalog.json&type=input`
(human-readable: `model-catalog.md`). Choose models, LoRAs and settings only from the catalog.

## Servers (already set up, do not install or configure anything)

- ComfyUI v0.46.2, two always-on servers, one per GPU (RTX 5060 Ti 16 GB each):
  - GPU 0 = master: `http://localhost:8081` (server port 8188)
  - GPU 1 = worker `gpu1`: `http://localhost:8189`
- Tunnel needed on your side: `ssh -N -L 8081:127.0.0.1:8188 -L 8189:127.0.0.1:8189 <user>@prajna`
- ComfyUI-Distributed v1.7.1 is installed and configured. No GPU-switching scripts needed.

## Using both GPUs

- Load-balanced (preferred): add `{"class_type": "DistributedCollector", "inputs": {"load_balance": true, "images": [<VAE Decode>, 0]}}`
  before the save node, then `POST http://localhost:8081/distributed/queue` with
  `{"prompt": <API workflow>, "client_id": "agent", "enabled_worker_ids": ["gpu1"]}`. Each job runs on whichever GPU is free.
- Direct: send to 8081 for GPU 0 or 8189 for GPU 1 with the normal `/prompt` API.
- Plain `/prompt` on 8081 only ever uses GPU 0.
- Safe in parallel: Qwen image + H3 video; two Qwen jobs; two H3 jobs. Two full-size (1344×768, 15 s) H3 jobs at once
  works but uses nearly all system RAM, so don't add a third job then.

## Hard rules

- Never use the `H3PipelineSplit` node (each server sees one GPU).
- H3 width and height must be multiples of 32: 720 fails, use 704×1280 (9:16) or 1344×768 (16:9).
- H3 frame counts: 124, 209, 277, 311, 362 (5.2–15.1 s at 24 fps).
- A LoRA may only be used with a model in its `pairs_with` (catalog).
- Qwen-Image 2.1 style LoRAs go on the BASE model; Turbo takes no LoRAs.
- Turbo LoRAs run at 8 steps only. Keep H3 `shift_video` at 12.

## Recipes (all tested on this machine)

**80s/90s film image (Qwen-Image 2.1)**
Base model `qwen_image_2.1_int8_convrot`, euler/simple, 25 steps, cfg 3, negative "blurry",
LoRAs `80sfy_v3_qwen` 1.0 then `grainscape_ultrareal_qwen21` 0.7. Don't put "heavy grain" in the negative.
Fast draft instead: Turbo model, 8 steps, cfg 1, no LoRAs.

**80s/90s film video (H3), consistent with the images**
1. Make the first frame with the image recipe above, at the video size (e.g. 704×1280).
2. Animate it with `minimax_h3_fl2va_pruned_w6a8` (first_frame = that image), 20 steps, res_multistep/simple,
   cfg 1, plus `1980s_Fantasy_Movies_Lora_h3` 1.0 or `Nostalgic_Cinema_h3` 1.0 (prompt starts with `r3tr0sks, 1980s cinematic scene, color film.`).
   A 704×1280×124 clip takes about 9 minutes.
3. Add grain afterwards so stills and video match:
   `ffmpeg -i in.mp4 -vf "noise=c0s=20:c1s=8:c2s=8:allf=t,gblur=sigma=0.45" -c:a copy -c:v libx264 -crf 16 -pix_fmt yuv420p out.mp4`
   (images: same filter; stronger grain: c0s=24).

**Consistent characters across shots (H3 r2v)**
`minimax_h3_ref2va_pruned_w6a8` with up to 9 reference images (address them as `<Picture 1>`...).
Default: `minimax_h3_ref2v_turbo_8step_v1.1_768p_comfyui_bf16` at 8 steps (2.5× faster, same quality).
Use tightly cropped references.

**Fast t2v/i2v drafts**
`minimax_h3_fl2v_turbo_8step_v1.2_768p_comfyui_bf16` at 8 steps. Finals: no turbo, 20 steps (turbo drops action beats).

## Prompting H3

Order: Subject → Scene → Action → Camera → Timing → Style → Audio. Put dialogue in quotes; about 2.5 words per second
(a 5 s clip holds about 12 words). Name a camera move and a timing beat.

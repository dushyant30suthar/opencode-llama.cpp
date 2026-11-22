# Model catalog: prajna ComfyUI (v0.46.2), updated 2026-11-22

Fetch: `GET http://<server>/view?filename=model-catalog.md&type=input`
Servers: GPU 0 = :8188 (master), GPU 1 = :8189. Each sees one GPU.
All settings below were measured on this machine unless marked "author".
Machine-readable version (use this to choose): `GET /view?filename=model-catalog.json&type=input`.
Every LoRA there has `pairs_with`; use a LoRA only with a model in that list.
Qwen-Image 2.1: style LoRAs go on the BASE model (`qwen_image_2.1_int8_convrot`); Turbo takes no LoRAs.

## 1. H3 video (MiniMax H3): text, image or reference to video with audio

Shared parts for every H3 job:
- CLIPLoader `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors`, type `minimax`
- VAE (video) `minimax_h3_video_vae_int8_convrot.safetensors`
- VAE (audio) `minimax_h3_audio_vae_fp32.safetensors`
- Sampler `res_multistep`, scheduler `simple`, cfg 1.0
- MiniMaxH3SigmaShift `shift_video` 12, `shift_audio` 3 (keep; 6 was worse)
- Width and height must be multiples of 32 (720 fails; use 704 or 736). Native 1344×768.
- Frames: 124, 209, 277, 311 or 362 (5.2 s to 15.1 s at 24 fps)
- Never use the `H3PipelineSplit` node.

| file | use for |
|---|---|
| `minimax_h3_fl2va_pruned_w6a8.safetensors` | **fl2va** = First/Last frame to video+audio: plain text-to-video, or animate from a first image, or between first and last images |
| `minimax_h3_ref2va_pruned_w6a8.safetensors` | **ref2va** = reference to video: put the characters from up to 9 reference images into a new scene (consistent characters). Address them as `<Picture 1>`, `<Picture 2>` |

### H3 LoRAs (LoraLoaderModelOnly between UNETLoader and MiniMaxH3SigmaShift)

| file | model | what it does | settings |
|---|---|---|---|
| `minimax_h3_ref2v_turbo_8step_v1.1_768p_comfyui_bf16.safetensors` | ref2va | speed: 8 steps instead of 20 | strength 1.0, **8 steps**. Measured: 119 s vs 294 s, no visible loss. **Default for r2v** |
| `minimax_h3_fl2v_turbo_8step_v1.2_768p_comfyui_bf16.safetensors` | fl2va | speed: 8 steps instead of 20 | strength 1.0, **8 steps**. Drafts only: drops action beats. Finals: no LoRA, 20 steps |
| `1980s_Fantasy_Movies_Lora_h3.safetensors` | fl2va | look of 1980s fantasy films | strength 1.0, 20 steps. Tested: i2v from a Qwen 80s frame keeps characters and the 80s look |
| `1980s_Horror_movies_lora_h3.safetensors` | fl2va | look of 1980s horror films | 20 steps; untested |
| `1980s_SciFi_Movies_Lora_h3.safetensors` | fl2va | look of 1980s sci-fi films | 20 steps; untested |
| `Nostalgic_Cinema_h3.safetensors` | fl2va | 1980s/90s cinematic look | trigger `r3tr0sks` at the start of the prompt, strength 1.0, 20 steps. Tested: very similar to 1980s Fantasy |
| `Minimax_H3_Cinematic_V2_h3.safetensors` | H3 | realistic film-camera look; also reduces limb warping and unwanted cuts | strength 0.85 (author); untested |

Rules: never run a turbo LoRA at 20 steps. Style LoRAs: use 20 steps without turbo until a combination is tested.

H3 prompt order: Subject → Scene → Action → Camera → Timing → Style → Audio. Put dialogue in quotes; about 2.5 words per second.

## 2. Qwen-Image 2.1: images (very good at text in images)

Shared parts:
- CLIPLoader `qwen3vl_8b_int8_convrot.safetensors`, type `qwen_image`
- VAELoader `qwen_image_2.1_vae_bf16.safetensors`
- Encode with `TextEncodeQwenImage21` (prompt, negative_prompt, resolution; optional `images` + `vae` for editing); model through `QwenImage21Cache`
- Native up to 2048×2048; any multiple of 32

| file | settings |
|---|---|
| `qwen_image_2.1_turbo_int8_convrot.safetensors` | **fast drafts**: euler, CFGGuider cfg 1.0, ManualSigmas `1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568, 0` (8 steps). About 5 s per 1024² image |
| `qwen_image_2.1_int8_convrot.safetensors` | **finals and style LoRAs**: euler, simple scheduler, 25 steps, cfg 3, negative "blurry". About 20 s per image. Looks more natural with the 80s LoRAs |
| `qwen3.5_9b_qwen_image_2.1_pe_t2i.int8_convrot.safetensors` | optional prompt enhancer (CLIPLoader type `stable_diffusion` + TextGenerate); rewrites short prompts into long ones |

### Qwen-Image 2.1 LoRAs

| file | what it does | settings |
|---|---|---|
| `80sfy_v3_qwen.safetensors` | 1980s/90s professional film-photo look: warm colour, contrast, softness. Also works as an edit: input image + prompt "80sfy the image." | author: strength 1.0, base model, 25 steps, cfg 3, negative "blurry". May change faces a bit |
| `grainscape_ultrareal_qwen21.safetensors` | built-in analog film grain and texture | author: strength **0.7** |

Tested 80s recipe: base model, 25 steps, cfg 3, 80sfy 1.0 + GrainScape 0.7, negative "blurry". Do NOT put "heavy grain" in the negative. For stronger grain, add it after generation (ffmpeg `noise=c0s=20:c1s=8:c2s=8:allf=t,gblur=sigma=0.45`), which also works on videos.

## 3. Using both GPUs

- Load-balanced: add `DistributedCollector` (load_balance true) after VAE Decode and POST the workflow to `/distributed/queue` on :8188 with `enabled_worker_ids: ["gpu1"]`.
- Or send jobs directly: :8188 = GPU 0, :8189 = GPU 1.
- Fits together: Qwen on one GPU + H3 on the other; two Qwen jobs; two H3 jobs (full size works but uses nearly all RAM).

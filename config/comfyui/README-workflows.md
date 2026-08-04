# MiniMax H3 on 2× RTX 5060 Ti

Everything here was authored against ComfyUI **v0.30.1**'s actual node schemas
(`/object_info`), not from a blog post — Comfy-Org had not published official H3
templates as of `comfyui-workflow-templates` 0.11.28, so these are hand-built.

All four graphs pass ComfyUI's own validator. Run it any time:

```sh
cd ~/Projects/comfyui
PYTHONPATH=. .venv/bin/python workflows/h3/validate.py
```

It calls `execution.validate_prompt()` — the same check the server runs before
queueing — and separates *structural* errors from *file not downloaded yet*, so
it is meaningful to run mid-download.

## The graphs

| File | Task | Checkpoint |
|---|---|---|
| `h3-t2v.json` | text → video+audio | `fl2va` |
| `h3-i2v.json` | first frame → video+audio | `fl2va` |
| `h3-t2v-dualgpu.json` | text → video+audio, **device-pinned** | `fl2va` |
| `h3-r2v.json` | reference images → video+audio | `ref2va` |

`MiniMaxH3ImageToVideo` covers **both** t2va and fl2va — omit `first_frame` and
`last_frame` and it is pure text-to-video. There is no separate T2V node.

The pipeline is worth knowing because it is not obvious: H3 latents are
`NestedTensor(video[B,24,T,H/16,W/16], audio[B,32,2,T·40])` packed pairs, and
sampling runs on the flat pack with **any stock sampler** — the model handles the
audio stream's shifted schedule internally. Decode therefore goes through the LTX
AV nodes, which are generic over the pack:

```
KSampler → LTXVSeparateAVLatent ─┬→ VAEDecode(video vae)      → IMAGE ─┐
                                 └→ LTXVAudioVAEDecode(audio) → AUDIO ─┴→ CreateVideo(24fps) → SaveVideo
```

The H3 nodes emit only `positive`, so negative conditioning is
`ConditioningZeroOut` of it.

## Which one to run on this box

**`h3-t2v-dualgpu.json`.** The DiT is 20.97 GB against 15.5 GiB of usable VRAM
per card, so something spills no matter what. What the device pins buy is *which*
things compete:

- `SelectModelDevice gpu:0` — the DiT owns cuda:0 for the entire sampling loop
- `SelectCLIPDevice gpu:1` — the 15.69 GB Qwen3-VL-32B encoder runs once, on the
  other card, and ComfyUI can evict it before sampling starts
- `SelectVAEDevice gpu:1` — both VAEs decode after sampling, off the hot card

Without the pins, the encoder and the DiT fight over cuda:0 and the DiT spills
further into system RAM.

**`MultiGPU_WorkUnits` is not the node you want here.** Despite the name it is a
*CFG split*: it deep-clones the whole model onto each GPU to run the positive and
negative passes in parallel. That needs 20.97 GB × 2 = 41.9 GB and this box has
32 GB total. It is for people whose model already fits on one card.

## Numbers that are the model's, and numbers that are guesses

Authoritative (they are the node authors' own defaults, or hard constraints):

- `shift_video 12.0`, `shift_audio 3.0` — `MiniMaxH3SigmaShift` defaults
- `1344×768` — the native canvas; `adapt_canvas()` caps area at 768×1344 px
- `length 124` ≈ 5.2s at 24 fps. Frame counts snap to a **17k+5** grid, and the
  tooltip states the trained range is **~124–362** (≈5–15s). Shorter is untested.
- `fps 24.0` — H3's native rate
- `CLIPLoader type: "minimax"` — the only string that selects the H3 encoder

**Starting points, not published values** — MiniMax and Comfy-Org have not
released recommended sampler settings, so tune these first:

- `steps 30`, `cfg 5.0`, `euler` + `simple`

## Running one

Start ComfyUI from the opencode panel (or `main.py --port 8188`), then:

```sh
.venv/bin/python workflows/h3/run.py h3-t2v-dualgpu.json
```

It refuses to start until every weight file is present at ≥99% of expected size,
so it cannot half-load a truncated download. Add `--wait-for-weights` to block
until `download-h3.sh` finishes and fire automatically. It reports wall-clock,
seconds per step, peak VRAM per card, and the output path.

## The performance caveat

Unmeasured, and the honest expectation is *slow*. Whatever spills off cuda:0
crosses PCIe every denoising step, and GPU1 sits on this board's x4 PCH slot at a
measured **2.76 GB/s** — the same wall documented in the llama.cpp prefill work.
The first real run is what settles it. If it OOMs, it will be at the VAE decode
after sampling completes: reach for `args = --reserve-vram 2.0` in
`~/.config/opencode/providers/comfyui/server.ini`.

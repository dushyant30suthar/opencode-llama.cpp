# ComfyUI H3 patches & custom nodes — how to reproduce the setup

Target: ComfyUI **v0.30.1** (`git checkout v0.30.1` in the ComfyUI clone).
Anything newer: re-validate first — these touch `comfy_aimdo`/`comfy_kitchen`
interactions and the H3 nodes, all fast-moving (see
`docs/HANDOVER-2026-08-04-h3-video.md` rules and watch list).

| file | what | apply |
|---|---|---|
| `comfyui-v0.30.1-local-fixes.patch` | the in-tree fix: H3 audio VAE `self.filter` cast to dtype but not device (`audio_vae.py:101`) — crashes any config where the module isn't resident on the compute device. Upstreamable one-liner. | `git -C ~/Projects/comfyui apply <this file>` |
| `comfyui-h3-dualpipe-node.py` | **H3PipelineSplit** — the two-GPU pipeline split (blocks 0..s-1 on cuda:0, s..49 on cuda:1), v3 vbar-native. Working through placement; one DLPack assert from done (handover T2). | `mkdir -p custom_nodes/h3_dualpipe && cp <this> custom_nodes/h3_dualpipe/__init__.py` |
| `comfyui-sage3-select-node.py` | **SageAttention3Select** — makes the FP4 `sage3` attention function reachable (ComfyUI registers it but ships no selector). Needs `sageattn3` built (`scripts/h3-install-accel.sh`). | `mkdir -p custom_nodes/sage3_select && cp <this> custom_nodes/sage3_select/__init__.py` |

Also required for the split: ComfyUI-MultiGPU must be **absent or renamed**
(its import-time kitchen patch breaks multi-device int8_linear); the phase-D
placement nodes (`SelectModelDevice` etc.) are native ComfyUI and unaffected.

Everything else needed to rebuild the environment from zero, in order:
`scripts/h3-download-weights.sh` → `scripts/h3-install-accel.sh` →
`config/comfyui/comfyui-server-tuned.ini` (or `launch-tuned.sh`) →
workflows in `config/comfyui/*.json`. Benchmarks: `scripts/h3-sweep.py`
(always under `scripts/h3-memguard.sh`).

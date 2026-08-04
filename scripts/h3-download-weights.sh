#!/usr/bin/env bash
# MiniMax H3 weights for ComfyUI (Comfy-Org repackage, 63.4 GB).
#
# The "pruned" DiT variants are the ones to take: H3's 33B has ~13B in AdaLN
# branches whose modulation outputs are precomputable, so inference never loads
# them. int8_convrot is the smallest that keeps quality; fp8_scaled is the same
# size and only exists for cards without int8 rotation support.
#
# The text encoder is Qwen3-VL-32B at NVFP4 — native on sm_120 (Blackwell),
# so the 5060 Tis run it at full rate rather than emulating.
#
# Resumable: re-run after an interrupt and curl -C - picks up where it stopped.
set -euo pipefail

cd "$(dirname "$0")"
BASE=https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main
mkdir -p models/diffusion_models models/text_encoders models/vae

# $1 dest, $2 path under the repo, $3 expected bytes (0 = unknown)
dl() {
  local dest=$1 src=$2 want=$3
  if [[ -f "$dest" && $want -gt 0 ]]; then
    local have
    have=$(stat -c%s "$dest")
    if [[ $have -eq $want ]]; then echo "SKIP $(basename "$dest") (complete)"; return 0; fi
  fi
  echo "GET  $(basename "$dest")"
  curl -fL --retry 5 --retry-delay 5 -C - -o "$dest" "$BASE/$src" \
    && echo "OK   $(basename "$dest")" \
    || { echo "FAIL $(basename "$dest")"; return 1; }
}

# Three at a time: network-bound, but five concurrent streams on one nvme
# starts costing more in seek than it wins in bandwidth.
dl models/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors \
   diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors 20971520000 &
dl models/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors \
   text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors 15690000000 &
dl models/vae/minimax_h3_video_vae_fp16.safetensors \
   vae/minimax_h3_video_vae_fp16.safetensors 5210000000 &
wait

dl models/diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors \
   diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors 20970000000 &
dl models/vae/minimax_h3_audio_vae_fp32.safetensors \
   vae/minimax_h3_audio_vae_fp32.safetensors 610000000 &
wait

echo "=== H3 weights ==="
ls -lh models/diffusion_models models/text_encoders models/vae
echo H3_DOWNLOAD_DONE

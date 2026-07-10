#!/usr/bin/env bash
set -u
DIR=~/.lmstudio/models/unsloth/Qwen3.6-27B-MTP-GGUF
LOG=~/.local/state/llamastack/downloads.log
mkdir -p "$DIR"
for attempt in $(seq 1 100); do
  curl -sL --fail -C - -o "$DIR/Qwen3.6-27B-UD-Q4_K_XL.gguf" \
    "https://huggingface.co/unsloth/Qwen3.6-27B-MTP-GGUF/resolve/main/Qwen3.6-27B-UD-Q4_K_XL.gguf" \
    && { echo "$(date +%T) DONE Qwen3.6-27B-UD-Q4_K_XL.gguf" >>"$LOG"; exit 0; }
  echo "$(date +%T) retry $attempt MTP download" >>"$LOG"; sleep 10
done
echo "$(date +%T) FAILED MTP download" >>"$LOG"

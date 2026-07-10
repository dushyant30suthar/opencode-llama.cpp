#!/usr/bin/env bash
# Self-resuming model downloader. Safe to re-run; resumes partial files.
set -u
LOG=~/.local/state/llamastack/downloads.log
DIR=~/.lmstudio/models/lmstudio-community
dl() { # dl <subdir> <filename> <url>
  mkdir -p "$DIR/$1"
  local out="$DIR/$1/$2"
  for attempt in $(seq 1 50); do
    curl -sL --fail -C - -o "$out" "$3" && { echo "$(date +%T) DONE $2" >>"$LOG"; return 0; }
    echo "$(date +%T) retry $attempt for $2" >>"$LOG"; sleep 5
  done
  echo "$(date +%T) FAILED $2" >>"$LOG"; return 1
}
dl GLM-4.7-Flash-GGUF GLM-4.7-Flash-Q4_K_M.gguf \
   "https://huggingface.co/lmstudio-community/GLM-4.7-Flash-GGUF/resolve/main/GLM-4.7-Flash-Q4_K_M.gguf"
dl Qwen3-Coder-30B-A3B-Instruct-GGUF Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf \
   "https://huggingface.co/lmstudio-community/Qwen3-Coder-30B-A3B-Instruct-GGUF/resolve/main/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf"
echo "$(date +%T) ALL DOWNLOADS COMPLETE" >>"$LOG"

#!/bin/bash
# Launch vLLM against a config YAML, with the nvcc host compiler pinned.
#
# WHY CUDAHOSTCXX: FlashInfer JIT-compiles kernels at runtime with nvcc. Fedora
# 44's default gcc is 16.1.1; CUDA 13.3 hard-refuses __GNUC__ > 15 in
# crt/host_config.h, so the ninja build fails and engine init dies with
# "Ninja build failed" -> "Engine core initialization failed". gcc15-c++ is
# already installed here (15.3.1) and nvcc accepts it. This is an environment
# requirement of this box, not a vLLM setting, which is why it lives in a
# launcher rather than in the model YAML.
set -u

MODEL="${1:?usage: vllm-launch.sh <model-path-or-repo> <config.yaml> [logfile]}"
CONFIG="${2:?missing config yaml}"
LOG="${3:-$HOME/Projects/vllm/server.log}"

export CUDAHOSTCXX=/usr/bin/g++-15
export CUDACXX=/usr/local/cuda/bin/nvcc
export NVCC_PREPEND_FLAGS="-ccbin /usr/bin/g++-15"
export TORCH_CUDA_ARCH_LIST="12.0"

mkdir -p "$(dirname "$LOG")"
exec "$HOME/Projects/vllm/venv/bin/vllm" serve "$MODEL" \
    --config "$CONFIG" \
    --host 0.0.0.0 --port 8000 >> "$LOG" 2>&1

#!/usr/bin/env bash
# Build llama.cpp with native CUDA for the local GPU.
#
# Rules this script enforces (learned the expensive way):
#   * CUDA 13.2/13.2.x is FORBIDDEN — nvcc miscompiles the quantization
#     kernels and models emit gibberish (llama.cpp #21255). 13.3+ only.
#   * Clean build dir every time — GGML_LTO caches stale objects across
#     compiler/flag changes and produces broken binaries.
#   * nvcc supports host GCC only up to a point (CUDA 13.3 -> GCC 15);
#     if a versioned g++-15 exists it is used as the CUDA host compiler.
#
# Usage:
#   ./scripts/build-llama.sh [path-to-llama.cpp]   # default: ./llama.cpp submodule
# Env:
#   CUDA_HOME   toolkit to use (default: newest /usr/local/cuda-*)
#   CUDA_ARCH   CMAKE_CUDA_ARCHITECTURES (default: native)
set -euo pipefail

LLAMA_DIR="${1:-$(cd "$(dirname "$0")/.." && pwd)/llama.cpp}"
[[ -f "$LLAMA_DIR/CMakeLists.txt" ]] || { echo "error: no llama.cpp checkout at $LLAMA_DIR (run: git submodule update --init llama.cpp)"; exit 1; }

CUDA_HOME="${CUDA_HOME:-$(ls -d /usr/local/cuda-*/ 2>/dev/null | sort -V | tail -1)}"
CUDA_HOME="${CUDA_HOME%/}"
[[ -x "$CUDA_HOME/bin/nvcc" ]] || { echo "error: no nvcc under '$CUDA_HOME' — install the CUDA toolkit (see docs/setup.md)"; exit 1; }

CUDA_VER="$("$CUDA_HOME/bin/nvcc" --version | grep -oP 'release \K[0-9]+\.[0-9]+')"
case "$CUDA_VER" in
  13.2) echo "error: CUDA 13.2 miscompiles quant kernels (llama.cpp #21255) — use 13.3+"; exit 1 ;;
esac
echo "using CUDA $CUDA_VER at $CUDA_HOME"

HOST_CXX_FLAG=()
if command -v g++-15 >/dev/null; then
  HOST_CXX_FLAG=(-DCMAKE_CUDA_HOST_COMPILER="$(command -v g++-15)")
  echo "using g++-15 as CUDA host compiler"
fi

GEN_FLAG=()
command -v ninja >/dev/null && GEN_FLAG=(-G Ninja)

cd "$LLAMA_DIR"
rm -rf build
cmake -B build "${GEN_FLAG[@]}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc" \
  -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH:-native}" \
  -DGGML_CUDA_FA_ALL_QUANTS=ON \
  -DGGML_CUDA_GRAPHS=ON \
  -DGGML_CUDA_COMPRESSION_MODE=speed \
  -DGGML_LTO=ON \
  -DGGML_NATIVE=ON \
  "${HOST_CXX_FLAG[@]}"
cmake --build build -j "$(nproc)"

echo
echo "built: $LLAMA_DIR/build/bin/llama-server"
echo "opencode finds it via \$PATH or \$LLAMASTACK_SERVER_BIN (see docs/setup.md)"

#!/usr/bin/env bash
# Build llama.cpp for an Intel iGPU laptop (Arc 140T / Arrow Lake, Xe-LPG+).
#
# Companion to build-llama.sh (which is CUDA/NVIDIA). On this hardware the
# winning backend is **Vulkan via Mesa** — it beat SYCL and the archived
# ipex-llm on generation by 50-75%, and unlike SYCL/OpenVINO it links the
# system libvulkan so the binary runs with NO environment sourcing (which
# matters: opencode's router spawns llama-server directly, no shell wrapper).
#
# Deliberately NOT used here (measured on 2026-07-22, see docs/tuning-intel.md):
#   * SYCL      — ≈ archived ipex-llm speed, needs `source setvars.sh`.
#   * OpenVINO  — great prefill but slow quantized gen, crashes on MoE.
#   * coopmat   — Mesa's Xe-LPG+ cooperative-matrix path is 3x SLOWER at
#                 prefill; upstream correctly gates it to the Windows driver.
#
# Requires: Mesa Vulkan (mesa-vulkan-drivers), Vulkan loader + headers, glslc,
# SPIRV-Headers. On Fedora, if SPIRV-Headers has no distro package, install it
# user-local (see docs/setup-intel.md) and pass its prefix via SPIRV_PREFIX.
#
# Usage: ./scripts/build-llama-intel.sh [path-to-llama.cpp]
set -euo pipefail

LLAMA_DIR="${1:-$(cd "$(dirname "$0")/.." && pwd)/llama.cpp}"
[[ -f "$LLAMA_DIR/CMakeLists.txt" ]] || { echo "error: no llama.cpp checkout at $LLAMA_DIR (run: git submodule update --init llama.cpp)"; exit 1; }

command -v glslc >/dev/null || { echo "error: glslc not found (install glslc / shaderc)"; exit 1; }

# SPIRV-Headers: use SPIRV_PREFIX if given (user-local install), else rely on system.
CMAKE_EXTRA=()
if [[ -n "${SPIRV_PREFIX:-}" ]]; then
  CMAKE_EXTRA+=(-DCMAKE_PREFIX_PATH="$SPIRV_PREFIX" -DCMAKE_CXX_FLAGS="-I$SPIRV_PREFIX/include")
fi

GEN_FLAG=()
command -v ninja >/dev/null && GEN_FLAG=(-G Ninja)

cd "$LLAMA_DIR"
rm -rf build-vulkan
cmake -B build-vulkan "${GEN_FLAG[@]}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_VULKAN=ON \
  -DGGML_NATIVE=ON \
  -DLLAMA_CURL=OFF \
  -DLLAMA_BUILD_UI=OFF \
  "${CMAKE_EXTRA[@]}"
# -DLLAMA_BUILD_UI=OFF skips an npm/esbuild web-UI step that otherwise stalls
# the build on a network fetch and is useless for a headless coding server.
cmake --build build-vulkan -j "$(nproc)" --target llama-server llama-bench llama-cli

echo
echo "built: $LLAMA_DIR/build-vulkan/bin/llama-server"
echo "point opencode at it:  export LLAMASTACK_SERVER_BIN=$LLAMA_DIR/build-vulkan/bin/llama-server"
echo "verify matrix path is (correctly) OFF:  build-vulkan/bin/llama-bench --list-devices  # expect 'matrix cores: none'"

#!/usr/bin/env bash
# Backend shootout for an Intel iGPU (Arc 140T, Arrow Lake-H, xe + Mesa/ANV).
#
# Unlike the CUDA rig, the interesting variable here is not arch flags but the
# *backend*: llama.cpp can drive this GPU three ways, each linking a different
# userspace stack. This races them on the same GGUF with llama-bench (pp/tg),
# the same method as bench/nvfp4-bench.sh.
#
#   build-vulkan/   → Mesa ANV libvulkan   — no env sourcing needed
#   build-sycl/     → oneAPI DPC++/L0      — needs `source setvars.sh`
#   build-openvino/ → OpenVINO Runtime     — needs `source setupvars.sh`
#
# Baselines to beat on this machine (ipex-llm, Intel's archived fork):
#   qwen2.5-coder-7b   q4_k_m:        pp 279  tg 10.4
#   Qwen3-Coder-30B-A3B q4_k_m (MoE): pp 107  tg 16.6
#
# Usage: ./intel-backend-sweep.sh <model.gguf> [out-file]
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL="${1:?usage: $0 <model.gguf> [out-file]}"
OUT="${2:-$ROOT/bench/intel-sweep-results.txt}"
ONEAPI="${ONEAPI_SETVARS:-/opt/intel/oneapi/setvars.sh}"
OPENVINO="${OPENVINO_SETVARS:-$HOME/.local/openvino/setupvars.sh}"

# bench <label> <build-dir> <env-source-file|-> <extra llama-bench args...>
bench() {
  local label="$1" dir="$2" envf="$3"; shift 3
  local bin="$ROOT/llama.cpp/$dir/bin/llama-bench"
  [[ -x "$bin" ]] || { echo "$label: SKIP (no $dir build)" >> "$OUT"; return; }
  local src=""; [[ "$envf" != "-" && -f "$envf" ]] && src="source '$envf' >/dev/null 2>&1;"
  echo "=== $label ===" >> "$OUT"
  bash -c "$src exec '$bin' -m '$MODEL' $*" 2>>"$OUT" \
    | grep -E 'pp[0-9]+|tg[0-9]+' >> "$OUT" \
    || echo "$label: FAILED (see log)" >> "$OUT"
}

echo "## $(basename "$MODEL") — Intel backend sweep" >> "$OUT"
# Vulkan is the reference: fa off/on, both matter on this GPU (fa costs a little
# tg at d=0 but should pay off deep — probe both).
bench "vulkan fa0"   build-vulkan   -          -ngl 99 -fa 0 -p 512 -n 128 -r 3
bench "vulkan fa1"   build-vulkan   -          -ngl 99 -fa 1 -p 512 -n 128 -r 3
bench "sycl fa1"     build-sycl     "$ONEAPI"  -ngl 99 -fa 1 -p 512 -n 128 -r 3
# OpenVINO: huge prefill via graph compile, but quantized tg is still WIP and
# some MoE get_rows layouts assert-fail (ops.cpp GGML_ASSERT) — expect SKIP/FAIL
# on MoE, real numbers on dense.
bench "openvino gpu" build-openvino "$OPENVINO" -ngl 99 -p 512 -n 128 -r 3
bench "cpu floor"    build-vulkan   -          -ngl 0  -p 512 -n 128 -r 2
echo "" >> "$OUT"
echo "wrote $OUT"

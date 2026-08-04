#!/usr/bin/env bash
# Attention backends for the H3 benchmark matrix.
#
# WHY THIS IS THE FIRST THING TO TRY: H3 uses full dense attention
# (comfy/ldm/minimax/model.py -> optimized_attention(q,k,v,mask=None), no
# sparsity, no windowing) over one packed sequence of text+video+audio tokens.
# By FLOP count that is 58% of the work for a 5s clip and 80% at 15s. Nothing
# else on the knob list moves a number that large.
#
# ComfyUI dispatches two separate packages (comfy/ldm/modules/attention.py):
#   sageattention  -> sageattn            , selected by --use-sage-attention
#   sageattn3      -> sageattn3_blackwell , registered as attention fn "sage3"
# sage3 is FP4 microscaling built FOR Blackwell (sm_120 = these 5060 Tis), but
# in v0.30.1 nothing selects it from a launch flag — it needs a custom node that
# sets transformer_options["optimized_attention_override"]. So sage2 is the
# clean experiment and sage3 is the stretch goal.
#
# This box: python 3.12, CUDA 13.0, torch 2.13.0+cu130, sm_120, triton 3.7.1
# already present (SageAttention needs triton).
set -uo pipefail

cd "$(dirname "$0")/.."
ROOT=$PWD
PY=$ROOT/.venv/bin/python
export TORCH_CUDA_ARCH_LIST="12.0+PTX"

say() { printf '\n=== %s ===\n' "$1"; }

# nvcc is installed on this box (rpm cuda-nvcc-13-3) but is NOT on PATH — the
# driver alone is not enough to build SageAttention's CUDA kernels. Find it
# rather than concluding the toolkit is absent, which is what an earlier version
# of this script did.
if ! command -v nvcc >/dev/null 2>&1; then
  for d in /usr/local/cuda /usr/local/cuda-13.3 /usr/local/cuda-13 /opt/cuda; do
    if [[ -x "$d/bin/nvcc" ]]; then
      export CUDA_HOME="$d"
      export PATH="$d/bin:$PATH"
      break
    fi
  done
fi
echo "nvcc: $(command -v nvcc || echo 'NOT FOUND')  CUDA_HOME=${CUDA_HOME:-unset}"

# Fedora 44 ships gcc 16; CUDA 13.3 refuses anything past gcc 15
# (crt/host_config.h: "unsupported GNU version! gcc versions later than 15 are
# not supported"). Every nvcc compile fails with that unless a supported host
# compiler is named. gcc15/gcc15-c++ are already installed on this box from the
# exllamav3 build, so point CUDAHOSTCXX at it rather than reaching for
# -allow-unsupported-compiler, which only silences the check.
if [[ -x /usr/bin/g++-15 ]]; then
  export CUDAHOSTCXX=/usr/bin/g++-15
  export CC=/usr/bin/gcc-15 CXX=/usr/bin/g++-15
  echo "host compiler: $CUDAHOSTCXX ($(/usr/bin/g++-15 --version | head -1))"
else
  echo "host compiler: g++-15 NOT FOUND — nvcc will reject gcc $(gcc -dumpversion)."
  echo "               sudo dnf install gcc15 gcc15-c++"
fi

say "before"
$PY - <<'EOF'
for m in ("sageattention","sageattn3","flash_attn","triton","torch"):
    try:
        mod=__import__(m); print(f"  {m:14} {getattr(mod,'__version__','?')}")
    except Exception: print(f"  {m:14} missing")
EOF

# ---- SageAttention 2.x -------------------------------------------------------
# MUST be built from source. PyPI's `sageattention` is stuck at 1.0.6 — the old
# pure-Triton version that exports only sageattn/sageattn_varlen and declares no
# __version__. It imports perfectly well, which is the trap: an "is it
# importable" check passes and you benchmark v1 believing it is v2. Only a
# source build of thu-ml/SageAttention gives the INT8-QK v2.2 kernels that the
# 30-35%-over-SDPA reports refer to.
say "SageAttention 2 (source — PyPI only has v1)"
uv pip uninstall --python $PY sageattention 2>&1 | tail -1
rm -rf /tmp/SageAttention
SAGE2_OK=0
if git clone -q --depth 1 https://github.com/thu-ml/SageAttention /tmp/SageAttention; then
  (cd /tmp/SageAttention && uv pip install --python $PY --no-build-isolation . 2>&1 | tail -6)
  $PY -c "import sageattention" 2>/dev/null && SAGE2_OK=1
else
  echo "  clone failed"
fi

# Never leave the box worse off than it started. Uninstalling v1 up front is
# necessary (a v2 install over it would be shadowed), but if the v2 build fails
# we would otherwise have removed a working backend and installed nothing —
# turning row A2 from "measures v1" into "cannot run at all".
if [[ $SAGE2_OK -eq 0 ]]; then
  echo "  v2 build FAILED — restoring PyPI v1 so row A2 still has something to measure."
  echo "  !! results from A2 are then SageAttention 1 (pure Triton), NOT the"
  echo "  !! INT8-QK v2.2 the 30-35% figures refer to. Label them accordingly."
  uv pip install --python $PY sageattention 2>&1 | tail -2
fi

# ---- SageAttention 3 (Blackwell FP4) ----------------------------------------
# Subdirectory of the same repo, installs as `sageattn3`. FP4 microscaling built
# for sm_120. Compiles CUDA kernels — several minutes.
# NOTE even on success: v0.30.1 has no launch flag or stock node that selects
# "sage3", so it needs ComfyUI-SageAttention3 to be reachable. Installing it now
# only removes the build from the critical path.
say "SageAttention 3 (Blackwell FP4)"
if command -v nvcc >/dev/null 2>&1; then
  if [[ -d /tmp/SageAttention/sageattention3_blackwell ]]; then
    (cd /tmp/SageAttention/sageattention3_blackwell \
      && uv pip install --python $PY --no-build-isolation . 2>&1 | tail -6)
  else
    echo "  sageattention3_blackwell/ not in the checkout"
  fi
else
  echo "  skipped: still no nvcc"
fi

# ---- FlashAttention ----------------------------------------------------------
# Opt-in: WITH_FLASH=1 bench/install-accel.sh
# torch 2.13 already exposes flash SDPA, so the A0/A1 rows already exercise
# FlashAttention kernels; this package only adds the explicit
# --use-flash-attention path (row A3). The build takes tens of minutes and
# several GB of RAM — on 31 GiB with a 63 GB download in flight that is a good
# way to push the box into zram swap, so it is not run by default.
if [[ "${WITH_FLASH:-0}" == "1" ]]; then
  say "flash-attn (opt-in, slow build)"
  uv pip install --python $PY flash-attn --no-build-isolation 2>&1 | tail -4
else
  say "flash-attn — skipped (set WITH_FLASH=1 to build; row A3 will be skipped)"
fi

say "after"
$PY - <<'EOF'
import torch
for m in ("sageattention","sageattn3","flash_attn","triton"):
    try:
        mod=__import__(m); print(f"  {m:14} {getattr(mod,'__version__','?')}  OK")
    except Exception as e: print(f"  {m:14} missing ({type(e).__name__})")
print(f"  torch          {torch.__version__}  sm_{''.join(map(str,torch.cuda.get_device_capability(0)))}")
EOF
echo
echo "Rows the matrix can now run:  A2 needs sageattention, A3 needs flash_attn."
echo "Missing ones are skipped automatically by bench/sweep.py."

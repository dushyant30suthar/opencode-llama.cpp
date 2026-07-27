#!/usr/bin/env bash
# Install OpenVINO Model Server (OVMS) and generate servables for the local IR models.
#
# Third backend in this repo, and the only one that is NOT built from source:
# OVMS ships prebuilt binaries and a Bazel build tree that takes hours, so this
# script installs the official release tarball. The `model_server` submodule is
# pinned to the SAME tag purely as the reference for docs and the graph template
# below — nothing here compiles it.
#
# Rules this script enforces (learned the expensive way):
#
#   * The **python_on** package is mandatory, not a nicety. The C++-only
#     (`python_off`) build renders chat templates with a cut-down engine that
#     silently DROPS THE SYSTEM MESSAGE and CANNOT emit tool calls. Qwen3-Coder's
#     template opens with `{% macro %}` — python_off mis-renders it outright.
#     An agentic coding client without tool calls or a system prompt is useless,
#     so we pay the Python dependency.
#   * python_on links **libpython3.12** — Fedora 44 ships only 3.14, so the
#     interpreter comes from uv's standalone build (or a distro python3.12 if
#     one exists). Everything is user-local; no sudo anywhere.
#   * Jinja2/MarkupSafe live in their OWN venv, deliberately NOT the
#     openvino-genai venv. OVMS bundles its own openvino/openvino-genai wheels
#     and upstream is explicit that a pip-installed openvino alongside them
#     breaks the install. `lib/python` is first on PYTHONPATH so it always wins.
#   * The **redhat** package is used, not ubuntu22/24 — Fedora is the closer
#     glibc/OpenSSL relative and it loads clean on glibc 2.43 (verified).
#   * A generated `ovms-serve` wrapper exports LD_LIBRARY_PATH/PYTHONPATH and
#     execs the binary, so OVMS can be spawned directly by a supervisor with no
#     shell sourcing — the same constraint build-llama-intel.sh calls out.
#
# Usage:
#   ./scripts/setup-ovms.sh                 # install + generate servables
#   ./scripts/setup-ovms.sh --reinstall     # force re-download and re-extract
# Env:
#   OVMS_VERSION     release to install            (default: 2026.2.1)
#   OVMS_DISTRO      redhat | ubuntu22 | ubuntu24  (default: redhat)
#   OVMS_PREFIX      install dir                   (default: ~/.local/opt/ovms)
#   OVMS_STATE_DIR   servable/config dir           (default: ~/.local/state/ovms)
#   OVMS_MODELS_DIR  IR models dir                 (default: ~/.lmstudio/models-ov)
#   OVMS_CACHE_SIZE  KV cache GB per servable      (default: 4)
#   OVMS_CACHE_INTERVAL  linear-attention checkpoint multiplier (default: 128)
#   OVMS_DEVICE      GPU | CPU | NPU               (default: GPU)
#   OVMS_PORT        default REST port             (default: 8100)
set -euo pipefail

OVMS_VERSION="${OVMS_VERSION:-2026.2.1}"
OVMS_DISTRO="${OVMS_DISTRO:-redhat}"
OVMS_PREFIX="${OVMS_PREFIX:-$HOME/.local/opt/ovms}"
OVMS_STATE_DIR="${OVMS_STATE_DIR:-$HOME/.local/state/ovms}"
OVMS_MODELS_DIR="${OVMS_MODELS_DIR:-$HOME/.lmstudio/models-ov}"
OVMS_CACHE_SIZE="${OVMS_CACHE_SIZE:-4}"
OVMS_CACHE_INTERVAL="${OVMS_CACHE_INTERVAL:-128}"
OVMS_DEVICE="${OVMS_DEVICE:-GPU}"
# 8100 rather than OVMS's own 8000, which collides with common dev servers.
# The opencode provider defaults to the same number — keep the two in step.
OVMS_PORT="${OVMS_PORT:-8100}"
SRC_DIR="${OVMS_SRC_DIR:-$HOME/.local/src/ovms}"

REINSTALL=0
[[ "${1:-}" == "--reinstall" ]] && REINSTALL=1

TARBALL="ovms_${OVMS_DISTRO}_${OVMS_VERSION}_python_on.tar.gz"
URL="https://github.com/openvinotoolkit/model_server/releases/download/v${OVMS_VERSION}/${TARBALL}"

# ---------------------------------------------------------------- 1. libpython
# python_on has a hard DT_NEEDED on libpython3.12.so.1.0. Find one anywhere.
find_libpython() {
  local c
  for c in /usr/lib64/libpython3.12.so.1.0 /usr/lib/x86_64-linux-gnu/libpython3.12.so.1.0; do
    [[ -f "$c" ]] && { dirname "$c"; return 0; }
  done
  # uv keeps standalone CPython builds with a shared libpython under python/*/lib
  c="$(ls -d "$HOME"/.local/share/uv/python/cpython-3.12.*/lib/libpython3.12.so.1.0 2>/dev/null | sort -V | tail -1)"
  [[ -n "$c" ]] && { dirname "$c"; return 0; }
  return 1
}

PYLIB_DIR="$(find_libpython)" || {
  echo "error: libpython3.12.so.1.0 not found — OVMS python_on needs CPython 3.12."
  echo "       install one user-local with:  uv python install 3.12"
  exit 1
}
PY312="$(ls -d "$(dirname "$PYLIB_DIR")"/bin/python3.12 2>/dev/null || true)"
[[ -x "$PY312" ]] || PY312="$(command -v python3.12 || true)"
[[ -x "$PY312" ]] || { echo "error: found $PYLIB_DIR but no matching python3.12 interpreter"; exit 1; }
echo "libpython3.12: $PYLIB_DIR"
echo "python3.12:    $PY312"

# ------------------------------------------------------------- 2. fetch/extract
mkdir -p "$SRC_DIR"
if [[ ! -f "$SRC_DIR/$TARBALL" || "$REINSTALL" == 1 ]]; then
  echo "downloading $TARBALL ..."
  curl -fL --progress-bar -o "$SRC_DIR/$TARBALL" "$URL"
  curl -fsSL -o "$SRC_DIR/$TARBALL.sha256" "$URL.sha256" || true
else
  echo "using cached $SRC_DIR/$TARBALL"
fi
if [[ -s "$SRC_DIR/$TARBALL.sha256" ]]; then
  ( cd "$SRC_DIR" && sha256sum -c "$TARBALL.sha256" >/dev/null ) \
    && echo "checksum OK" \
    || { echo "error: checksum mismatch on $TARBALL — delete it and re-run"; exit 1; }
fi

if [[ ! -x "$OVMS_PREFIX/bin/ovms" || "$REINSTALL" == 1 ]]; then
  echo "extracting to $OVMS_PREFIX ..."
  rm -rf "$OVMS_PREFIX.tmp" && mkdir -p "$OVMS_PREFIX.tmp"
  tar -xzf "$SRC_DIR/$TARBALL" -C "$OVMS_PREFIX.tmp"     # tarball root is ./ovms
  rm -rf "$OVMS_PREFIX" && mv "$OVMS_PREFIX.tmp/ovms" "$OVMS_PREFIX"
  rmdir "$OVMS_PREFIX.tmp"
else
  echo "ovms already installed at $OVMS_PREFIX (use --reinstall to replace)"
fi

# ------------------------------------------------------------- 3. jinja2 venv
# Kept separate from openvino-genai/.venv on purpose: OVMS ships its own
# openvino + openvino-genai + openvino-tokenizers and upstream warns that a
# pip-installed openvino on the same path breaks dependencies.
if [[ ! -x "$OVMS_PREFIX/pyenv/bin/python" || "$REINSTALL" == 1 ]]; then
  echo "creating template venv (Jinja2 only) ..."
  rm -rf "$OVMS_PREFIX/pyenv"
  "$PY312" -m venv "$OVMS_PREFIX/pyenv"
  "$OVMS_PREFIX/pyenv/bin/pip" -q install "Jinja2==3.1.6" "MarkupSafe==3.0.2"
fi
SITE_PKGS="$OVMS_PREFIX/pyenv/lib/python3.12/site-packages"

# --------------------------------------------------------------- 4. servables
# graph.pbtxt is the LLM-calculator MediaPipe graph. This is upstream's own
# text_generation template (demos/common/export_models/export_model.py) with the
# knobs filled in. Only node_options is ours; the rest must stay verbatim — the
# LLM_NODE_RESOURCES side packet in particular, or the graph is rejected.
#
# tool_parser turns the model's native tool syntax into OpenAI tool_calls.
# Picking the wrong one is a SILENT failure: the raw markup arrives as assistant
# text, tool_calls stays empty, and an agent client just sees the model refusing
# to use its tools. Supported upstream: llama3, phi4, hermes3, mistral,
# qwen3coder, gptoss, devstral, lfm2, gemma4.
#
# Read the model's own chat template rather than guessing from its name. Names
# lie: the Qwen3.6 checkpoints ship Qwen3-Coder's <tool_call><function=...>
# XML verbatim despite not being "coder" models, so a name rule sends them to
# hermes3 and breaks tool calling. Two markers tell the families apart —
#   <function=   -> qwen3coder  (XML-nested function/parameter blocks)
#   tool_call    -> hermes3     (a JSON object inside <tool_call> tags)
# and <function= must be tested first, since those templates contain both.
tool_parser_for() {
  local name="${1,,}" dir="$2" tpl
  for tpl in "$dir/chat_template.jinja" "$dir/tokenizer_config.json"; do
    [[ -f "$tpl" ]] || continue
    grep -q '<function=' "$tpl" 2>/dev/null && { echo "qwen3coder"; return; }
    grep -q 'tool_call'  "$tpl" 2>/dev/null && { echo "hermes3";    return; }
  done
  # No template on disk — fall back to the name.
  case "$name" in
    *qwen3-coder*|*qwen3coder*) echo "qwen3coder" ;;
    *qwen2.5*|*qwen3*)          echo "hermes3" ;;
    *llama-3*|*llama3*)         echo "llama3" ;;
    *mistral*)                  echo "mistral" ;;
    *)                          echo "" ;;
  esac
}

# reasoning_parser lifts chain-of-thought out of `content` into a separate
# `reasoning_content` field. Without it a thinking model's private reasoning is
# served as ordinary assistant text, trailing a bare unmatched `</think>` —
# user-visible garbage, and it defeats any client that special-cases reasoning.
# Detected from the template again: `<think>` present means the model thinks.
# Upstream supports qwen3, gptoss, gemma4 — the <think>/</think> convention is
# Qwen's, so that is the default for anything else carrying those tags.
reasoning_parser_for() {
  local name="${1,,}" dir="$2" tpl
  for tpl in "$dir/chat_template.jinja" "$dir/tokenizer_config.json"; do
    [[ -f "$tpl" ]] || continue
    grep -q '<think>' "$tpl" 2>/dev/null || { echo ""; return; }
    case "$name" in
      *gemma*)            echo "gemma4" ;;
      *gpt-oss*|*gptoss*) echo "gptoss" ;;
      *)                  echo "qwen3" ;;
    esac
    return
  done
  echo ""
}

write_servable() {
  local name="$1" model_dir="$2" device="$3" parser reasoner dir
  parser="$(tool_parser_for "$name" "$model_dir")"
  reasoner="$(reasoning_parser_for "$name" "$model_dir")"
  dir="$OVMS_STATE_DIR/servables/$name"
  mkdir -p "$dir"
  {
    echo '# OVMS_GRAPH_QUEUE_MAX_SIZE: AUTO'
    echo 'input_stream: "HTTP_REQUEST_PAYLOAD:input"'
    echo 'output_stream: "HTTP_RESPONSE_PAYLOAD:output"'
    echo
    echo 'node: {'
    echo '  name: "LLMExecutor"'
    echo '  calculator: "HttpLLMCalculator"'
    echo '  input_stream: "LOOPBACK:loopback"'
    echo '  input_stream: "HTTP_REQUEST_PAYLOAD:input"'
    echo '  input_side_packet: "LLM_NODE_RESOURCES:llm"'
    echo '  output_stream: "LOOPBACK:loopback"'
    echo '  output_stream: "HTTP_RESPONSE_PAYLOAD:output"'
    echo '  input_stream_info: {'
    echo "    tag_index: 'LOOPBACK:0',"
    echo '    back_edge: true'
    echo '  }'
    echo '  node_options: {'
    echo '      [type.googleapis.com / mediapipe.LLMCalculatorOptions]: {'
    echo "          models_path: \"$model_dir\","
    # u8 KV halves cache bytes per token, which on a 4 GB budget is the
    # difference between ~43k and ~87k tokens of context on the 30B MoE.
    echo "          plugin_config: '{\"KV_CACHE_PRECISION\":\"u8\"}',"
    echo '          enable_prefix_caching: true,'
    echo "          cache_size: $OVMS_CACHE_SIZE,"
    # Hybrid models (Qwen3.6, Qwen3.5) checkpoint their whole fp32 recurrent
    # state every kv_block_size * this value for prefix caching. The default of
    # 8 makes those snapshots dwarf the KV cache — measured 267 KiB/token, i.e.
    # ~15k usable context, and it wrecks prefill too (a 44k prompt writes ~23GB
    # of snapshots). At 128 it is 33 KiB/token and 90k prompts answer.
    # Harmless on non-hybrid models. Fixed upstream in openvino.genai #4050 (2026.3).
    echo "          cache_interval_multiplier: ${OVMS_CACHE_INTERVAL:-128},"
    echo '          max_num_seqs: 8,'
    echo "          device: \"$device\","
    [[ -n "$parser" ]] && echo "          tool_parser: \"$parser\","
    [[ -n "$reasoner" ]] && echo "          reasoning_parser: \"$reasoner\","
    echo '      }'
    echo '  }'
    echo '  input_stream_handler {'
    echo '    input_stream_handler: "SyncSetInputStreamHandler",'
    echo '    options {'
    echo '      [mediapipe.SyncSetInputStreamHandlerOptions.ext] {'
    echo '        sync_set {'
    echo '          tag_index: "LOOPBACK:0"'
    echo '        }'
    echo '      }'
    echo '    }'
    echo '  }'
    echo '}'
  } > "$dir/graph.pbtxt"
  echo "  $name  ->  $model_dir  [device=$device${parser:+, tool_parser=$parser}${reasoner:+, reasoning_parser=$reasoner}]"
}

echo
echo "generating servables in $OVMS_STATE_DIR/servables:"
found=0
if [[ -d "$OVMS_MODELS_DIR" ]]; then
  for d in "$OVMS_MODELS_DIR"/*/; do
    # Text models export openvino_model.xml; VLMs split into
    # openvino_language_model.xml + vision/embedding parts. OVMS infers the
    # servable type from the directory itself, so the graph is the same either
    # way — but requiring one of these skips half-finished downloads.
    [[ -f "$d/openvino_model.xml" || -f "$d/openvino_language_model.xml" ]] || continue
    name="$(basename "$d")"
    # NPU-exported models are stateful-only; OVMS picks that path from the device.
    dev="$OVMS_DEVICE"
    [[ "${name,,}" == *npu* ]] && dev="NPU"
    write_servable "$name" "${d%/}" "$dev"
    found=$((found + 1))
  done
fi
[[ "$found" == 0 ]] && echo "  (no IR models found under $OVMS_MODELS_DIR — set OVMS_MODELS_DIR)"

# ---------------------------------------------------------------- 5. launcher
# Self-contained: sets its own env and execs, so a supervisor can spawn it
# directly. lib/python MUST precede the Jinja venv so OVMS's bundled openvino
# wins over anything else on the system.
cat > "$OVMS_PREFIX/bin/ovms-serve" <<EOF
#!/usr/bin/env bash
# Generated by scripts/setup-ovms.sh — serve one servable over the OpenAI API.
# Usage: ovms-serve <servable-name> [--rest_port N] [extra ovms args...]
set -euo pipefail
export LD_LIBRARY_PATH="$OVMS_PREFIX/lib:$PYLIB_DIR\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}"
export PYTHONPATH="$OVMS_PREFIX/lib/python:$SITE_PKGS"
name="\${1:?usage: ovms-serve <servable-name> [ovms args...]}"; shift
dir="$OVMS_STATE_DIR/servables/\$name"
[[ -f "\$dir/graph.pbtxt" ]] || { echo "error: no servable '\$name' in $OVMS_STATE_DIR/servables"; exit 1; }
port_given=0; for a in "\$@"; do [[ "\$a" == "--rest_port" ]] && port_given=1; done
[[ \$port_given == 1 ]] || set -- --rest_port "\${OVMS_PORT:-$OVMS_PORT}" "\$@"
exec "$OVMS_PREFIX/bin/ovms" --model_path "\$dir" --model_name "\$name" "\$@"
EOF
chmod +x "$OVMS_PREFIX/bin/ovms-serve"

echo
echo "installed: $OVMS_PREFIX/bin/ovms"
LD_LIBRARY_PATH="$OVMS_PREFIX/lib:$PYLIB_DIR" "$OVMS_PREFIX/bin/ovms" --version 2>/dev/null | sed 's/^/  /'
echo "launcher:  $OVMS_PREFIX/bin/ovms-serve <servable-name>"
echo
echo "serve the daily driver (one model per process):"
echo "  $OVMS_PREFIX/bin/ovms-serve Qwen3-Coder-30B-A3B-Instruct-int4-ov"
echo "then:  curl http://localhost:$OVMS_PORT/v3/models"
echo "OpenAI base URL: http://localhost:$OVMS_PORT/v3   (see docs/openvino-server.md)"

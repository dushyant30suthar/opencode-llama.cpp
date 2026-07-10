#!/usr/bin/env bash
# Production-faithful max-context search: probes via llama-server with the SAME
# config the router uses — mmproj included, tensor split, q8_0 KV, fa on.
set -u
SRV="$(dirname "$0")/../llama.cpp/build/bin/llama-server"
OUT=~/.local/state/llamastack/ctx-results-v2.txt
LOG=~/.local/state/llamastack/ctx-search-v2.log
PORT=9444
: > "$OUT"; : > "$LOG"

probe() { # probe <model> <mmproj> <ctx> <ub> -> 0 if server loads and answers
  local pid ok=1
  "$SRV" -m "$1" --mmproj "$2" -c "$3" -ub "$4" -sm tensor -ngl 99 -fa on \
    -ctk q8_0 -ctv q8_0 --jinja --host 127.0.0.1 --port $PORT >>"$LOG" 2>&1 &
  pid=$!
  for _ in $(seq 1 60); do
    sleep 3
    kill -0 $pid 2>/dev/null || break                      # crashed
    if curl -s --max-time 2 http://127.0.0.1:$PORT/health | grep -q '"ok"'; then
      curl -s --max-time 60 http://127.0.0.1:$PORT/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d '{"messages":[{"role":"user","content":"hi"}],"max_tokens":1}' \
        | grep -q '"content"' && ok=0
      break
    fi
  done
  kill $pid 2>/dev/null; wait $pid 2>/dev/null; sleep 2
  return $ok
}

search() { # search <name> <model> <mmproj> <lo> <hi> <ub>
  local name="$1" model="$2" mmproj="$3" lo="$4" hi="$5" ub="$6" best=0
  if probe "$model" "$mmproj" "$lo" "$ub"; then best=$lo; else
    echo "$name FAIL_AT_FLOOR" >>"$OUT"; return; fi
  while (( hi - lo > 8192 )); do
    local mid=$(( (lo + hi) / 2 / 8192 * 8192 ))
    (( mid == lo )) && break
    if probe "$model" "$mmproj" "$mid" "$ub"; then best=$mid; lo=$mid; else hi=$mid; fi
    echo "$(date +%T) $name: tried $mid best=$best" >>"$LOG"
  done
  echo "$name $best" >>"$OUT"
}

M=~/.lmstudio/models/lmstudio-community
search "Qwen3.6-27B-GGUF" "$M/Qwen3.6-27B-GGUF/Qwen3.6-27B-Q4_K_M.gguf" \
  "$M/Qwen3.6-27B-GGUF/mmproj-Qwen3.6-27B-BF16.gguf" 131072 258048 2048
search "Qwen3.6-35B-A3B-GGUF" "$M/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-Q4_K_M.gguf" \
  "$M/Qwen3.6-35B-A3B-GGUF/mmproj-Qwen3.6-35B-A3B-BF16.gguf" 131072 258048 2048
search "gemma-4-31B-it-QAT-GGUF" "$M/gemma-4-31B-it-QAT-GGUF/gemma-4-31B-it-QAT-Q4_0.gguf" \
  "$M/gemma-4-31B-it-QAT-GGUF/mmproj-gemma-4-31B-it-QAT-BF16.gguf" 98304 196608 512
echo "DONE" >>"$OUT"

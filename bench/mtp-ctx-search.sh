#!/usr/bin/env bash
set -u
SRV="$(dirname "$0")/../llama.cpp/build/bin/llama-server"
MODEL=~/.lmstudio/models/unsloth/Qwen3.6-27B-MTP-GGUF/Qwen3.6-27B-UD-Q4_K_XL.gguf
OUT=~/.local/state/llamastack/mtp-ctx-results.txt
PORT=9444
: > "$OUT"

probe() {
  local pid ok=1
  "$SRV" -m "$MODEL" -c "$1" -ngl 99 -fa on -sm tensor -ub 2048 -ctk q8_0 -ctv q8_0 \
    --jinja -np 1 --spec-type draft-mtp --spec-draft-n-max 4 \
    --host 127.0.0.1 --port $PORT >/dev/null 2>&1 &
  pid=$!
  for _ in $(seq 1 60); do
    sleep 3
    kill -0 $pid 2>/dev/null || break
    if curl -s --max-time 2 http://127.0.0.1:$PORT/health | grep -q '"ok"'; then
      curl -s --max-time 60 http://127.0.0.1:$PORT/v1/chat/completions -H "Content-Type: application/json" \
        -d '{"messages":[{"role":"user","content":"hi"}],"max_tokens":1}' | grep -q '"content"' && ok=0
      break
    fi
  done
  kill $pid 2>/dev/null; wait $pid 2>/dev/null; sleep 2
  return $ok
}

lo=65536; hi=245760; best=0
if probe $lo; then best=$lo; else echo "MTP-27B FAIL_AT_FLOOR" >>"$OUT"; exit 1; fi
while (( hi - lo > 8192 )); do
  mid=$(( (lo + hi) / 2 / 8192 * 8192 ))
  (( mid == lo )) && break
  if probe $mid; then best=$mid; lo=$mid; else hi=$mid; fi
  echo "probe $mid -> best=$best" >>"$OUT"
done
echo "MTP-27B MAX $best" >>"$OUT"

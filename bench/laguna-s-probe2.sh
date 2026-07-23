#!/usr/bin/env bash
# laguna-s-probe2.sh — follow-ups after the K ladder in laguna-s-tune.sh.
#   1. --no-mmap: llama.cpp warns "tensor overrides to CPU are used with mmap
#      enabled - consider using --no-mmap for better performance". Worth real
#      numbers; safer at high K because less of the model sits in host RAM.
#   2. context probe at the winning K (S is 1M-capable but VRAM-bound here).
# Usage: ./laguna-s-probe2.sh <K>      (K = expert layers per GPU that won)
set -u
K="${1:?usage: laguna-s-probe2.sh <K>}"
BIN="$(cd "$(dirname "$0")/.." && pwd)/llama.cpp/build/bin"
M=~/.lmstudio/models/unsloth/Laguna-S-2.1-GGUF/Laguna-S-2.1-UD-IQ3_XXS.gguf
OUT="$(dirname "$0")/laguna-s-tune-results.txt"
LOG="$(dirname "$0")/laguna-s-tune.log"
PORT=9466

cpu_regex() {
  python3 -c "
K=$1
cpu=[l for l in range(48) if not (24-K<=l<=23 or 48-K<=l<=47)]
print('blk\\\\.(' + '|'.join(map(str,cpu)) + ')\\\\.ffn_.*_exps\\\\.=CPU')"
}
RE=$(cpu_regex "$K")

PROMPT='Review this Python diff for bugs and respond with the corrected function and a one-line verdict:\n```\n def get(self, key):\n-    if key in self.cache:\n-        self.cache.move_to_end(key)\n-    return self.cache[key]\n+    if key not in self.cache:\n+        return None\n+    self.cache.move_to_end(key)\n+    return self.cache[key]\n```'

measure() {
  local label="$1"; shift
  "$BIN/llama-server" -m "$M" -ngl 99 -fa on -ub 512 --jinja --no-warmup \
    --temp 1.0 --top-k 20 --top-p 1.0 --min-p 0 --cache-ram 0 \
    --host 127.0.0.1 --port $PORT "$@" >>"$LOG" 2>&1 &
  local pid=$! up=0
  for _ in $(seq 1 200); do sleep 3; kill -0 $pid 2>/dev/null || break
    curl -s --max-time 2 http://127.0.0.1:$PORT/health | grep -q '"ok"' && { up=1; break; }; done
  if [ "$up" != 1 ]; then echo "$label: LOAD_FAILED" >>"$OUT"; kill $pid 2>/dev/null; wait $pid 2>/dev/null; sleep 3; return 1; fi
  local vram t pp
  vram=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | paste -sd/ -)
  read -r t pp < <(curl -s --max-time 1200 http://127.0.0.1:$PORT/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d "{\"messages\":[{\"role\":\"user\",\"content\":\"$PROMPT\"}],\"max_tokens\":200}" \
    | jq -r '"\(.timings.predicted_per_second // 0) \(.timings.prompt_per_second // 0)"')
  printf '%s: gen=%.2f t/s prompt=%.2f t/s vram=%s MiB\n' "$label" "${t:-0}" "${pp:-0}" "$vram" >>"$OUT"
  kill $pid 2>/dev/null; wait $pid 2>/dev/null; sleep 4
  return 0
}

echo "== S probe2 at K=$K ==" >>"$OUT"
measure "K$K-nommap"    -ot "$RE" -ctk q8_0 -ctv q8_0 -c 32768 --no-mmap
echo "-- context probe (mmap default) --" >>"$OUT"
for C in 65536 131072; do
  measure "K$K-ctx$C" -ot "$RE" -ctk q8_0 -ctv q8_0 -c $C || { echo "K$K-ctx$C: DID NOT FIT" >>"$OUT"; break; }
done
echo "PROBE2 DONE $(date +%T)" >>"$OUT"

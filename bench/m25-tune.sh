#!/usr/bin/env bash
# m25-tune.sh v2 — MiniMax M2.5 IQ2_XXS on 2×16GB + 31GB RAM.
# v1 lesson: --n-cpu-moe puts all GPU-expert layers on GPU1 (last layers) —
# half the VRAM idle. v2 places K expert layers on EACH GPU explicitly via
# -ot, co-located with that GPU's attention half (GPU0: layers ≤30, GPU1: ≥31).
set -u
BIN="$(dirname "$0")/../llama.cpp/build/bin"
M=~/.lmstudio/models/UD-IQ2_XXS/MiniMax-M2.5-UD-IQ2_XXS-00001-of-00003.gguf
OUT="$(dirname "$0")/m25-tune-results.txt"
LOG="$(dirname "$0")/m25-tune.log"
PORT=9444
: > "$OUT"; : > "$LOG"

cpu_regex() { # cpu_regex <K> -> -ot regex sending all but 2K expert layers to CPU
  python3 -c "
K=$1
cpu=[l for l in range(62) if not (31-K<=l<=30 or 62-K<=l<=61)]
print('blk\\\\.(' + '|'.join(map(str,cpu)) + ')\\\\.ffn_.*_exps\\\\.=CPU')"
}

PROMPT='Review this Python diff for bugs and respond with the corrected function and a one-line verdict:\n```\n def get(self, key):\n-    if key in self.cache:\n-        self.cache.move_to_end(key)\n-    return self.cache[key]\n+    if key not in self.cache:\n+        return None\n+    self.cache.move_to_end(key)\n+    return self.cache[key]\n def put(self, key, value):\n     self.cache[key] = value\n     self.cache.move_to_end(key)\n-    if len(self.cache) >= self.cap:\n+    if len(self.cache) > self.cap:\n         self.cache.popitem(last=False)\n```'

measure() { # measure <label> <extra server flags...>
  local label="$1"; shift
  "$BIN/llama-server" -m "$M" -ngl 99 -fa on -c 32768 -ub 512 --jinja --no-warmup \
    --temp 1.0 --top-p 0.95 --top-k 40 --min-p 0.01 --cache-ram 0 \
    --host 127.0.0.1 --port $PORT "$@" >>"$LOG" 2>&1 &
  local pid=$! up=0
  for _ in $(seq 1 160); do
    sleep 3
    kill -0 $pid 2>/dev/null || break
    curl -s --max-time 2 http://127.0.0.1:$PORT/health | grep -q '"ok"' && { up=1; break; }
  done
  if [ "$up" != 1 ]; then echo "$label: LOAD_FAILED" >>"$OUT"; kill $pid 2>/dev/null; wait $pid 2>/dev/null; return 1; fi
  local vram t
  vram=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | paste -sd/ -)
  t=$(curl -s --max-time 1500 http://127.0.0.1:$PORT/v1/chat/completions -H "Content-Type: application/json" \
    -d "{\"messages\":[{\"role\":\"user\",\"content\":\"$PROMPT\"}],\"max_tokens\":150}" \
    | jq -r '.timings.predicted_per_second // empty')
  echo "$label: ${t:-NO_TIMING} t/s vram=${vram}MiB" >>"$OUT"
  kill $pid 2>/dev/null; wait $pid 2>/dev/null; sleep 3
  return 0
}

curl -s --max-time 10 -X POST http://127.0.0.1:9337/models/unload \
  -H "Content-Type: application/json" -d '{"model":"unsloth/Devstral-Small-2-24B-Instruct-2512-GGUF"}' >/dev/null 2>&1

echo "== balanced -ot ladder: K expert layers per GPU (2K on GPU total)" >>"$OUT"
BESTK=""
for K in 9 11 13; do
  RE=$(cpu_regex $K)
  if measure "K$K-q8kv" -ot "$RE" -ctk q8_0 -ctv q8_0; then BESTK=$K; else break; fi
done

if [ -n "$BESTK" ]; then
  NG=$((BESTK-3)); [ $NG -lt 6 ] && NG=6
  echo "== ngram speculation (f16 KV) at reduced K=$NG" >>"$OUT"
  RE=$(cpu_regex $NG)
  measure "K$NG-ngram-f16kv" -ot "$RE" -ctk f16 -ctv f16 --spec-type ngram-mod
fi
echo "M25 TUNE V2 DONE (best plain K=$BESTK)" >>"$OUT"
cat "$OUT"

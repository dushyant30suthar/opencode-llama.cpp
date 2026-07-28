#!/bin/bash
# EXL3 tensor-parallel test: official wheel (tabbyAPI/venv, exllamav3 1.1.0) first,
# patched source (exllamav3/venv, master+fix) as fallback. MTP disabled in both.
# Runs in a hard window; restores the production router at the end no matter what.
set -u
OUT=/home/dushyant30suthar/Projects/opencode-localhost
RESTORE=/tmp/claude-1000/-home-dushyant30suthar-Projects-opencode-localhost/af5f9353-9954-46ff-aba2-9bd7fdd6f16d/scratchpad/RESTORE.sh

test_engine() {
  local name=$1 py=$2
  echo "=== ENGINE: $name ==="
  cd /home/dushyant30suthar/Projects/tabbyAPI
  nohup "$py" main.py --config config-tp.yml > "$OUT/tabby-tp-$name.log" 2>&1 &
  local pid=$!
  for i in $(seq 1 50); do
    curl -s --max-time 2 http://127.0.0.1:5000/health 2>/dev/null | grep -qi "healthy\|ok" && { echo "HEALTHY"; break; }
    kill -0 $pid 2>/dev/null || { echo "DIED-AT-LOAD:"; tail -4 "$OUT/tabby-tp-$name.log"; return 1; }
    sleep 5
  done
  # sampled probe first (the #245 trigger condition)
  local r
  r=$(curl -s --max-time 90 http://127.0.0.1:5000/v1/completions -H 'Content-Type: application/json' \
    -d '{"model":"Qwen3.6-27B-exl3-5.00bpw","prompt":"def quicksort(arr):","max_tokens":48,"temperature":0.6,"top_k":20}' \
    | python3 -c "import json,sys; d=json.load(sys.stdin); print('SAMPLED-OK' if 'choices' in d and d['choices'][0]['text'] else 'SAMPLED-FAIL')" 2>&1)
  echo "probe: $r"
  if [ "$r" = "SAMPLED-OK" ]; then
    cd "$OUT" && python3 sweepx.py http://127.0.0.1:5000 "exl3-TP-noMTP-$name" sweep exl3-results.jsonl > "$OUT/sweep-tp-$name.log" 2>&1
    echo "sweep done: $(grep -c decode_tps "$OUT/sweep-tp-$name.log" 2>/dev/null || echo '?') rows"
  fi
  kill $pid 2>/dev/null; sleep 5; pkill -9 -f "main[.]py --config config-tp" 2>/dev/null; sleep 2
  [ "$r" = "SAMPLED-OK" ] && return 0 || return 1
}

# window open: stop router + model
pkill -f "models[-]preset" 2>/dev/null; sleep 2
for p in $(pgrep -x llama-server); do kill $p; done; sleep 4
echo "window open; VRAM: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader | tr '\n' ' ')"

# truncated sweep for TP: patch sweepx depths via env? No — sweepx uses fixed DEPTHS; full list is fine (adds prefill time but complete data)
if test_engine official /home/dushyant30suthar/Projects/tabbyAPI/venv/bin/python; then
  echo "TP-VERDICT: official wheel works"
else
  echo "official failed -> trying patched source build"
  if test_engine patched /home/dushyant30suthar/Projects/exllamav3/venv/bin/python; then
    echo "TP-VERDICT: patched source works (official failed)"
  else
    echo "TP-VERDICT: both failed"
  fi
fi

bash "$RESTORE"
echo "TP-TEST-COMPLETE"

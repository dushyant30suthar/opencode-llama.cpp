#!/bin/bash
# Find the largest cache_size that survives a cold deep prefill.
# Acceptance test is a real 98k cold prefill (the thing that OOM'd at 360448),
# not just a healthy /health — at-rest VRAM tells you nothing about the peak.
set -u
EX=/home/dushyant30suthar/Projects/exllamav3
TB=/home/dushyant30suthar/Projects/tabbyAPI
ST=/home/dushyant30suthar/.local/state/opencode/providers/exl3
CFG=$TB/config-crown.yml
CORPUS=$(cd "$(dirname "$0")" && pwd)/corpus.txt

for VAL in "$@"; do
  echo "=== cache_size = $VAL ==="
  sed -i "/^  cache_size:/d" "$CFG"
  sed -i "/^model:/a\\  cache_size: $VAL" "$CFG"
  pkill -f "main[.]py --config" 2>/dev/null
  for i in $(seq 1 40); do ss -tln 2>/dev/null | grep -q ":5000 " || break; sleep 2; done
  pkill -9 -f "main[.]py --config" 2>/dev/null
  for i in $(seq 1 20); do ss -tln 2>/dev/null | grep -q ":5000 " || break; sleep 2; done
  sleep 2
  cd "$TB"
  nohup "$EX/venv/bin/python" main.py --config config-crown.yml > "$ST/server.log" 2>&1 &
  echo $! > "$ST/server.pid"
  UP=0
  for i in $(seq 1 60); do
    curl -s --max-time 2 http://127.0.0.1:5000/health 2>/dev/null | grep -q healthy && { UP=1; break; }
    kill -0 "$(cat $ST/server.pid)" 2>/dev/null || break
    sleep 5
  done
  [ $UP -eq 0 ] && { echo "  WOULD NOT LOAD"; continue; }
  echo "  at rest: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader | tr '\n' ' ')"

  OK=$(python3 - "$CORPUS" <<'PY'
import json, sys, urllib.request
corpus = open(sys.argv[1]).read()
def post(p, o, t=1800):
    r = urllib.request.Request("http://127.0.0.1:5000"+p, json.dumps(o).encode(),
                               {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=t))
try:
    toks = post("/v1/token/encode", {"text": corpus[:1400000]})["tokens"]
    body = post("/v1/token/decode", {"tokens": toks[:98304]})["text"]
    # two distinct cold prefills: the second also proves the pool survives reuse
    for rep in range(2):
        post("/v1/completions", {"model": "Qwen3.6-27B-exl3-5.00bpw",
                                 "prompt": f"// probe {rep} {rep*7919}\n" + body,
                                 "max_tokens": 8, "temperature": 0.6, "top_k": 20})
    print("PASS")
except Exception as e:
    print(f"FAIL {type(e).__name__}")
PY
)
  echo "  cold 98k prefill x2: $OK   peak: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader | tr '\n' ' ')"
done
echo "CACHE-MAX-COMPLETE"

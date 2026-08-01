#!/bin/bash
# Confirmation run: cache modes at the depths that matter, 4 reps each, so
# acceptance noise averages out. Reports median + spread from server metrics.
set -u
EX=/home/dushyant30suthar/Projects/exllamav3
TB=/home/dushyant30suthar/Projects/tabbyAPI
ST=/home/dushyant30suthar/.local/state/opencode/providers/exl3
CFG=/home/dushyant30suthar/.config/opencode/providers/exl3/models/Qwen3.6-27B-exl3-5.00bpw.yml
CORPUS=$(cd "$(dirname "$0")" && pwd)/corpus.txt

for VAL in "$@"; do
  echo "=== cache_mode = $VAL ==="
  sed -i "/^  cache_mode:/d" "$CFG"
  sed -i "/^model:/a\\  cache_mode: $VAL" "$CFG"
  # Wait for the PORT, not just the process: TP workers outlive the parent
  # briefly, and a half-dead server still answers /health while the next
  # instance is loading — which silently benchmarks the wrong server (or 503s).
  pkill -f "main[.]py --config" 2>/dev/null
  for i in $(seq 1 40); do ss -tln 2>/dev/null | grep -q ":5000 " || break; sleep 2; done
  pkill -9 -f "main[.]py --config" 2>/dev/null
  for i in $(seq 1 20); do ss -tln 2>/dev/null | grep -q ":5000 " || break; sleep 2; done
  sleep 2
  MARK=$(date "+%Y-%m-%d %H:%M:%S")
  cd "$TB"
  nohup "$EX/venv/bin/python" main.py --config /home/dushyant30suthar/.config/opencode/providers/exl3/models/Qwen3.6-27B-exl3-5.00bpw.yml > "$ST/server.log" 2>&1 &
  echo $! > "$ST/server.pid"
  UP=0
  for i in $(seq 1 60); do
    curl -s --max-time 2 http://127.0.0.1:5000/health 2>/dev/null | grep -q healthy && { UP=1; break; }
    kill -0 "$(cat $ST/server.pid)" 2>/dev/null || break
    sleep 5
  done
  # /health can go green before the model is servable; require a real answer
  if [ $UP -eq 1 ]; then
    curl -s --max-time 120 http://127.0.0.1:5000/v1/completions -H 'Content-Type: application/json' \
      -d '{"model":"Qwen3.6-27B-exl3-5.00bpw","prompt":"ping","max_tokens":1}' \
      | grep -q '"choices"' || { echo "  server up but not serving"; UP=0; }
  fi
  [ $UP -eq 0 ] && { echo "  FAILED TO START"; tail -3 "$ST/server.log"; continue; }
  echo "  VRAM: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader | tr '\n' ' ')"

  python3 - "$CORPUS" <<'PY'
import json, sys, urllib.request
corpus = open(sys.argv[1]).read()
def post(path, obj, t=1200):
    req = urllib.request.Request("http://127.0.0.1:5000"+path, json.dumps(obj).encode(),
                                 {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=t))
toks = post("/v1/token/encode", {"text": corpus[:1400000]})["tokens"]
for depth in (98304, 131072):
    text = post("/v1/token/decode", {"tokens": toks[:depth]})["text"]
    for rep in range(5):          # rep0 warms; 4 measured
        post("/v1/completions", {"model": "Qwen3.6-27B-exl3-5.00bpw", "prompt": text,
                                 "max_tokens": 400, "temperature": 0.6, "top_k": 20,
                                 "top_p": 0.95, "min_p": 0.0})
print("  workload done", flush=True)
PY

  python3 - "$ST/server.log" "$MARK" <<'PY'
import re, sys, statistics as st
raw = re.sub(r"\s+", " ", open(sys.argv[1]).read())
pat = (r"(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)[^M]*Metrics \(ID: [a-f0-9]+\): (\d+) tokens generated in "
       r"([\d.]+) seconds \(Queue: ([\d.]+) s, Process: (\d+) cached tokens and (\d+) new tokens at "
       r"([\d.]+) T/s, Generate: ([\d.]+) T/s, Context: (\d+) tokens, Draft: (\d+) / (\d+) tokens "
       r"accepted \(([\d.]+)%\)\)")
rows = [m for m in re.findall(pat, raw) if m[0] >= sys.argv[2] and int(m[1]) > 100]
by = {}
for m in rows:
    b = min([98304, 131072], key=lambda d: abs(d - int(m[8])))
    by.setdefault(b, []).append((float(m[7]), float(m[11])))
for d in sorted(by):
    tg = sorted(x[0] for x in by[d]); acc = [x[1] for x in by[d]]
    print(f"  ctx {d}: decode med {st.median(tg):>5.1f}  [{tg[0]:.1f}-{tg[-1]:.1f}]  "
          f"acc med {st.median(acc):>4.0f}%  n={len(tg)}")
PY
done
echo "DEEP-CONFIRM-COMPLETE"

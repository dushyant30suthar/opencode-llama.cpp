#!/bin/bash
# A/B a prefill knob. Every measurement is a COLD prefill: each rep prepends a
# unique marker so the page hashes miss and the whole prompt is reprocessed.
# usage: prefill-ab.sh <yaml-key> <val1> <val2> ...
set -u
KEY=$1; shift
EX=/home/dushyant30suthar/Projects/exllamav3
TB=/home/dushyant30suthar/Projects/tabbyAPI
ST=/home/dushyant30suthar/.local/state/opencode/providers/exl3
CFG=$TB/config-crown.yml
CORPUS=$(cd "$(dirname "$0")" && pwd)/corpus.txt

for VAL in "$@"; do
  echo "=== $KEY = $VAL ==="
  sed -i "/^  $KEY:/d" "$CFG"
  sed -i "/^model:/a\\  $KEY: $VAL" "$CFG"
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
  nohup "$EX/venv/bin/python" main.py --config config-crown.yml > "$ST/server.log" 2>&1 &
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
  [ $UP -eq 0 ] && { echo "  FAILED TO START"; tail -4 "$ST/server.log"; continue; }
  echo "  VRAM: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader | tr '\n' ' ')"

  python3 - "$CORPUS" <<'PY'
import json, sys, urllib.request
corpus = open(sys.argv[1]).read()
def post(path, obj, t=1800):
    req = urllib.request.Request("http://127.0.0.1:5000"+path, json.dumps(obj).encode(),
                                 {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=t))
toks = post("/v1/token/encode", {"text": corpus[:1400000]})["tokens"]
body = post("/v1/token/decode", {"tokens": toks[:98304]})["text"]
for rep in range(3):
    # unique prefix => page hashes miss => full cold prefill every time
    post("/v1/completions", {"model": "Qwen3.6-27B-exl3-5.00bpw",
                             "prompt": f"// cold run {rep} unique marker {rep*7919}\n" + body,
                             "max_tokens": 8, "temperature": 0.6, "top_k": 20})
print("  workload done", flush=True)
PY

  echo "  VRAM peak: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader | tr '\n' ' ')"
  python3 - "$ST/server.log" "$MARK" <<'PY'
import re, sys, statistics as st
raw = re.sub(r"\s+", " ", open(sys.argv[1]).read())
pat = (r"(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)[^M]*Metrics \(ID: [a-f0-9]+\): (\d+) tokens generated in "
       r"([\d.]+) seconds \(Queue: ([\d.]+) s, Process: (\d+) cached tokens and (\d+) new tokens at "
       r"([\d.]+) T/s, Generate: ([\d.]+) T/s, Context: (\d+) tokens")
rows = [m for m in re.findall(pat, raw) if m[0] >= sys.argv[2] and int(m[5]) > 50000]
if not rows:
    print("  no cold prefills recorded"); raise SystemExit
pp = sorted(float(m[6]) for m in rows)
secs = [int(m[5]) / float(m[6]) for m in rows]
print(f"  cold prefill: median {st.median(pp):>5.0f} t/s  [{pp[0]:.0f}-{pp[-1]:.0f}]  "
      f"= {st.median(secs):.0f}s for {int(rows[0][5]):,} tokens  n={len(pp)}")
PY
done
echo "PREFILL-AB-COMPLETE"

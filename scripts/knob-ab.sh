#!/bin/bash
# A/B one TabbyAPI config knob, measuring with the SERVER's own metrics.
#
# usage: knob-ab.sh <yaml-key> <value1> <value2> ...
# e.g.   knob-ab.sh draft_num_tokens 4 6 8
#
# For each value: rewrite the key (adding it under model: if absent), restart,
# run a fixed workload at three depths with realistic generation lengths, then
# read back only the Metrics lines produced during that window.
set -u
KEY=$1; shift
EX=/home/dushyant30suthar/Projects/exllamav3
TB=/home/dushyant30suthar/Projects/tabbyAPI
ST=/home/dushyant30suthar/.local/state/opencode/providers/exl3
CFG=$TB/config-crown.yml
CORPUS=$(cd "$(dirname "$0")" && pwd)/corpus.txt   # absolute: the script cd's to $TB later

for VAL in "$@"; do
  echo "=== $KEY = $VAL ==="
  # Section matters: TabbyAPI's pydantic models use extra='ignore', so a key
  # written under the wrong heading is silently dropped and the run measures
  # nothing. draft_* belongs to DraftModelConfig, everything else to ModelConfig.
  case "$KEY" in
    draft_*) ANCHOR="^draft_model:" ;;
    *)       ANCHOR="^model:" ;;
  esac
  sed -i "/^  $KEY:/d" "$CFG"                       # drop any stale copy anywhere
  sed -i "/$ANCHOR/a\\  $KEY: $VAL" "$CFG"
  "$EX/venv/bin/python" - "$CFG" "$KEY" "$VAL" <<'PY'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1])); key, val = sys.argv[2], sys.argv[3]
sec = "draft_model" if key.startswith("draft_") else "model"
got = cfg.get(sec, {}).get(key)
assert str(got) == val, f"{key} not landed in {sec}: {got!r}"
print(f"  verified: {sec}.{key} = {got}")
PY
  pkill -f "main[.]py --config" 2>/dev/null
  for i in $(seq 1 30); do pgrep -f "main[.]py --config" >/dev/null || break; sleep 2; done
  pkill -9 -f "main[.]py --config" 2>/dev/null; sleep 3
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
  [ $UP -eq 0 ] && { echo "  FAILED TO START"; tail -3 "$ST/server.log"; continue; }

  python3 - "$CORPUS" <<'PY'
import json, sys, time, urllib.request
corpus = open(sys.argv[1]).read()
def post(path, obj, t=900):
    req = urllib.request.Request("http://127.0.0.1:5000"+path, json.dumps(obj).encode(),
                                 {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=t))
toks = post("/v1/token/encode", {"text": corpus[:1200000]})["tokens"]
for depth in (4096, 49152, 98304):
    text = post("/v1/token/decode", {"tokens": toks[:depth]})["text"]
    for rep in range(2):                      # rep0 warms the cache, rep1 is the measurement
        post("/v1/completions", {"model": "Qwen3.6-27B-exl3-5.00bpw", "prompt": text,
                                 "max_tokens": 400, "temperature": 0.6, "top_k": 20,
                                 "top_p": 0.95, "min_p": 0.0})
print("  workload done", flush=True)
PY

  python3 - "$ST/server.log" "$MARK" "$VAL" <<'PY'
import re, sys, statistics as st
raw = re.sub(r"\s+", " ", open(sys.argv[1]).read())
pat = (r"(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)[^M]*Metrics \(ID: [a-f0-9]+\): (\d+) tokens generated in "
       r"([\d.]+) seconds \(Queue: ([\d.]+) s, Process: (\d+) cached tokens and (\d+) new tokens at "
       r"([\d.]+) T/s, Generate: ([\d.]+) T/s, Context: (\d+) tokens, Draft: (\d+) / (\d+) tokens "
       r"accepted \(([\d.]+)%\)\)")
rows = [m for m in re.findall(pat, raw) if m[0] >= sys.argv[2]]
real = [m for m in rows if int(m[1]) > 100]
if not real:
    print("  no measurements"); raise SystemExit
by = {}
for m in real:
    ctx = int(m[8]); bucket = min([4096, 49152, 98304], key=lambda d: abs(d - ctx))
    by.setdefault(bucket, []).append((float(m[7]), float(m[11]), int(m[10])))
for d in sorted(by):
    tg = [x[0] for x in by[d]]; acc = [x[1] for x in by[d]]; dt = [x[2] for x in by[d]]
    print(f"  ctx {d:>6}: decode {st.median(tg):>5.1f} t/s | acceptance {st.median(acc):>4.0f}% | drafted {st.median(dt):>4.0f}")
PY
done
echo "KNOB-AB-COMPLETE"

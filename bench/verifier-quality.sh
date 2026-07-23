#!/usr/bin/env bash
# verifier-quality.sh — does the S tier actually catch real defects?
#
# Speed benchmarks say nothing about whether this model is fit to be a
# source-of-truth reviewer. This feeds both quants the same 6 planted bugs
# (graded from obvious to subtle) and scores whether each is identified.
#
# Usage: ./verifier-quality.sh <model.gguf> <K> <label> [extra server flags...]
set -u
MODEL="${1:?usage: verifier-quality.sh <model.gguf> <K> <label> [flags...]}"
K="${2:?}"; LABEL="${3:?}"; shift 3
BIN="$(cd "$(dirname "$0")/.." && pwd)/llama.cpp/build/bin"
OUT="$(dirname "$0")/verifier-quality.txt"
LOG="$(dirname "$0")/verifier-quality.log"
PORT=9520

RE=$(python3 -c "
K=$K
cpu=[l for l in range(48) if not (24-K<=l<=23 or 48-K<=l<=47)]
print('blk\\\\.(' + '|'.join(map(str,cpu)) + ')\\\\.ffn_.*_exps\\\\.=CPU')")

"$BIN/llama-server" -m "$MODEL" -ngl 99 -fa on -c 98304 -ub 2048 --jinja --no-warmup \
  -ctk q8_0 -ctv q8_0 --no-mmap --cache-ram 0 --split-mode layer -ot "$RE" \
  --temp 1.0 --top-k 20 --top-p 1.0 --min-p 0 \
  --host 127.0.0.1 --port $PORT "$@" >>"$LOG" 2>&1 &
SRV=$!
up=0
for _ in $(seq 1 150); do sleep 4; kill -0 $SRV 2>/dev/null || break
  curl -s --max-time 2 http://127.0.0.1:$PORT/health | grep -q '"ok"' && { up=1; break; }; done
[ "$up" != 1 ] && { echo "$LABEL: LOAD_FAILED" >>"$OUT"; exit 1; }

python3 - "$LABEL" "$PORT" <<'PY' >>"$OUT"
import json,sys,urllib.request
label,port=sys.argv[1],sys.argv[2]

# (name, code, keywords that indicate the bug was actually found)
CASES=[
("off-by-one-infinite-loop", """def binary_search(arr, target):
    lo, hi = 0, len(arr)
    while lo < hi:
        mid = (lo + hi) // 2
        if arr[mid] == target: return mid
        elif arr[mid] < target: lo = mid
        else: hi = mid
    return -1""", ["mid + 1","mid+1","infinite","never terminat","does not terminate","hang"]),

("mutable-default-arg", """def add_item(item, basket=[]):
    basket.append(item)
    return basket""", ["mutable default","shared","default argument","none","persists between"]),

("race-condition", """import threading
count = 0
def worker():
    global count
    for _ in range(100000):
        count += 1
threads=[threading.Thread(target=worker) for _ in range(4)]
[t.start() for t in threads]; [t.join() for t in threads]""",
 ["race","not atomic","lock","gil","synchron","data race"]),

("silent-truncation", """def average(nums):
    return sum(nums) // len(nums)""", ["floor","integer division","truncat","//","float","zerodivision","empty"]),

("resource-leak", """def read_config(path):
    f = open(path)
    data = f.read()
    if not data:
        return None
    f.close()
    return data""", ["not closed","leak","early return","context manager","with open","fd"]),

("subtle-slice-bug", """def last_n(items, n):
    return items[-n:]""", ["n = 0","n==0","zero","entire list","whole list","empty"]),
]

def ask(code):
    body=json.dumps({"messages":[{"role":"user","content":
        "You are a code reviewer. Find any bug in this code and state it precisely.\n\n```python\n"+code+"\n```"}],
        "max_tokens":7000}).encode()
    req=urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
        data=body, headers={"Content-Type":"application/json"})
    r=json.load(urllib.request.urlopen(req, timeout=1200))
    m=r["choices"][0]["message"]
    return (m.get("content") or ""), (m.get("reasoning_content") or ""), r["timings"]

hits=0; total=0; secs=0.0
for name,code,keys in CASES:
    total+=1
    try:
        ans,think,t=ask(code)
    except Exception as e:
        print(f"  {label} | {name:26s} ERROR {str(e)[:40]}"); continue
    blob=(ans+" "+think).lower()
    found=any(k.lower() in blob for k in keys)
    hits+=found
    secs+=t["predicted_n"]/max(t["predicted_per_second"],1e-9)
    print(f"  {label} | {name:26s} {'FOUND' if found else 'MISSED':6s} "
          f"think={len(think):5d} ans={len(ans):5d} tok={t['predicted_n']:5d}")
print(f"  {label} | SCORE {hits}/{total}  total_gen_time={secs:.0f}s")
PY
kill $SRV 2>/dev/null; wait $SRV 2>/dev/null

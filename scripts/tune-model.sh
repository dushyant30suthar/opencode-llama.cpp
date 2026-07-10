#!/usr/bin/env bash
# Auto-tuner: find the fastest llama-server config for a model on THIS machine
# and make it the default that opencode's /config "Reset to recommended" applies.
#
# Method: greedy coordinate descent over the speed-relevant knobs, measured the
# same way as every number in docs/tuning.md — a real llama-server (port 9444),
# a ~600-token code-generation request, timings.predicted_per_second.
#   phase 1: KV cache type   (q8_0, f16)
#   phase 2: ubatch          (1024, 2048)
#   phase 3: split-mode      (tensor, layer)
#   phase 4: MTP draft n-max (2, 3, 4, 5)   [only if the section uses draft-mtp]
# ~7 server loads, ≈20 min. Winners are written to
# ~/.local/state/llamastack/recommended.ini — opencode reads that file, so
# "Reset to recommended" now means "reset to what benchmarks won here".
# ctx-size is NOT tuned here (see bench/ctx-search-v2.sh for the max-ctx probe);
# measurement runs at 32k ctx for fast loads — decode speed is ctx-insensitive.
#
# Usage:
#   ./scripts/tune-model.sh <models.ini section name>       # e.g. ./scripts/tune-model.sh michaelw9999/Qwen3.6-27B-NVFP4-MTP-GGUF
#   ./scripts/tune-model.sh <section> --apply               # also write winners into models.ini itself
#
# Needs idle GPUs: unload the router's model first (opencode: pick nothing /
# restart router; or curl "http://127.0.0.1:9337/models?reload=1" after editing).
set -u
SECTION="${1:?usage: tune-model.sh <models.ini section name> [--apply]}"
APPLY="${2:-}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRV="$ROOT/llama.cpp/build/bin/llama-server"
STATE=~/.local/state/llamastack
INI="$STATE/models.ini"
REC="$STATE/recommended.ini"
OUT="$STATE/tune-$(echo "$SECTION" | tr '/:' '__').txt"
LOG="$STATE/tune.log"
PORT=9444
[ -x "$SRV" ] || { echo "no llama-server at $SRV — run scripts/build-llama.sh"; exit 1; }

MODEL=$(awk -v s="[$SECTION]" '$0==s{f=1;next} /^\[/{f=0} f && $1=="model"{print $3}' "$INI")
[ -n "$MODEL" ] && [ -f "${MODEL/#\~/$HOME}" ] || { echo "section [$SECTION] with a model file not found in $INI"; exit 1; }
MODEL="${MODEL/#\~/$HOME}"
MTP=$(awk -v s="[$SECTION]" '$0==s{f=1;next} /^\[/{f=0} f && $1=="spec-type"{print $3}' "$INI")

FREE0=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
[ "$FREE0" -lt 2000 ] || { echo "GPU0 has ${FREE0}MiB in use — unload the router's model first"; exit 1; }

: > "$OUT"; : > "$LOG"
# ~2k-token prompt so prompt_per_second is a real prefill number too
PROMPT=$(python3 -c "print(('Refactor this. def f(x):\n  return [i*i for i in range(x) if i%3]\n' * 120).replace(chr(10),' '))")

measure() { # measure <label> <extra server args...> -> "tg pp"
  local label="$1"; shift
  "$SRV" -m "$MODEL" -ngl 99 -fa on -c 32768 --jinja --host 127.0.0.1 --port $PORT "$@" >>"$LOG" 2>&1 &
  local pid=$! up=0
  for _ in $(seq 1 80); do
    sleep 3; kill -0 $pid 2>/dev/null || break
    curl -s --max-time 2 http://127.0.0.1:$PORT/health | grep -q '"ok"' && { up=1; break; }
  done
  if [ "$up" != 1 ]; then echo "$label: LOAD_FAILED" | tee -a "$OUT"; kill $pid 2>/dev/null; wait $pid 2>/dev/null; echo "0 0"; return; fi
  curl -s --max-time 120 http://127.0.0.1:$PORT/v1/chat/completions -H "Content-Type: application/json" \
    -d '{"messages":[{"role":"user","content":"hi"}],"max_tokens":16}' >/dev/null
  local timings
  timings=$(python3 - "$PROMPT" <<'PY'
import json,sys,urllib.request
body=json.dumps({"messages":[{"role":"user","content":sys.argv[1]+" Write a Python class implementing an LRU cache with get, put, delete, plus unit tests."}],"max_tokens":600,"temperature":0}).encode()
r=urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:9444/v1/chat/completions",body,{"Content-Type":"application/json"}),timeout=300)
t=json.load(r).get("timings",{})
print(round(t.get("predicted_per_second",0),1), round(t.get("prompt_per_second",0),1))
PY
  ) || timings="0 0"
  echo "$label: tg=$(echo "$timings" | cut -d' ' -f1) t/s  pp=$(echo "$timings" | cut -d' ' -f2) t/s" | tee -a "$OUT"
  kill $pid 2>/dev/null; wait $pid 2>/dev/null; sleep 2
  echo "$timings"
}

tg_of() { echo "$1" | cut -d' ' -f1; }
better() { python3 -c "import sys; sys.exit(0 if float('$1') > float('$2') else 1)"; }

# greedy descent, starting from the current stack defaults
KV="q8_0"; UB="2048"; SM="tensor"; NMAX="3"
mtp_args() { [ "$MTP" = "draft-mtp" ] && echo "--spec-type draft-mtp --spec-draft-n-max $NMAX -np 1" || true; }

echo "== tuning [$SECTION] ($MODEL)" | tee -a "$OUT"
BASE=$(measure "kv=$KV ub=$UB sm=$SM n=$NMAX" -ctk $KV -ctv $KV -ub $UB -sm $SM $(mtp_args))
for kv in f16; do
  R=$(measure "kv=$kv ub=$UB sm=$SM n=$NMAX" -ctk $kv -ctv $kv -ub $UB -sm $SM $(mtp_args))
  better "$(tg_of "$R")" "$(tg_of "$BASE")" && { KV=$kv; BASE=$R; }
done
for ub in 1024; do
  R=$(measure "kv=$KV ub=$ub sm=$SM n=$NMAX" -ctk $KV -ctv $KV -ub $ub -sm $SM $(mtp_args))
  better "$(tg_of "$R")" "$(tg_of "$BASE")" && { UB=$ub; BASE=$R; }
done
for sm in layer; do
  R=$(measure "kv=$KV ub=$UB sm=$sm n=$NMAX" -ctk $KV -ctv $KV -ub $UB -sm $sm $(mtp_args))
  better "$(tg_of "$R")" "$(tg_of "$BASE")" && { SM=$sm; BASE=$R; }
done
if [ "$MTP" = "draft-mtp" ]; then
  for n in 2 4 5; do
    NM_SAVE=$NMAX; NMAX=$n
    R=$(measure "kv=$KV ub=$UB sm=$SM n=$n" -ctk $KV -ctv $KV -ub $UB -sm $SM $(mtp_args))
    if better "$(tg_of "$R")" "$(tg_of "$BASE")"; then BASE=$R; else NMAX=$NM_SAVE; fi
  done
fi

echo "== winner: kv=$KV ub=$UB sm=$SM$([ "$MTP" = "draft-mtp" ] && echo " n-max=$NMAX") -> $(tg_of "$BASE") t/s" | tee -a "$OUT"

python3 - "$SECTION" "$KV" "$UB" "$SM" "$NMAX" "$MTP" "$(tg_of "$BASE")" "$REC" <<'PY'
import re, sys, time
section, kv, ub, sm, nmax, mtp, tg, rec = sys.argv[1:9]
keys = [f"cache-type-k = {kv}", f"cache-type-v = {kv}", f"ubatch-size = {ub}", f"split-mode = {sm}"]
if mtp == "draft-mtp": keys.append(f"spec-draft-n-max = {nmax}")
block = f"# tuned by scripts/tune-model.sh: {tg} t/s decode\n[{section}]\n" + "\n".join(keys) + "\n"
try: text = open(rec).read()
except FileNotFoundError: text = "# machine-benchmarked defaults — written by scripts/tune-model.sh,\n# read by opencode /config 'Reset to recommended'. Safe to delete.\n"
pattern = re.compile(r"(^#[^\n]*\n)?^\[" + re.escape(section) + r"\]\n(?:(?!^\[).*\n?)*", re.M)
text = pattern.sub("", text).rstrip() + "\n\n" + block
open(rec, "w").write(text)
print(f"written to {rec}")
PY

if [ "$APPLY" = "--apply" ]; then
  python3 - "$SECTION" "$KV" "$UB" "$SM" "$NMAX" "$MTP" "$INI" <<'PY'
import re, sys
section, kv, ub, sm, nmax, mtp, ini = sys.argv[1:8]
want = {"cache-type-k": kv, "cache-type-v": kv, "ubatch-size": ub, "split-mode": sm}
if mtp == "draft-mtp": want["spec-draft-n-max"] = nmax
lines = open(ini).read().split("\n"); out=[]; active=False
for line in lines:
    m = re.match(r"^\s*\[([^\]]+)\]\s*$", line)
    if m: active = m.group(1) == section
    elif active:
        kvm = re.match(r"^\s*([^#;=\s][^=]*?)\s*=", line)
        if kvm and kvm.group(1) in want:
            line = f"{kvm.group(1)} = {want.pop(kvm.group(1))}"
    out.append(line)
open(ini, "w").write("\n".join(out))
print("applied to models.ini (reload: curl 'http://127.0.0.1:9337/models?reload=1')")
PY
fi

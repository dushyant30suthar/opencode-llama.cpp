#!/usr/bin/env bash
# q27-pcie-probe.sh — settle whether PCIe Gen1 x8/x4 is a real limit or idle
# down-training, and get a clean tg baseline for the 27B NVFP4 + MTP.
set -u
BIN="$(cd "$(dirname "$0")/.." && pwd)/llama.cpp/build/bin"
M=~/.lmstudio/models/michaelw9999/Qwen3.6-27B-NVFP4-MTP-GGUF/Qwen3.6-27B-NVFP4-MTP-GGUF.gguf
PORT=9477
OUT="$(dirname "$0")/q27-pcie-probe.txt"
LOG="$(dirname "$0")/q27-pcie-probe.log"
: > "$OUT"; : > "$LOG"

echo "== PCIe at IDLE ==" >>"$OUT"
nvidia-smi --query-gpu=index,pcie.link.gen.current,pcie.link.width.current,utilization.gpu --format=csv,noheader >>"$OUT"

"$BIN/llama-server" -m "$M" -ngl 99 -fa on -c 32768 -ub 2048 --jinja --no-warmup \
  -ctk q8_0 -ctv q8_0 -sm tensor --cache-ram 0 \
  --spec-type draft-mtp --spec-draft-n-max 3 \
  --host 127.0.0.1 --port $PORT >>"$LOG" 2>&1 &
SRV=$!
for _ in $(seq 1 120); do
  sleep 3; kill -0 $SRV 2>/dev/null || break
  curl -s --max-time 2 http://127.0.0.1:$PORT/health | grep -q '"ok"' && break
done

# sample PCIe while a long generation runs
( for i in $(seq 1 12); do
    nvidia-smi --query-gpu=index,pcie.link.gen.current,pcie.link.width.current,utilization.gpu,utilization.memory \
      --format=csv,noheader | paste -sd' | ' - >>"$OUT.samples"
    sleep 2
  done ) &
SAMP=$!

R=$(curl -s --max-time 900 http://127.0.0.1:$PORT/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Write a detailed technical explanation of how a B-tree index works in a relational database, including insertion, splitting, and range scans."}],"max_tokens":400}')
wait $SAMP 2>/dev/null

echo "== PCIe UNDER LOAD (gen,width,gpu%,mem%) ==" >>"$OUT"
cat "$OUT.samples" 2>/dev/null >>"$OUT"; rm -f "$OUT.samples"
echo "== baseline timings ==" >>"$OUT"
echo "$R" | jq -r '"tg=\(.timings.predicted_per_second) t/s  pp=\(.timings.prompt_per_second) t/s  n_pred=\(.timings.predicted_n)"' >>"$OUT"
echo "== draft acceptance (from server log) ==" >>"$OUT"
grep -iE "draft acceptance|n_drafted|n_accept|accept" "$LOG" | tail -5 >>"$OUT"

kill $SRV 2>/dev/null; wait $SRV 2>/dev/null
cat "$OUT"

#!/bin/bash
# EXL3 layer-split vs tensor-parallel sweep, both with MTP: patched exllamav3, MTP working. Layer-split MTP sweep + agents,
# then TP+MTP (never run anywhere before). Router restored at the end.
OUT=/home/dushyant30suthar/Projects/opencode-localhost
RESTORE=/tmp/claude-1000/-home-dushyant30suthar-Projects-opencode-localhost/af5f9353-9954-46ff-aba2-9bd7fdd6f16d/scratchpad/RESTORE.sh
PY=/home/dushyant30suthar/Projects/exllamav3/venv/bin/python

launch() {  # config, logname
  cd /home/dushyant30suthar/Projects/tabbyAPI
  pkill -9 -f "main[.]py --config" 2>/dev/null; sleep 3
  nohup $PY main.py --config "$1" > "$OUT/$2" 2>&1 &
  local pid=$!
  for i in $(seq 1 50); do
    curl -s --max-time 2 http://127.0.0.1:5000/health 2>/dev/null | grep -qi "healthy\|ok" && { echo "HEALTHY($1)"; return 0; }
    kill -0 $pid 2>/dev/null || { echo "DIED($1):"; tail -4 "$OUT/$2"; return 1; }
    sleep 5
  done
  return 1
}

# Phase A: layer-split + MTP (config-dbg.yml = known-good 4-slot config, draft mtp)
if launch config-dbg.yml tabby-layersplit-mtp.log; then
  cd "$OUT"
  python3 sweepx.py http://127.0.0.1:5000 exl3-LS-MTP-FIXED sweep exl3-results.jsonl > "$OUT/layersplit-mtp-sweep.log" 2>&1
  echo "A-sweep: $(tail -1 $OUT/layersplit-mtp-sweep.log)"
  python3 sweepx.py http://127.0.0.1:5000 exl3-LS-MTP-FIXED agents exl3-results.jsonl > "$OUT/layersplit-mtp-agents.log" 2>&1
  echo "A-agents: $(tail -1 $OUT/layersplit-mtp-agents.log)"
fi

# Phase B: TP + MTP
cd /home/dushyant30suthar/Projects/tabbyAPI
sed 's/^  tensor_parallel: false$/  tensor_parallel: true/' config-dbg.yml > config-tp-mtp.yml
if launch config-tp-mtp.yml tabby-tp-mtp.log; then
  cd "$OUT"
  python3 sweepx.py http://127.0.0.1:5000 exl3-TP-MTP sweep exl3-results.jsonl > "$OUT/tp-mtp-sweep.log" 2>&1
  echo "B-sweep: $(tail -1 $OUT/tp-mtp-sweep.log)"
  python3 sweepx.py http://127.0.0.1:5000 exl3-TP-MTP agents exl3-results.jsonl > "$OUT/tp-mtp-agents.log" 2>&1
  echo "B-agents: $(tail -1 $OUT/tp-mtp-agents.log)"
else
  echo "TP+MTP failed to launch"
fi

pkill -9 -f "main[.]py --config" 2>/dev/null; sleep 3
bash "$RESTORE"
echo "SWEEP-COMPLETE"

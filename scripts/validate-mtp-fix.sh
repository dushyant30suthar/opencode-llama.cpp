#!/bin/bash
# Validation gauntlet for the eos-beats-requeue fix. Assumes GPUs free.
cd /home/dushyant30suthar/Projects/tabbyAPI
pkill -9 -f "main[.]py --config" 2>/dev/null; sleep 3
nohup /home/dushyant30suthar/Projects/exllamav3/venv/bin/python main.py --config config-dbg.yml \
  > /home/dushyant30suthar/Projects/opencode-localhost/tabby-dbg.log 2>&1 &
TPID=$!; echo $TPID > /tmp/tabby.pid
for i in $(seq 1 50); do
  curl -s --max-time 2 http://127.0.0.1:5000/health 2>/dev/null | grep -qi "healthy\|ok" && { echo HEALTHY; break; }
  kill -0 $TPID 2>/dev/null || { echo DIED-AT-LOAD; tail -5 /home/dushyant30suthar/Projects/opencode-localhost/tabby-dbg.log; exit 1; }
  sleep 5
done

probe() {  # prompt, max_tokens
  curl -s --max-time 90 http://127.0.0.1:5000/v1/completions -H 'Content-Type: application/json' \
    -d "{\"model\":\"Qwen3.6-27B-exl3-5.00bpw\",\"prompt\":$(python3 -c "import json,sys;print(json.dumps(sys.argv[1]))" "$1"),\"max_tokens\":$2,\"temperature\":0.6,\"top_k\":20}" \
    | python3 -c "import json,sys; d=json.load(sys.stdin); print('ok' if 'choices' in d else 'FAIL')" 2>&1
}

F=0; for n in $(seq 1 8); do R=$(probe "def onetok_$n(a):" 1); [ "$R" != "ok" ] && F=$((F+1)); done
echo "PHASE1 one-token x8: FAILS=$F"

F=0; for n in $(seq 1 12); do R=$(probe "def distinct_$n(a, b):" 24); [ "$R" != "ok" ] && F=$((F+1)); done
echo "PHASE2 distinct x12: FAILS=$F"

P="You are a coding assistant."
F=0; for n in $(seq 1 10); do
  P="$P Turn $n: write function number $n."
  R=$(probe "$P" 32); [ "$R" != "ok" ] && F=$((F+1))
done
echo "PHASE3 growing-conversation x10: FAILS=$F"
echo "GAUNTLET-DONE"

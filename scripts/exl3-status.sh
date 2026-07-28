#!/bin/bash
# EXL3 pilot status — run anytime: ! ~/Projects/opencode-localhost/exl3-status.sh
M=/home/dushyant30suthar/.lmstudio/models/turboderp
echo "=== DOWNLOAD (main ~17G @5bpw, MTP head ~1-2G) ==="
du -sh $M/Qwen3.6-27B-exl3-5.00bpw $M/Qwen3.6-27B-MTP-exl3 2>/dev/null || echo "  not started"
pgrep -f "hf_transfer\|snapshot_dow" >/dev/null 2>&1 && echo "  downloader: RUNNING" || \
  { ls $M/Qwen3.6-27B-exl3-5.00bpw/*.safetensors >/dev/null 2>&1 && echo "  downloader: done or stopped"; }
echo "=== ENV ==="
if [ -d /home/dushyant30suthar/Projects/tabbyAPI/venv ]; then
  /home/dushyant30suthar/Projects/tabbyAPI/venv/bin/python -c "import exllamav3,torch; print('  exllamav3', exllamav3.__version__, '| torch', torch.__version__, '| cuda ok:', torch.cuda.is_available())" 2>/dev/null || echo "  venv exists, exllamav3 not ready yet"
else
  echo "  venv not created yet"
fi
echo "=== SERVER ==="
pgrep -af "tabby\|main.py" 2>/dev/null | grep -v grep | head -2 || echo "  tabbyAPI not running"
curl -s --max-time 2 http://127.0.0.1:5000/health 2>/dev/null && echo "  :5000 healthy"
echo "=== BENCH ==="
tail -3 /home/dushyant30suthar/Projects/opencode-localhost/exl3-results.jsonl 2>/dev/null || echo "  no results yet"

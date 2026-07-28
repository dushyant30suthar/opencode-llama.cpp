#!/bin/bash
# Self-healing EXL3 download: standard HF backend (timeouts+retries), outer retry loop.
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HUB_DOWNLOAD_TIMEOUT=15
for i in $(seq 1 30); do
  python3 - <<'EOF' && break
from huggingface_hub import snapshot_download
p1 = snapshot_download("turboderp/Qwen3.6-27B-exl3", revision="5.00bpw",
                       local_dir="/home/dushyant30suthar/.lmstudio/models/turboderp/Qwen3.6-27B-exl3-5.00bpw",
                       max_workers=4)
p2 = snapshot_download("turboderp/Qwen3.6-27B-MTP-exl3",
                       local_dir="/home/dushyant30suthar/.lmstudio/models/turboderp/Qwen3.6-27B-MTP-exl3",
                       max_workers=4)
print("EXL3-DOWNLOAD-COMPLETE:", p1, p2)
EOF
  echo "attempt $i failed, retrying in 10s..."; sleep 10
done

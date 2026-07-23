#!/usr/bin/env bash
# verifier-setup.sh — one-shot: download whatever verifier-tier files are
# missing and wire them into the running opencode/llama.cpp router.
# Idempotent + resumable: partial downloads continue, complete ones are
# skipped, config is never duplicated. Interrupt and re-run as often as
# you like (network switches are fine).

MODELS="$HOME/.lmstudio/models"
INI="$HOME/.local/state/llamastack/models.ini"
ROUTER="http://127.0.0.1:9337"
W=(wget -4 -c -t 0 --retry-connrefused --timeout=30 -q --show-progress)

echo "== [1/4] MiniMax M2.5 verifier — missing shard 2 of 3 (47.5GB)"
"${W[@]}" -O "$MODELS/UD-IQ2_XXS/MiniMax-M2.5-UD-IQ2_XXS-00002-of-00003.gguf" \
  "https://huggingface.co/unsloth/MiniMax-M2.5-GGUF/resolve/main/UD-IQ2_XXS/MiniMax-M2.5-UD-IQ2_XXS-00002-of-00003.gguf" \
  || { echo "!! shard 2 interrupted — re-run this script to resume"; exit 1; }

echo "== [2/4] Devstral Small 2 24B cross-checker (14.5GB)"
mkdir -p "$MODELS/unsloth/Devstral-Small-2-24B-Instruct-2512-GGUF"
"${W[@]}" -O "$MODELS/unsloth/Devstral-Small-2-24B-Instruct-2512-GGUF/Devstral-Small-2-24B-Instruct-2512-UD-Q4_K_XL.gguf" \
  "https://huggingface.co/unsloth/Devstral-Small-2-24B-Instruct-2512-GGUF/resolve/main/Devstral-Small-2-24B-Instruct-2512-UD-Q4_K_XL.gguf" \
  || { echo "!! devstral interrupted — re-run this script to resume"; exit 1; }

echo "== [3/4] models.ini — add Devstral section if absent"
if ! grep -q "Devstral-Small-2-24B" "$INI"; then
cat >> "$INI" <<EOF

[unsloth/Devstral-Small-2-24B-Instruct-2512-GGUF]
model = $MODELS/unsloth/Devstral-Small-2-24B-Instruct-2512-GGUF/Devstral-Small-2-24B-Instruct-2512-UD-Q4_K_XL.gguf
# verifier/cross-checker: different lab (Mistral), dense 24B, 256K native
ctx-size = 131072
gpu-layers = 99
flash-attn = on
cache-type-k = q8_0
cache-type-v = q8_0
jinja = true
split-mode = layer
# Mistral-recommended sampler for Devstral
temp = 0.15
EOF
  echo "   section appended"
else
  echo "   section already present"
fi

echo "== [4/4] reload router config"
curl -s "$ROUTER/models?reload=1" | python3 -c "
import json,sys
d = json.load(sys.stdin)
for m in d['data']:
    if 'Devstral' in m['id'] or 'IQ2_XXS' in m['id']:
        print('   router sees:', m['id'])" || echo "   (router not running — will pick up on next start)"

echo
echo "ALL DONE. Verifier tier ready:"
ls -lh "$MODELS/UD-IQ2_XXS/" "$MODELS/unsloth/Devstral-Small-2-24B-Instruct-2512-GGUF/" | grep -v total
echo "Test MiniMax:  opencode → select UD-IQ2_XXS (expect ~2-4 t/s, that's normal)"
echo "Test Devstral: opencode → select unsloth/Devstral-Small-2-24B-Instruct-2512-GGUF"

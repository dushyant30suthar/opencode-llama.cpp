#!/usr/bin/env bash
# Build the opencode fork into a single binary and install it to ~/.opencode/bin.
#
# Usage:
#   ./scripts/build-opencode.sh [path-to-opencode]   # default: ./opencode submodule
set -euo pipefail

OPENCODE_DIR="${1:-$(cd "$(dirname "$0")/.." && pwd)/opencode}"
[[ -f "$OPENCODE_DIR/package.json" ]] || { echo "error: no opencode checkout at $OPENCODE_DIR (run: git submodule update --init opencode)"; exit 1; }
command -v bun >/dev/null || { echo "error: bun is required (https://bun.sh)"; exit 1; }

cd "$OPENCODE_DIR"
bun install --ignore-scripts
bun run --cwd packages/core fix-node-pty
cd packages/opencode
bun run script/build.ts --single --skip-embed-web-ui

BIN="dist/opencode-linux-x64/bin/opencode"
install -Dm755 "$BIN" "$HOME/.opencode/bin/opencode"
echo
echo "installed: ~/.opencode/bin/opencode ($("$HOME/.opencode/bin/opencode" --version 2>/dev/null || echo 'version check skipped'))"

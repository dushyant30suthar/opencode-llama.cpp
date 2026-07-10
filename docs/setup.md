# Setup — from bare GPU box to working stack

Four stages: CUDA toolkit → llama.cpp build → opencode fork build → models.

## 1. CUDA toolkit (13.3 or newer — never 13.2)

Two hard-won rules:

> **Never build with CUDA 13.2/13.2.x.** A confirmed nvcc bug miscompiles the
> quantization kernels — models load fine and emit gibberish
> ([llama.cpp #21255](https://github.com/ggml-org/llama.cpp/issues/21255)).

> **Install the latest toolkit from NVIDIA directly if your distro lags.**
> This stack exists because prebuilt binaries (LM Studio) ran PTX-fallback
> kernels on Blackwell — the toolkit that knows your architecture natively is
> worth the manual install.

On Fedora, NVIDIA's own repo carries the current toolkit:

```sh
sudo dnf config-manager addrepo --from-repofile=https://developer.download.nvidia.com/compute/cuda/repos/fedora41/x86_64/cuda-fedora41.repo
sudo dnf install cuda-toolkit
```

The toolkit lands in `/usr/local/cuda-<version>`. Multiple versions coexist;
`scripts/build-llama.sh` picks the newest and refuses 13.2.

**Host compiler:** nvcc supports host GCC only up to a point (CUDA 13.3 →
GCC 15). If your distro's default GCC is newer, install the versioned package
(Fedora: `gcc15-c++`) — the build script auto-detects `g++-15`.

The GPU **driver** is separate from the toolkit and usually distro-managed
(RPM Fusion on Fedora); any recent driver works.

## 2. Build llama.cpp

```sh
./scripts/build-llama.sh            # builds ./llama.cpp (the submodule)
./scripts/build-llama.sh ~/src/llama.cpp   # or any other checkout
```

What it encodes (see the script for the full story):

- **Refuses CUDA 13.2**, picks the newest `/usr/local/cuda-*` otherwise
  (override with `CUDA_HOME`).
- **`rm -rf build` first** — GGML_LTO caches stale objects; incremental
  rebuilds after a compiler or flag change produce broken binaries.
- Native GPU arch (`CMAKE_CUDA_ARCHITECTURES=native`, override with
  `CUDA_ARCH=120` etc.), flash-attention kernels for all KV quants, CUDA
  graphs, LTO, Release.

The binary ends up at `<checkout>/build/bin/llama-server`.

## 3. Build and install the opencode fork

Requires [bun](https://bun.sh).

```sh
./scripts/build-opencode.sh         # builds ./opencode (the submodule)
```

This produces a single self-contained binary and installs it to
`~/.opencode/bin/opencode`. Fork builds never self-update to stock upstream
(that would silently remove the whole local stack) — update by rebasing and
rebuilding, see [features/no-self-update.md](features/no-self-update.md).

### Updating the opencode fork later

The fork's branch (`opencode-llama.cpp`) sits on top of upstream `dev`
(sst/opencode's main branch). To pick up upstream improvements:

```sh
cd opencode
git fetch upstream                  # upstream = github.com/sst/opencode
git rebase upstream/dev
bun install --ignore-scripts
bun run --cwd packages/core fix-node-pty
cd packages/opencode && bun run script/build.ts --single --skip-embed-web-ui
cp dist/opencode-linux-x64/bin/opencode ~/.opencode/bin/opencode
```

Keeping a stock upstream binary at `~/.opencode/bin/opencode-upstream-backup`
is handy for A/B-ing fork regressions.

## 4. Models

Models live in the **LM Studio layout**: `~/.lmstudio/models/<publisher>/<repo>/*.gguf`.
Anything LM Studio downloads appears in opencode automatically; or use the
self-resuming downloaders in `scripts/` as templates
(`download-models.sh`, `download-mtp.sh`).

## 5. First run

```sh
opencode
```

On startup the fork probes for a local server and, finding none, spawns
`llama-server` in router mode itself — models show up in the model picker at
zero cost. Per-model settings are generated into
`~/.local/state/llamastack/models.ini`, which is **yours to edit** and never
overwritten. Start from [`config/models.ini.example`](../config/models.ini.example)
to see what a fully tuned file looks like.

### Paths and overrides

| Env var | Meaning | Default |
| --- | --- | --- |
| `LLAMASTACK_SERVER_BIN` | llama-server binary | `$PATH`, then conventional build dirs (`~/Projects/llama/llama.cpp/build/bin`, `~/Projects/llama.cpp/build/bin`, `~/llama.cpp/build/bin`, `/usr/local/bin`) |
| `LLAMASTACK_MODELS_DIR` | models directory | `~/.lmstudio/models` |
| `OPENCODE_DISABLE_LLAMASTACK` | set to `1` to skip local-stack detection entirely | unset |

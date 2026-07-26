# Setup — from bare GPU box to working stack

Four stages: CUDA toolkit → llama.cpp build → opencode + plugin → models.

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

## 3. opencode + the local-models plugin

opencode is installed normally — there is no fork any more:

```bash
curl -fsSL https://opencode.ai/install | bash
```

Then point it at [opencode-localhost](https://github.com/dushyant30suthar/opencode-localhost),
which registers llama.cpp as a provider and draws the hardware panel:

```jsonc
// ~/.config/opencode/opencode.jsonc
{ "plugin": ["opencode-localhost"] }

// ~/.config/opencode/tui.jsonc
{ "plugin": ["opencode-localhost"] }
```

On first launch it writes `~/.config/opencode/providers/llamacpp/server.ini`.
Set `bin` to the `llama-server` built in stage 2 and `models-dir` to your
models directory:

```ini
[server]
bin        = ~/Projects/opencode-llama.cpp/llama.cpp/build/bin/llama-server
models-dir = ~/.lmstudio/models
```

The panel picks the change up within a couple of seconds — no restart.


## 4. Models

Models live in the **LM Studio layout**: `~/.lmstudio/models/<publisher>/<repo>/*.gguf`.
Anything LM Studio downloads appears in opencode automatically; or use the
self-resuming downloaders in `scripts/` as templates
(`download-models.sh`, `download-mtp.sh`).

## 5. First run

```sh
opencode
```

The plugin scans `models-dir`, spawns `llama-server` against a generated
preset, and registers the provider — models show up in the picker with their
real context windows. Per-model settings land in
`~/.config/opencode/providers/llamacpp/models.ini`, which is **yours to edit**
and never overwritten. Start from
[`config/models.ini.example`](../config/models.ini.example) to see what a fully
tuned file looks like.

### Paths

| What | Where |
| --- | --- |
| server settings | `~/.config/opencode/providers/llamacpp/server.ini` |
| per-model settings | `~/.config/opencode/providers/llamacpp/models.ini` |
| pidfile and log | `~/.local/state/opencode/providers/llamacpp/` |

`bin` and `models-dir` live in `server.ini` — there are no environment
variables to remember, and the file always shows what is actually in use.

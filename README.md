# opencode-llama.cpp

A complete local-AI coding stack: [llama.cpp](https://github.com/ggml-org/llama.cpp)
built natively for your GPU, driven by an [opencode](https://github.com/sst/opencode)
fork with a built-in zero-config local provider — plus the build recipes, tuned
per-model configs, and benchmark harness that make it fast.

The goal: **hand a local model a real ticket, get a mergeable branch** — a
reliable coding-agent stack running entirely on consumer hardware.

## Why build from source at all

Prebuilt stacks leave a lot on the table. On Blackwell GPUs (sm_120), LM Studio
shipped PTX-fallback CUDA kernels; a native CUDA 13.3 build of llama.cpp was
measured **~2.5× faster** on the same hardware. Squeezing a consumer box means
owning the whole chain:

1. **Latest CUDA toolkit** — installed straight from NVIDIA when the distro
   lags. 13.3+ is mandatory: 13.2 nvcc miscompiles quantization kernels into
   gibberish ([llama.cpp #21255](https://github.com/ggml-org/llama.cpp/issues/21255)).
2. **llama.cpp compiled for your exact GPU arch** with tuned CMake flags.
3. **Per-model configs probed, not guessed** — max context found by binary
   search through the real server, split-mode/ubatch chosen by benchmark sweep,
   MTP speculative decoding tuned by measurement.
4. **A frontend that understands local models** — the opencode fork adds model
   management, live load/prefill progress, thinking preservation for agentic
   loops, LAN hosting, and more. See [docs/features](docs/features/README.md).

## Repository layout

| Path | What it is |
| --- | --- |
| `llama.cpp/` | Submodule → [our llama.cpp fork](https://github.com/dushyant30suthar/llama.cpp) (tracks upstream ggml-org), pinned at the commit the configs were validated against |
| `opencode/` | Submodule → [our opencode fork](https://github.com/dushyant30suthar/opencode) (branch `opencode-llama.cpp`), where all stack code lives |
| `docs/` | Common documentation: [setup](docs/setup.md), [architecture](docs/architecture.md), [tuning](docs/tuning.md), [features](docs/features/README.md) |
| `scripts/` | `build-llama.sh`, `build-opencode.sh`, model downloaders — the build knowledge as executable fact |
| `config/` | `models.ini.example` — the tuned per-model settings file, documented |
| `bench/` | The experiment harness and raw results behind every number in the docs |

## Quick start

```sh
git clone --recursive https://github.com/dushyant30suthar/opencode-llama.cpp
cd opencode-llama.cpp

# 1. CUDA toolkit 13.3+ and a supported host compiler — see docs/setup.md
# 2. Build the engine (checks the CUDA version for you)
./scripts/build-llama.sh
# 3. Build + install the opencode fork
./scripts/build-opencode.sh
# 4. Put GGUF models in ~/.lmstudio/models (LM Studio layout), then:
opencode
```

Local models appear in the model picker under "Llama Stack (local)". No config
files, no API keys. Full walkthrough: [docs/setup.md](docs/setup.md).

## Reference rig

All numbers in these docs come from one machine — treat them as a calibrated
example, not a promise:

- 2× RTX 5060 Ti 16 GB (Blackwell sm_120), no GPU P2P → `split-mode = tensor`
- i5-9400F, 32 GB DDR4, Fedora, CUDA 13.3.1, host g++-15

Highlights ([full tables](docs/tuning.md)):

| Model | Max context | Generation |
| --- | --- | --- |
| **Qwen3.6-27B NVFP4-MTP (champion)** | 208,896 (probed) | **76.3 t/s** (MTP draft n=3) |
| Qwen3.6-27B-MTP Q4_K_XL | 180,224 | 70.6 t/s (MTP draft n=4) |
| Qwen3.6-35B-A3B Q4_K_M (MoE) | 245,760 | 152 t/s |
| Qwen3.6-27B Q4_K_M | 258,048 (no mmproj) | 40 t/s |
| gemma-4-31B QAT Q4_0 | 147,456 | 37 t/s |

## Keeping it current

Development never happens in this repo — it happens in the two forks, each an
independent, upstream-connected repo. This repo pins the versions that are
proven to work together.

The easy way: **`opencode upgrade`** — on fork builds it checks both
upstreams, proves the sync is conflict-free, then pulls, compiles, and swaps
the binaries itself. `opencode upgrade --check` for a read-only status.
Full details and the manual fallback: [docs/upgrading.md](docs/upgrading.md).

- **opencode fork** — new features land on branch `opencode-llama.cpp`; rebase onto
  upstream `sst/opencode` to pick up their improvements, rebuild with
  `scripts/build-opencode.sh`.
- **llama.cpp fork** — carries no patches today, so updating is just syncing
  the fork with upstream (`gh repo sync <you>/llama.cpp --source ggml-org/llama.cpp`),
  rebuilding with `scripts/build-llama.sh`, and re-running the `bench/` probes
  if the release notes touch your model families. If a patch is ever needed,
  it lives on the fork the same way the opencode ones do.

When a new combination is validated, bump the submodule pins here in one
commit — this repo is the record of "these versions work together."

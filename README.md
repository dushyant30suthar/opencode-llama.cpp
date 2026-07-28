# opencode-llama.cpp

The rig behind a local-AI coding stack: [llama.cpp](https://github.com/ggml-org/llama.cpp)
built natively for your GPU, plus the build recipes, tuned per-model configs,
and benchmark harness that make it fast.

The opencode side lives in its own repo now —
[**opencode-localhost**](https://github.com/dushyant30suthar/opencode-localhost),
an npm plugin that registers llama.cpp as a provider and shows live GPU/VRAM/CPU
in the sidebar. No fork of opencode, no patches.

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
4. **A frontend that understands local models** —
   [opencode-localhost](https://github.com/dushyant30suthar/opencode-localhost)
   discovers your GGUFs, supervises `llama-server`, and puts hardware telemetry
   on screen.

## Repository layout

| Path | What it is |
| --- | --- |
| `llama.cpp/` | Submodule → [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp), pinned at the commit the configs were validated against |
| `docs/` | [setup](docs/setup.md), [tuning](docs/tuning.md), [inference flag map](docs/inference-flag-map.md), [Laguna S retune](docs/laguna-retune-2026-07-27.md), [27B at production depth](docs/q27b-production-depth-2026-07-28.md) |
| `scripts/` | `build-llama.sh`, model downloaders, the tuner — the build knowledge as executable fact |
| `config/` | `models.ini.example` — the tuned per-model settings file, documented |
| `bench/` | The experiment harness and raw results behind every number in the docs |

## Quick start

```sh
git clone --recursive https://github.com/dushyant30suthar/opencode-llama.cpp
cd opencode-llama.cpp

# 1. CUDA toolkit 13.3+ and a supported host compiler — see docs/setup.md
# 2. Build the engine (checks the CUDA version for you)
./scripts/build-llama.sh
# 3. Install opencode and the plugin — see docs/setup.md
curl -fsSL https://opencode.ai/install | bash
# 4. Put GGUF models somewhere, point the plugin at them, then:
opencode
```

Local models appear in the model picker under "llama.cpp (local)".
Full walkthrough: [docs/setup.md](docs/setup.md).

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

This repo pins the llama.cpp commit that the configs were validated against.

- **opencode** updates itself (`opencode upgrade`) — it is stock now, with no
  patches to preserve.
- **opencode-localhost** updates like any npm package.
- **llama.cpp** — move the submodule pin, rebuild with `scripts/build-llama.sh`,
  and re-run the `bench/` probes if the release notes touch your model families.
  Context ceilings and MTP draft lengths in particular are worth re-probing.

When a new combination is validated, bump the pin here in one commit — this
repo is the record of "these versions work together."

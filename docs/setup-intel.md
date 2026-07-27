# Setup — Intel iGPU laptop

The Intel counterpart to [setup.md](setup.md), which covers the CUDA desktop.
Different hardware, different stack, and one important difference in kind:

> On the CUDA box you compile llama.cpp *for* your GPU, because the kernels live
> inside it. On Intel you mostly do not compile anything — OpenVINO ships as a
> precompiled runtime and JIT-compiles **the model** for your GPU at load time.
> Those 20–40 second first loads *are* the compile.

For what any of these numbers mean and which knobs are worth touching, read
[tuning-intel.md](tuning-intel.md). For running the server itself, see
[openvino-server.md](openvino-server.md).

## 1. Drivers — check before you install anything

The common assumption is that Intel GPU performance problems are old drivers.
On a current Fedora that is usually false. Verify rather than reinstall:

```sh
uname -r                                     # kernel
rpm -qa | grep -E 'intel-compute-runtime|intel-igc|level-zero|mesa-vulkan'
vulkaninfo --summary | grep -E 'driverName|driverInfo'
```

You want, roughly: `intel-compute-runtime` (NEO) and `intel-igc` recent, a
Level Zero loader present, and Mesa 25+ for Vulkan. OpenVINO deliberately does
**not** bundle the GPU runtime and does not pin an exact version — install the
distro's current one and move on.

Two things worth knowing:

- The kernel driver may be **`xe`** rather than `i915` on Arrow Lake, and Mesa
  will warn that support is experimental. It works. There is an open OpenVINO
  issue about `i915` vs `xe` throughput differences, so if GPU numbers look
  wrong, that is a variable to A/B — not the first thing to suspect.
- Do **not** chase the archived `ipex-llm`. Intel archived it in January 2026;
  its successor is the OpenVINO backend, and upstream llama.cpp's Vulkan path
  already beats ipex by 50–75% on generation.

## 2. Pick a stack

Two viable paths. They are not equivalent — see the
[backend comparison](tuning-intel.md#which-backend).

### A. OpenVINO + OVMS (recommended)

Fastest measured generation, best prefill, and it is the path Intel actually
supports. Costs you the GGUF ecosystem: it consumes **OpenVINO IR**, not GGUF.

1. **Runtime** — download the OpenVINO archive and extract it user-local (no
   root needed):

   ```sh
   curl -L -o openvino.tgz \
     "https://storage.openvinotoolkit.org/repositories/openvino/packages/2026.2.1/linux/openvino_toolkit_ubuntu24_2026.2.1.21919.ede283a88e3_x86_64.tgz"
   mkdir -p ~/.local/openvino && tar -xzf openvino.tgz -C ~/.local/openvino --strip-components=1
   source ~/.local/openvino/setupvars.sh
   ```

   Ubuntu builds run fine on Fedora — they need GLIBC 2.38 and Fedora 44 has
   2.43.

2. **Server** — see [openvino-server.md](openvino-server.md) and
   `scripts/setup-ovms.sh`. One thing that is not optional:

   > Take the **`python_on`** build. The C++-only `python_off` build renders
   > chat templates with a cut-down engine that **silently drops the system
   > message and cannot emit tool calls** — which looks exactly like a model
   > that refuses to use its tools.

3. **Models** — Intel publishes prebuilt int4 IR on the
   [`OpenVINO`](https://huggingface.co/OpenVINO) HF org, so no conversion is
   needed for common models:

   ```
   OpenVINO/Qwen3.6-35B-A3B-int4-ov          # MoE, the daily driver
   OpenVINO/Qwen3.6-27B-int4-ov              # dense, higher quality, ~5x slower
   OpenVINO/Qwen3-Coder-30B-A3B-Instruct-int4-ov
   ```

   Put them in `~/.lmstudio/models-ov/`. To convert something yourself:

   ```sh
   optimum-cli export openvino --model <hf-id> \
     --weight-format int4 --sym --group-size 128 --ratio 1.0 <out-dir>
   ```

   Since OpenVINO 2025.3, `openvino-genai` can also read **GGUF directly** —
   useful for a quick trial, though the prebuilt IR is better optimised.

### B. llama.cpp + Vulkan

Keeps GGUF and the whole llama.cpp ecosystem; generation is ~15% behind OVMS on
MoE. Use `scripts/build-llama-intel.sh`. Two flags matter and both differ from
the desktop: **`flash-attn = off`** and **`ubatch-size = 2048`**.

Not worth building: **SYCL** (no faster than the archived ipex-llm, and needs
`source setvars.sh` before every run) and the **llama.cpp OpenVINO backend**
(preview quality; crashes on MoE).

## 3. Wire it into opencode

Use the [opencode-localhost](https://github.com/dushyant30suthar/opencode-localhost)
plugin. It is a plugin, not a fork — nothing to merge on every opencode update.

```jsonc
// ~/.config/opencode/opencode.jsonc   AND   ~/.config/opencode/tui.jsonc
{ "plugin": ["opencode-localhost"] }
```

Then configure the backend in `~/.config/opencode/providers/<backend>/server.ini`,
which the plugin writes on first launch.

**If you are developing the plugin**, note that opencode resolves
`"plugin": ["opencode-localhost"]` by installing the *published npm package*
into its own cache — it ignores `~/.config/opencode/node_modules`. Your edits
will silently never run. Point the cache at your checkout:

```sh
C=~/.cache/opencode/packages/opencode-localhost@latest/node_modules/opencode-localhost
mv "$C" "$C.npm-backup" && ln -s ~/path/to/opencode-localhost "$C"
```

## 4. Verify

```sh
opencode models | grep -E '^(openvino|llamacpp)/'
```

If nothing appears, the provider was not registered. The plugin contributes
nothing when a backend is unconfigured or still starting — and OVMS takes
25–60 s to compile the graph onto the GPU before it answers, so the *first*
launch after a cold start often registers nothing and the second one works.

Sanity-check that generation is real and not silently empty:

```sh
opencode run --model openvino/<your-model> "Write a Python function that reverses a string."
```

An empty reply is a known failure shape, not a mystery — see
[context is bounded by the KV cache](tuning-intel.md#context-is-bounded-by-the-kv-cache-not-the-checkpoint).

## 5. Optional: the NPU

It works but is **3.5× slower than the GPU** and constrained in several ways;
it is an efficiency device, not a coding engine. If you want it anyway, the
driver pairing is exact — OpenVINO 2026.2 needs NPU user-mode driver **1.33.0**,
and a mismatch shows up as `Unsupported configuration key: NPU_MAX_TILES`. See
[the NPU section](tuning-intel.md#the-npu).

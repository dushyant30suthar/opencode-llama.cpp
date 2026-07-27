# OpenVINO Model Server — OpenAI API on the Intel iGPU

`llama-server` is the CUDA/Vulkan path. **OVMS** is the third backend: it serves
the OpenVINO IR models over an OpenAI-compatible HTTP API on the Arc 140T iGPU,
with continuous batching, paged attention and prefix caching.

This page is *how to run it*. For what the numbers mean, which knobs are worth
touching and the dead ends already ruled out, see
[tuning-intel.md](tuning-intel.md).

Why it exists: raw OpenVINO GenAI already beat Vulkan on the 30B MoE, but GenAI
is a Python library, not a server. OVMS is the same GenAI engine behind an HTTP
front end — so the win survives the trip through an API.

Measured on Qwen3-Coder-30B-A3B int4, GPU, KV `u8`:

| Backend | Generation |
| --- | --- |
| llama.cpp Vulkan Q4 | 24.9 t/s |
| OpenVINO GenAI (library, int4 IR) | 30.3 t/s |
| **OVMS (server, int4 IR)** | **36.1 t/s** decode · 32.5 t/s end-to-end over HTTP |

(The two OVMS figures are the same run measured two ways — decode rate alone,
and wall-clock including prefill and HTTP. Both beat the in-process library, so
the API front end costs nothing. The current daily driver, Qwen3.6-35B-A3B,
measures **29.4 t/s** on the same setup; it is a larger model.)

> **Generation is the win; prefill is the price — but less than first measured.**
> The original numbers here (7,348 tok → 42.6 s; 29,338 tok → 263 s, "the rate
> degrades as the prompt grows") were taken with `cache_size: 4` and
> `max_num_seqs: 8`, which silently caps a single request — see
> [max_num_seqs](#4-the-servable-format). Re-measured on the 35B MoE with
> `cache_size: 6, max_num_seqs: 1`, by streaming TTFT with the output verified:
>
> | Prompt | Time to first token | Prefill rate |
> | --- | --- | --- |
> | 2,075 tokens | 12.8 s | 162 t/s |
> | 8,755 tokens | 52.3 s | 167 t/s |
> | 17,995 tokens | 117.3 s | 153 t/s |
> | 36,795 tokens | 158.8 s | **232 t/s** |
>
> So the rate does **not** degrade with length — it holds and then improves,
> because Qwen3.6 is a hybrid linear-attention model. Prefill is still the
> binding constraint (a 36k context costs ~2.6 minutes cold), but it is paid
> **once**: generation then holds ~25-29 t/s regardless of context, and prefix
> caching returned a re-sent 36k prompt in **0.3 s**. Read
> [Context budget](#context-budget) before wiring it into an agent.

## 1. Install

```sh
./scripts/setup-ovms.sh              # ~120 MB download, installs to ~/.local/opt/ovms
./scripts/setup-ovms.sh --reinstall  # force re-download
```

Nothing is compiled. OVMS builds with Bazel and takes hours; the script installs
the **official release tarball** instead. The `model_server` submodule is pinned
to the same tag (`v2026.2.1`) purely as the doc/template reference — the binary's
own build hash `1122f03bf` matches the pinned commit exactly.

Three things the script encodes, all of them load-bearing:

> **The `python_on` package is mandatory.** The C++-only `python_off` build
> renders chat templates with a cut-down engine that **silently drops the system
> message and cannot emit tool calls**. Qwen3-Coder's template opens with
> `{% macro %}` — python_off mis-renders it outright. An agent client without a
> system prompt or tool calls is useless, so we pay the Python dependency.

> **`python_on` links `libpython3.12`, which Fedora 44 does not have** (it ships
> 3.14 only). The script pulls the interpreter from uv's standalone CPython
> (`~/.local/share/uv/python/cpython-3.12.*`) and puts its `lib/` on
> `LD_LIBRARY_PATH`. If you have no 3.12: `uv python install 3.12`.

> **Jinja2 lives in its own venv**, deliberately not `openvino-genai/.venv`.
> OVMS ships its own `openvino`, `openvino-genai` and `openvino-tokenizers`
> wheels under `lib/python`, and upstream is explicit that a pip-installed
> openvino next to them breaks the install. `lib/python` is first on
> `PYTHONPATH` so the bundled copy always wins.

The **redhat** package is used rather than ubuntu22/24 — Fedora is the closer
relative and it loads clean on glibc 2.43 (`ldd` reports nothing missing). No
`sudo` anywhere; everything lands in `~/.local`.

Layout after install:

| Path | What |
| --- | --- |
| `~/.local/opt/ovms/bin/ovms` | the server binary |
| `~/.local/bin/ovms-serve` | generated wrapper — sets env, execs `ovms`. On `$PATH`, so clients discover it with no configuration |
| `~/.local/opt/ovms/lib/python` | OVMS's bundled openvino/genai/tokenizers |
| `~/.local/opt/ovms/pyenv` | Jinja2-only venv for chat templates |
| `~/.local/state/ovms/servables/<name>/graph.pbtxt` | one servable per model |
| `~/.local/src/ovms/` | downloaded tarballs (cache) |

## 2. Launch

```sh
~/.local/bin/ovms-serve Qwen3.6-35B-A3B-int4-ov --rest_port 8100
```

The servable name is the **model directory name** under
`~/.lmstudio/models-ov/`, and it is also the model id the API advertises.

`ovms-serve` is self-contained: it exports `LD_LIBRARY_PATH` and `PYTHONPATH`
itself and then `exec`s the binary. That matters for the same reason Vulkan won
on the llama.cpp side — a supervisor can spawn it directly, with no shell
sourcing. The equivalent raw invocation is:

```sh
export LD_LIBRARY_PATH=$HOME/.local/opt/ovms/lib:$HOME/.local/share/uv/python/cpython-3.12.12-linux-x86_64-gnu/lib
export PYTHONPATH=$HOME/.local/opt/ovms/lib/python:$HOME/.local/opt/ovms/pyenv/lib/python3.12/site-packages
$HOME/.local/opt/ovms/bin/ovms \
  --model_path $HOME/.local/state/ovms/servables/Qwen3.6-35B-A3B-int4-ov \
  --model_name Qwen3.6-35B-A3B-int4-ov \
  --rest_port 8100
```

The 35B loads in ~25-30 s with a warm page cache (62 s measured cold, when the
19 GB read is on the critical path) and logs `state changed to: AVAILABLE`. Port defaults to
8100 (`OVMS_PORT`, or `--rest_port`) — 8000 is OVMS's own default but collides
with too many dev servers.

**Serve one model at a time.** OVMS *can* take a `config.json` listing several
servables, but it loads them all at startup — and the 35B alone is 19.7 GB of a
32 GB shared-memory box. So `setup-ovms.sh` generates one servable per model and
`ovms-serve` runs exactly one.

The practical consequence: `~/.local/state/ovms/servables/` lists everything
*available*, while `/v3/models` lists only what is *currently up* — one entry.
Switching models means stopping the process and re-running `ovms-serve` with the
other name:

```sh
pkill -x ovms                                    # not pkill -f, see Gotchas
~/.local/bin/ovms-serve Qwen3.6-27B-int4-ov
```

Any client that discovers models from `/v3/models` will therefore only ever show
the running one. That is a property of the memory budget, not a bug to route
around — loading both would swap the box.

## 3. The API

Base URL is **`http://localhost:8100/v3`** — note `/v3`, not `/v1`. OpenAI
clients that hardcode `/v1` will 404.

| Path | Method | Notes |
| --- | --- | --- |
| `/v3/chat/completions` | POST | chat, streaming, tools |
| `/v3/completions` | POST | raw text completion — **not available for VLM servables**, see below |
| `/v3/models` | GET | lists the served model id |
| `/v3/tokenize` | POST | token ids for a string, same tokenizer |
| `/v1/config` | GET | servable load state (`AVAILABLE`) |
| `/v2/health/ready` | GET | 200 when ready — **use this for readiness**, not `/v3/models` |

> **VLM servables are restricted to the chat endpoints.** Both Qwen3.6
> checkpoints are VLM exports, so `/v3/completions` returns
> `"Wrong endpoint. VLM Servable allowed only on /v3/chat/completions,
> /v3/responses endpoint or /v3/tokenize"`. This is not just an inconvenience:
> the raw-completion endpoint is the only way to bypass the chat template, and
> the template is what injects the stop tokens that break prompt-lookup
> decoding. Being forced onto chat closes that door.

> **Do not probe `/v3/models` for readiness.** It answers **200 with an empty
> list** for the whole graph-compile window — a healthy-looking server with
> nothing to offer, which reads to a client as "provider present, no models"
> and gets it dropped. `/v2/health/ready` returns 503 for exactly that window
> and 200 once the graph is servable.

Auth is off unless `--api_key_file` or `API_KEY` is set; clients still need to
send some placeholder key if their SDK insists.

```sh
curl -s http://localhost:8100/v3/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3.6-35B-A3B-int4-ov",
    "max_tokens": 150,
    "temperature": 0,
    "messages": [
      { "role": "system", "content": "You are a concise Go expert. Code only, no prose." },
      { "role": "user", "content": "Write a Go function that checks if a string is a palindrome, ignoring case." }
    ]
  }'
```

Streaming is standard SSE (`"stream": true` → `data: {...}` chunks with
`delta.content`, terminated by `data: [DONE]`).

### Tool calling

Works, including **parallel calls in one response**, and is the reason for the
`python_on` package. Requests use the normal OpenAI `tools` array; responses come
back with `finish_reason: "tool_calls"` and a populated `tool_calls` array:

```json
"tool_calls": [
  { "id": "KhwZum6rq", "type": "function",
    "function": { "name": "list_dir", "arguments": "{\"path\":\"/home/user/project\"}" } },
  { "id": "B82E8fCBl", "type": "function",
    "function": { "name": "read_file", "arguments": "{\"path\":\"/home/user/project/README.md\"}" } }
]
```

Feeding results back as `{"role": "tool", "tool_call_id": ..., "content": ...}`
completes the round trip normally.

> **The `tool_parser` must match the model, and the model's *name* will lie to
> you.** It converts the model's native tool syntax into OpenAI `tool_calls`; the
> wrong one is a silent failure — raw markup arrives as assistant text,
> `tool_calls` stays empty, and the model just looks like it is refusing to use
> its tools.
>
> `setup-ovms.sh` therefore reads each model's **chat template**, not its name:
>
> | Marker in the template | Parser |
> | --- | --- |
> | `<function=` (XML function/parameter blocks) | `qwen3coder` |
> | `tool_call` (a JSON object inside `<tool_call>` tags) | `hermes3` |
>
> `<function=` is tested first because those templates contain both markers.
> This matters concretely: the **Qwen3.6** checkpoints ship Qwen3-Coder's
> `<tool_call><function=...>` format verbatim despite not being "coder" models —
> their tool-call block is byte-identical to the 30B's. A name-based rule sends
> them to `hermes3` and breaks tool calling. Upstream supports `llama3`, `phi4`,
> `hermes3`, `mistral`, `qwen3coder`, `gptoss`, `devstral`, `lfm2`, `gemma4`.

### Reasoning models need `reasoning_parser` too

Same detection, different marker: a template containing `<think>` means the
model thinks, and without `reasoning_parser` its chain-of-thought is served as
**ordinary assistant text** with a trailing unmatched `</think>`. Measured on
Qwen3.6-27B, no parser set:

```json
"content": "The user wants to read the file `/etc/hostname`.\nI should use the
            read_file tool ... Let's call the tool.\n</think>\n\n"
```

With `reasoning_parser: "qwen3"` the same request splits cleanly:

```json
"content":           "\n\n",
"reasoning_content": "The user wants to read the file `/etc/hostname`. ...",
"tool_calls":        [{ "function": { "name": "read_file", ... } }]
```

`setup-ovms.sh` adds it automatically for `<think>` templates (`qwen3`, or
`gemma4`/`gptoss` by family). Qwen3-Coder has no `<think>` and correctly gets
none. Upstream supports `qwen3`, `gptoss`, `gemma4`.

Two consequences for clients: `content` on a reasoning turn is
**whitespace, not empty string**, so trim before testing emptiness; and any
"did this turn produce output" check must count `reasoning_content`, or a
legitimate thinking turn looks like a failure.

That second one is not theoretical. A Qwen3.6 turn that reasons and then stops
carries **empty `content` and no tool calls** — a client testing only those two
fields hard-errors a perfectly good response, on the first request anyone makes.

What the local models actually resolve to, and what "normal" looks like for each:

| Model | `tool_parser` | `reasoning_parser` | Thinks? |
| --- | --- | --- | --- |
| Qwen3.6-27B-int4-ov | `qwen3coder` | `qwen3` | **yes** |
| Qwen3.6-35B-A3B-int4-ov | `qwen3coder` | `qwen3` | **yes** |
| *Qwen3-Coder-30B-A3B-Instruct-int4-ov* | `qwen3coder` | — | no |

All three share Qwen3-Coder's tool syntax — which is the point of detecting from
the template rather than the name — but only the Qwen3.6 pair think. Swap
between them and the shape of a "successful" response changes underneath you.
(The Qwen3-Coder row is kept as the contrasting non-thinking case; it is a
Qwen3-generation model and is no longer kept on this machine.)

## 4. The servable format

OVMS does not serve an IR directory directly. Each model needs a **servable
directory containing `graph.pbtxt`** — a MediaPipe graph wrapping the LLM
calculator — and `--model_path` points at *that* directory, not at the IR.
(The `model_name/1/` version layout applies to classic single-model serving;
generative servables use the graph instead.)

`setup-ovms.sh` generates one per model found in `~/.lmstudio/models-ov/`, from
upstream's own `text_generation` template. Only `node_options` is ours — the rest
must stay verbatim, in particular `input_side_packet: "LLM_NODE_RESOURCES:llm"`,
without which the graph is rejected.

```protobuf
node_options: {
    [type.googleapis.com / mediapipe.LLMCalculatorOptions]: {
        models_path: "/home/you/.lmstudio/models-ov/Qwen3.6-35B-A3B-int4-ov",
        plugin_config: '{"KV_CACHE_PRECISION":"u8"}',
        enable_prefix_caching: true,
        cache_size: 4,
        max_num_seqs: 1,
        device: "GPU",
        tool_parser: "qwen3coder",
        reasoning_parser: "qwen3",
    }
}
```

`models_path` is absolute here on purpose — the IR directories stay pristine, so
a re-download never clobbers a config and one model can have several servables
(GPU vs NPU, different cache budgets).

OVMS infers the **servable type** from the directory contents, so the same graph
covers vision models too: a text export has `openvino_model.xml`, a VLM splits
into `openvino_language_model.xml` plus vision/embedding parts, and OVMS picks
the VLM pipeline on its own. The script accepts either (and skips directories
with neither, which is what a half-finished download looks like).

The VLM path is since **verified for text generation** — both Qwen3.6
checkpoints ship as VLM exports and serve text, tool calls and reasoning
normally through it. Images in have still not been exercised. Note the VLM
packaging is not free: it is what makes these models "embedder models"
(blocking speculative decoding) and VLM servables (restricting them to the chat
endpoints), which closes every route to assisted decoding — see
[tuning-intel.md](tuning-intel.md#dead-ends-with-reasons).

| Option | Meaning |
| --- | --- |
| `device` | `GPU`, `CPU`, or `NPU`. NPU forces the stateful pipeline and ignores most continuous-batching options. |
| `cache_size` | KV cache in GB. `0` = grow dynamically, which can eat all RAM — pin it. Caps your context **and your footprint** — see below. |
| `enable_prefix_caching` | Reuse KV for repeated prefixes. Large win for chat, where the whole history is resent every turn — a re-sent 36k prompt came back in 0.3 s. |
| `max_num_seqs` | Concurrent sequences. **Set this to 1.** See below. |

> **Size `cache_size` against your RAM, not just your context wish.** The model
> and its cache have to leave room for the OS. On a 30 GiB box with the 19.7 GB
> MoE, measured on one prompt with everything else identical:
>
> | `cache_size` | MemAvailable after load | Generation |
> | --- | --- | --- |
> | 6 | 3.0 G | **5.14 tok/s** |
> | 4 | 8.9 G | **28.36 tok/s** |
> | 2 | 9.0 G | 28.52 tok/s |
>
> A 5.5× swing from one value. It does not fail loudly — it just spills and
> crawls. `setup-ovms.sh` defaults to 4 for this reason.

> **`max_num_seqs: 8` was wrong, and it fails silently.** Two measured reasons:
>
> 1. **It divides your context by 8.** The cache is split across sequence
>    slots, so `cache_size: 4` with 8 slots gives one request ~0.5 GB — about
>    11k tokens. Longer prompts return an empty 200 (below) with no hint that a
>    scheduler setting caused it. With `max_num_seqs: 1` the same 4 GB serves
>    the whole request.
> 2. **Concurrency is actively harmful on MoE anyway.** Measured on the 35B:
>    one stream 25.6 t/s aggregate, **two streams 13.2** — because two
>    sequences route to different experts and each pass must load the *union*
>    of their expert sets. There is no throughput to buy here, only context to
>    lose.
| `plugin_config` | Raw OpenVINO plugin properties. `KV_CACHE_PRECISION: u8` roughly halves cache bytes per token. |

Regenerate after editing the defaults:

```sh
OVMS_CACHE_SIZE=8 OVMS_DEVICE=GPU ./scripts/setup-ovms.sh
```

Script overrides: `OVMS_VERSION`, `OVMS_DISTRO`, `OVMS_PREFIX`, `OVMS_STATE_DIR`,
`OVMS_MODELS_DIR`, `OVMS_CACHE_SIZE`, `OVMS_DEVICE`.

### Context budget

`cache_size` sets the context ceiling, and the server prints its own usage:

```
[llm_executor] All requests: 1; Scheduled requests: 1; Cache type: static, cache usage: 22.1% of 4.0 GB;
```

On a **pristine** cache a 44,006-token prompt peaked at **50.1 % of 4 GB** —
47.7 KiB/token, matching the arithmetic for `u8` KV exactly
(48 layers x 4 KV heads x 128 dim x 2 for K+V x 1 byte = 48 KiB). So:

| KV precision | Bytes/token | 4 GB holds | 8 GB holds |
| --- | --- | --- | --- |
| `u8` | 48 KiB | ~85k tokens | ~170k tokens |
| default (fp16) | 96 KiB | ~43k tokens | ~85k tokens |

> **Treat that table as an upper bound, not a budget.** It is arithmetic, and
> reality came in far lower: at `cache_size: 6, max_num_seqs: 1` a 19,306-token
> prompt answered and a **39,406-token prompt returned zero tokens**. Weights,
> activations and prefix-cache retention all compete, and with `max_num_seqs > 1`
> the cache is divided again. Advertise a context you have actually served —
> ~32k with a 6 GB cache is defensible; 85k is not.

Two traps when reading that percentage:

- **Prefix caching retains freed blocks.** Usage is cumulative across requests,
  not per-request, so a mid-session reading tells you nothing about one prompt's
  cost. Only a freshly started server gives a clean measurement.
- **Sample mid-prefill and you undercount.** The same 44k prompt reads 22 %
  halfway through. Take the peak, after the request completes.

At 100 % requests get preempted and recomputed; a single request that outgrows
the whole cache is terminated outright.

> **Overflowing the cache fails silently — plan for it in the client.** A request
> that exceeds the cache does **not** return 400. Verified twice against a
> deliberately small 1 GB cache with a 44,006-token prompt:
>
> ```
> HTTP 200
> {"choices":[{"finish_reason":"stop","index":0,
>   "message":{"content":"","role":"assistant","tool_calls":[]}}],
>  "usage":{"prompt_tokens":44014,"completion_tokens":0,"total_tokens":44014}}
> ```
>
> HTTP 200, `finish_reason: "stop"`, empty content, zero completion tokens, no
> `error` field, and nothing in the server log either — the cache climbs to
> ~94 % and the request dies. On the wire it is identical to a model that chose
> to say nothing.
>
> `prompt_tokens` is still reported, so it is detectable: treat
> `completion_tokens == 0` with empty content and no tool calls as
> **context overflow**, not an empty completion. Make it a hard error rather
> than a retry — a retry cannot succeed and costs another full prefill
> (138 s in the test above).
>
> **Streaming does the same thing, and is harder to spot.** The same overflow
> with `"stream": true` returns a clean, well-formed, complete SSE stream —
> HTTP 200, proper chunked close, terminated by `data: [DONE]` — containing one
> frame and no output. The entire 169-byte body:
>
> ```
> data: {"choices":[{"index":0,"logprobs":null,"finish_reason":"stop"}],"created":1784733236,"model":"ctxtest","object":"chat.completion.chunk"}
>
> data: [DONE]
> ```
>
> Three details that matter to any client parsing this:
>
> - **`delta` is absent**, not `{}`. Reading `choice.delta.content` throws on
>   `undefined` before you can detect anything.
> - **`usage` never appears in streaming** — not here, not in a healthy stream.
>   It is non-streaming-only, so the prompt size is unavailable on this path.
> - **In a healthy stream there is no standalone stop frame.**
>   `finish_reason: "stop"` rides on the *last content delta*, and the first
>   frame is `delta: {"role":"assistant","content":null}` — a `null`, not a
>   string. Test for a non-empty string, not for presence.
>
> Detect it as *a stream that ended having yielded no output*, counting
> `content`, `reasoning_content` and `tool_calls` as output — a reasoning-only
> turn is legitimate and must not trip the check.
>
> Measured against KV-cache exhaustion specifically; other overflow modes are
> untested.

**The practical ceiling is prefill time, not cache bytes.** 4 GB of `u8` KV
holds ~85k tokens, but filling even 30k of it takes over four minutes. Keep
`enable_prefix_caching: true` — it is what makes multi-turn chat tolerable,
since only the new suffix of a resent conversation gets prefilled.

## 5. Verifying it is really on the GPU

`intel_gpu_top` needs perf capabilities we do not have unprivileged. The DRM
fdinfo works instead and is more direct:

```sh
PID=$(pgrep -x ovms)
for fd in $(ls /proc/$PID/fd); do
  case "$(readlink /proc/$PID/fd/$fd)" in */dev/dri/*)
    grep -E "drm-driver|drm-resident-gtt|drm-cycles-ccs" /proc/$PID/fdinfo/$fd ;;
  esac
done
```

```
drm-driver:         xe
drm-resident-gtt:   20379040 KiB      # 19.4 GB = 16 GB weights + 4 GB KV
drm-cycles-ccs:     196303673         # compute-engine cycles — real inference
```

Non-zero `drm-cycles-ccs` on the `xe` driver is the proof; a CPU fallback would
hold no GTT memory and burn no compute cycles. `ovms` host RSS stays ~0.4 GB
because the weights live in GPU memory, so don't read `ps` and panic.

## 6. Gotchas

- **`/v3`, not `/v1`.** `/v1` only carries `/v1/config`; `/v1/models` is a 404.
- **A 200 from `/v3/models` is not a readiness signal.** While a graph compiles,
  the REST port is already open and answers `{"data":[],"object":"list"}` —
  HTTP 200, empty list. Use `/v2/health/ready`, which returns **503** for
  exactly that window and 200 once the model is `AVAILABLE`. A client that
  probes `/v3/models` mid-load sees a healthy server with no models and will
  conclude the provider is absent.
- **A client timeout kills the generation.** Disconnecting mid-request aborts the
  graph and logs `CANCELLED: CalculatorGraph::Run() failed` — no partial result,
  and the prefill work is lost. With long prompts this is easy to hit by
  accident: set client timeouts in *minutes*, not seconds.
- **`Could not find platform dependent libraries <exec_prefix>`** on startup is
  harmless — the embedded interpreter has no `PYTHONHOME`. Templates still render.
- **`pkill -f ovms` kills your own shell** if the pattern matches the command
  line running it. Use `pkill -x ovms`.
- **First load reads the whole IR from disk** — ~25-30 s for the 35B warm, 62 s cold, longer
  cold. `--cache_dir` caches the compiled GPU blob and cuts subsequent loads.
- **The NPU path is stateful-only.** `setup-ovms.sh` routes any model directory
  with `npu` in its name to `device: "NPU"`, where continuous-batching options are
  ignored. The iGPU is ~3.5× faster anyway.
- **Serving from Hugging Face directly** also works and generates the graph for
  you, if you would rather not hand-manage servables:
  ```sh
  ovms --source_model OpenVINO/Qwen3-8B-int4-ov --model_repository_path models \
       --task text_generation --target_device GPU --rest_port 8100
  ```
  It downloads on first run. `setup-ovms.sh` does not use this — it keeps the
  models we already have on disk, and works offline.

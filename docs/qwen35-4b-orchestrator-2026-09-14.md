# Qwen3.5-4B as the local orchestrator — measured 2026-09-14

One file, one row of truth per measurement. Hardware: Core Ultra 7 255H
(Arrow Lake-P), Arc Pro 130T/140T iGPU, 30 GB RAM. Engine: OVMS python_on
bundle at `~/.local/opt/ovms` (OpenVINO 2026.3.1), port 8100. Deployed config:
`config/providers/openvino/` (mirrored live files, measured comments inline).

---

## Why this model

The orchestrator's whole job: read the human's notes (often a photo), launch
the developer subagent, relay results, flip the board. Requirements fell out
of that: **vision, reliable tool calling, small and fast** — the 27B is
already the developer model, and the relay talks to a human over Telegram, so
latency is visible.

Candidates, eliminated:

| Model | Verdict |
|---|---|
| Gemma 4 26B-A4B (previous occupant) | Looped in agent use; hit `exceed_allocatable_mem_size` on the iGPU under load (server.log 2026-09-14 13:45, VLM pipeline generation failed) |
| Ministral 3 8B (2512) | **No vision.** The Ministral line is text-only; Mistral's vision lives in Pixtral/Small. Fails the first requirement |
| Qwen3-VL-4B (2025-10) | Vision + agent, but a year older and clearly behind on tool calling (TIR-Bench 22.5 vs 38.9) |
| Gemma 4 E4B | Vision + function calling, but Gemma is the family we are moving off of |
| **Qwen3.5-4B** | Native early-fusion vision, #1 in the 2026 13-model local tool-calling eval (97.5% pass, best multi-tool 7/8), 4B dense, 262K native context, ~4 GB |

The eval also showed the pattern that matters for loops: **dense models
terminate tool-call loops where MoE/elastic variants re-call the same tool.**
Qwen3.5-4B is dense.

## What is deployed

- Weights: `OpenVINO/Qwen3.5-4B-int4-ov` (2026-06-12, needs OV ≥ 2026.2.0) →
  `~/.lmstudio/models-ov/Qwen3.5-4B-int4-ov`, 3.3 GB. Full VLM export:
  language + vision + merger + pos-embeddings + text-embeddings IRs.
- Servable: `config/providers/openvino/servables/Qwen3.5-4B-int4-ov.graph.pbtxt`
  — `device: GPU`, `pipeline_type: VLM`, `KV_CACHE_PRECISION: u8`,
  `cache_size: 4`, `cache_interval_multiplier: 1024` (hybrid DeltaNet —
  matches the Qwen3.6-27B servable), `tool_parser: qwen3coder`,
  `reasoning_parser: qwen3`, prefix caching on.
- `server.ini`: `context = 65536`, Qwen "precise coding" sampling
  (0.6 / 0.95 / 20).

## Measured

Startup: health-ready in **~10 s** (vs 24–60 s for the 19–26 GB models on
this box). RAM after load: **12 GiB free** of 30.

| Test | Result |
|---|---|
| Text chat | trivial prompt, 3.2 s |
| Tool calling | correct function, **typed** args (`{"port":8100}` integer, not string), `finish_reason: tool_calls` |
| Multi-turn loop | image → `launch_developer` call → tool result → final answer; **self-terminates**, no stray re-calls |
| Vision | 6-line note image transcribed; image costs ~450 prompt tokens |
| 20k-token needle | retrieved, 15.7 s |
| 40k-token needle | retrieved, 41.9 s (~950 tok/s prefill) |
| README (~4k varied text) + image | both answers correct, 5.8 s |

KV math at u8: 8 of 32 layers are full attention → ~16 KB/token → the 4 GB
cache holds ~262k tokens. The advertised 65536 is a quarter of the ceiling;
40k was the practical verification.

## Thinking mode: fixation, not a loop

> **CORRECTION 2026-09-14 (later).** The decision at the bottom of this
> section — force thinking off for the 4B — was wrong in live use. The
> template's thinking-off path appends an empty think block to the prompt,
> and the 4B leaks bare closing think tags out of that state. Final state:
> the plugin sends **no** chat_template_kwargs for the 4B (commit `6743426`),
> i.e. the template default, thinking on. Measurements below stand as
> measured; the decision history is at the end of the section.

The model is a hybrid thinker (Qwen3-style `<think>` tags, on by default).
The question that decided the deployment: is thinking-on the Gemma-style
loop we are escaping, or just slow?

Measured on the same 6-line transcription, four runs:

| | thinking off | thinking on |
|---|---|---|
| tokens | 41 | 1,718 (one run: >2,048, clipped) |
| time | 5.8 s | 66.6 s |
| answer | correct | correct |
| finish | stop | stop |

The full thinking output, read line by line: **no verbatim repeated
sentences at all** — it is not a mechanical loop, and it terminates. What it
does instead is *fixate*: the word "llms" appears **68 times** in the
reasoning, "wait" 17×, "look closely" 29×. It re-examines one ambiguous
glyph ("ll" vs "lI") roughly fifteen times, wanders off, comes back, and
finally converges. A 4B model thinking hard about a picture of text.

Two properties decide the deployment:

1. **Thinking length varies run-to-run.** One run needed >2,048 tokens on a
   prompt whose answer is 41 tokens. Against a tight output budget that
   surfaces as `finish_reason: length` with empty content — which reads as
   the model refusing to answer.
2. **~10× latency on trivial relay turns.** For a Telegram-facing relay,
   every reply costs 30–90 s of thinking about nothing.

Decision round 1 (opencode-localhost commit `1ab0627`, superseded): the
plugin forces `enable_thinking: true` for Qwen3 **developer** models
(27B/35B — where the Qwen model card's agentic-capability claim applies) and
**excludes the 4B orchestrator**. That exclusion was not a disable — the
template default is thinking on — so the 4B ran thinking-on with a 32k
budget, and a trivial "hi" burned it whole (~21 min of generation).

Decision round 2 (commit `92ee8fb`, superseded): explicit
`enable_thinking: false` for the 4B. Fast (14 tokens, 1.3 s on "hi") — but
the template's thinking-off path appends an empty think block to the prompt,
and in live opencode use the 4B **leaked bare closing think tags into the
visible output** and lost track of a second image in the conversation
("I can't see the second image", four times in a row).

Decision round 3 (commit `6743426`, **final**): no kwargs at all for the 4B —
pure template default, thinking on. The model's trained mode. Measured
against the live server, the exact failing scenario (two different note
images, two turns):

| Test | Result |
|---|---|
| Single turn + image A | 217 tokens, 9 s, no leak, transcribed |
| Two turns, image B on turn 2 | 133 tokens, 6 s, no leak, **second image read correctly** |
| History contains a leaked bare tag | 71 tokens, 3 s, no leak, **recovered** |

Thinking length on image tasks stayed short (40–170 tokens); the 32k-budget
burn was one outlier run, not the regime. The 9B was considered and not
needed: the 4B passes every orchestrator requirement in its default mode,
and the 9B would roughly double generation latency for it.

## The repetitive-context trap

A 4B model loses an image under a wall of **repetitive** text. 20k tokens of
the *same sentence* + the note image: the model answered the transcription
question with the filler text instead of the image content. The same test
with varied text (README, ~4k) + image: both answers correct, 5.8 s.

Real orchestrator context (AGENTS.md + conversation + photo) is varied, so
this is a caveat, not a blocker — but if the relay ever starts answering
from the wrong source in a long session, suspect repeated boilerplate in the
prefix before suspecting the model.

## Commits

- `opencode-localhost` `80c25ce` — `openvino: flag VLM exports as
  vision-capable` (hasVision: the picker gets `vision: true`, images stop
  being stripped)
- `opencode-localhost` `1ab0627` — `server: scope qwen3 thinking kwargs to
  developer models` (superseded: exclusion was not disable)
- `opencode-localhost` `92ee8fb` — `server: explicitly disable thinking for
  the 4B orchestrator` (superseded: forced-off mode leaks think tags)
- `opencode-localhost` `6743426` — `server: run the 4B orchestrator on the
  template default (thinking on)` (final)
- `opencode-llama.cpp` `c278dca` — deployed config mirrored under
  `config/providers/openvino/`

## Still untested

- **27B + 4B both hot on the iGPU.** The Gemma OOM at 13:45 was a 26B +
  cache footprint problem; 4B + 27B int4 + two 4 GB caches is close to the
  same total. First time both are live, watch `free` and the OVMS log for
  `exceed_allocatable_mem_size`.
- **Real handwritten photos.** Everything here is clean synthetic DejaVu
  text. Expect the glyph-level errors to get worse (one "ll"→"11" already at
  clean-font quality).
- **Thinking-length distribution.** Four runs is a sample, not a
  distribution. If the 4B ever runs thinking-on, log completion tokens per
  turn for a day before trusting a budget.
- **Prefix-cache hits on the real conversation shape.** AGENTS.md sits in the
  system prompt every turn; with `enable_prefix_caching: true` the steady
  state should be near-100% cached after turn one. Not yet measured.
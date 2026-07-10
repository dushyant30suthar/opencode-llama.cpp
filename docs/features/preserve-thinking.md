# Preserve thinking — the tool-call-loop fix

**Symptom:** a thinking model (Qwen3.6-class) works fine for a few turns of an
agentic task, then starts looping — repeating the same tool calls, sometimes
with empty `{}` arguments.

**Root cause:** chat templates strip previous-turn `<think>` blocks from the
rendered prompt by default. Mid-task, the model literally loses its own plan —
it can see *that* it called three tools, but not *why* — and re-derives the
same next step forever.

Two halves make the round-trip work; the stack has both:

1. **Client side** (stock opencode behavior, verified): assistant messages are
   replayed from the session DB each step with `reasoning_content` intact —
   `@ai-sdk/openai-compatible` emits it natively.
2. **Server side** (`models.ini`): `reasoning-preserve = true` makes
   llama-server's template keep the reasoning for **all** history turns.
   Qwen3.6 calls this `preserve_thinking` and recommends it for agents.
   Support is auto-detected per chat template (llama-server logs
   "chat template supports preserving reasoning"), so the key is harmless for
   non-thinking models — new sections seed it automatically.

**Side benefit:** with thinking preserved, the rendered prompt is append-only
across turns, so the KV prefix cache stays hot instead of re-evaluating the
conversation tail every step — a large turn-start latency win at 100k+
context.

**Verification technique:** `POST /apply-template` A/B — plant a sentinel
string in a historical turn's `reasoning_content` and check whether it
survives template rendering with and without the flag.

Related: sampling for local models is owned by `models.ini`, and the fork
sends no sampler overrides — see [architecture.md](../architecture.md). If
loops ever reappear at correct samplers, `presence-penalty` up to 1.5 is the
escape hatch.

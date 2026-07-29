# Why Claude Code stalls a local server — and whose fault it isn't

Measured on this box, 2026-07-29. Applies to any prefix-caching local engine
(llama.cpp, exllamav3, vLLM, SGLang) serving an agentic client built for a
hosted API.

---

## The event

From the exllamav3 server log, three consecutive requests in one live session:

```
19:06:42  ctx 108,462  cached 107,776  new     686   ← normal turn, 2.4 s
19:06:45  ctx 108,317  cached       0  new 108,317   ← 207 SECONDS
19:10:13  ctx 108,713  cached 108,288  new     425   ← normal again, 4.0 s
```

`cached` went to **zero** on a conversation that had been 99.4% cached three
seconds earlier. Note the context also got *smaller* — 108,462 → 108,317. That
is the signature: something near the **start** of the prompt changed, not the
end.

## The mechanism

Every prefix cache — local or hosted — is a **prefix match**. Content is stored
in blocks/pages keyed by the bytes that precede them. A single byte changed at
position N invalidates every block from N onward. Anthropic's own caching
documentation states it in the same terms: *"Any change anywhere in the prefix
invalidates everything after it."*

So a change at position ~0 costs the entire conversation. There is no partial
recovery, on any engine. exllamav3's page hashes, llama.cpp's linear prefix
match, vLLM's block hashes — all identical in this respect.

## What the client changes

Claude Code (the CLI, not the model and not the desktop app) rebuilds the
request every turn and can inject:

- `<system-reminder>` blocks
- file contents, re-injected when a watched file changes on disk
- todo / task-list state
- auto-memory content
- a per-request attribution header in the **first system block**

The last one is the worst case — a per-request hash at position 0 means a
guaranteed full invalidation on every single turn (upstream llama.cpp issue
#19494 describes exactly this).

## Why this is nearly free against Anthropic's API

Two reasons, and the first is the one worth understanding:

1. **The API has an escape hatch local engines don't implement.** Operator
   instructions can be appended as a `{"role": "system", ...}` entry inside
   `messages[]` instead of editing the top-level `system` field. Because that
   lands *after* the cached history, the prefix survives intact. Anthropic
   documents this as the recommended pattern for mid-conversation context, and
   it is model-gated (Opus 5 / Opus 4.8 / Fable 5 / Mythos 5). Injections done
   this way cost nothing. Injections that rewrite the front of the prompt
   still break the cache there too — the difference is what the client chooses
   to do.
2. **Even a full break is cheap on their hardware.** Reprocessing ~110k tokens
   takes seconds on datacenter inference. On 2x RTX 5060 Ti at ~530 t/s it
   takes **207 seconds**.

Same client behaviour, two very different consequences.

## What actually helps

Client-side, on the machine running Claude Code:

```sh
export CLAUDE_CODE_ATTRIBUTION_HEADER=0      # kills the position-0 per-request hash
export CLAUDE_CODE_DISABLE_AUTO_MEMORY=1     # stops memory rewrites at the top
export CLAUDE_CODE_DISABLE_BACKGROUND_TASKS=1
export API_TIMEOUT_MS=600000                 # see below
```

The timeout is not about reducing stalls — it is about not **doubling** them.
A client that gives up during a 3.5-minute rebuild discards the work; the
retry starts from zero. An abandoned rebuild is pure loss.

Behavioural:

- **Compact by ~90-120k.** Every rebuild costs `depth ÷ prefill_rate` seconds.
  At 530 t/s, 110k tokens = 207 s; 60k tokens = 113 s. Depth is the only
  multiplier you control.
- **One model at a time across machines.** A second client requesting a
  different model evicts the first model entirely (`--models-max 1`), which is
  a superset of this problem.

## What no engine can fix

Nothing on the server side recovers a position-0 change. Specifically:

- Bigger cache pools don't help — the pages weren't evicted, they were
  invalidated.
- Denser recurrent checkpoints don't help — a checkpoint below the edit
  doesn't exist when the edit is at the start.
- Switching engines doesn't help — measured on both llama.cpp (divergences
  clustered at 14-22k, 35-94 s each) and exllamav3 (this 207 s event).

The upstream position is also settled: llama.cpp's maintainer explicitly
declined to support insertion-tolerant caching, calling it client
inefficiency the server shouldn't accommodate (PR #24035 discussion).

## Bottom line

This is not a model behaviour, not a bug in the serving stack, and not
something a config change fixes. It is a client optimised for a backend where
re-prefill is nearly free, talking to hardware where it costs minutes. The
levers are all client-side: reduce what gets injected at the front, keep the
conversation shallow enough that a rebuild is survivable, and never let the
client abandon a rebuild it already paid for.

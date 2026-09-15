# LM Studio: UHDR photos kill the engine — "Channel Error" masquerades as token truncation

Investigated 2026-09-15. Incident: 2026-09-14, `qwen/qwen3.5-9b` served by
LM Studio on the PC (`192.168.1.48:1234`, OpenAI-compatible API). Client:
pi coding agent (narrative project session).

---

## Symptom

pi printed `Response was truncated before completion.` nine times in a row
across ~8 hours. pi renders that string for **any** assistant message ending
with `stopReason: "length"` — it does not mean the model ran out of output
tokens.

The session file (`~/.pi/agent/sessions/.../2026-09-14T15-52-46-*.jsonl`)
recorded every one of those responses as:

```json
"usage": { "input": 20922, "output": 0, "reasoning": 0, "totalTokens": 20922 },
"stopReason": "length", "rawStopReason": "length", "content": []
```

**0 output tokens, `finish_reason: "length"`.** A real max-tokens stop cannot
produce this: the model would have to emit at least one token before the
ceiling bites.

## Root cause

LM Studio log for the same window: **`Error: Channel Error`** — the RPC
channel between the LM Studio API process and the inference engine process
broke, i.e. **the engine died mid-request**. When the engine dies, the API
layer returns a degenerate completion: HTTP 200, `finish_reason: "length"`,
`completion_tokens: 0`. The client sees a "truncated response" that is
actually a crash.

## Trigger: full-res UHDR phone photos in the context

The dead window's context contained 4–5 photos per request, each:

- 8160×4592, **16-bit UHDR JPEG**, 8–10 MB on the wire (Telegram attachments,
  sent at full resolution — pi's `read` tool downscales to ~1125×2000 / 315 KB,
  but the Telegram attachment path does not).

Memory math per image at decode: 37.5 MP × 16-bit RGB ≈ **225 MB raw**;
five at once is a multi-GB transient spike. First failure in each cluster
took **12–17 min** (engine choking on the decode/resize), later ones 1–2 min
(engine dying faster on the reloaded model).

Why it persisted for 8 hours: the photos stayed inside the compaction kept
window, so **every** subsequent request re-sent the same images and re-killed
the engine. Even a plain-text "what is the issue" got 0 tokens.

## Ruled out

| Hypothesis | Test | Result |
|---|---|---|
| Context overflow (prompt ≥ 32768) | filler probes at 26k/28k/30k/31k/32k prompt tokens | all completed normally; server's real context ≈ 32768 |
| Server clamps `max_tokens` to 0 on overflow | 34k/40k-token prompt | **HTTP 400** `tokens to keep … greater than the context length` — never a 200/length/0 |
| pi undercounting tokens | session `input` values are server-reported usage | counts are ground truth from the server |
| Deterministic content trigger | replayed the exact failing request (all 5 photos + full 42-message conversation + 9 tool schemas, ×3) plus the 4-photo message, 2026-09-15 | **all succeeded** (88 s–6 min, normal completions) — crash was state-dependent (PC memory pressure / LM Studio build at the time) |

## Secondary finding: the model never saw the photos

The server counts each UHDR photo as **~3 prompt tokens** (a real Qwen-VL
image costs thousands). The 16-bit UHDR decode evidently fails or is dropped
silently, so the vision content was empty — replay responses confabulated
about the images ("the image appears to be rotated 90 degrees"). The 12–17
min latency was spent chewing on files the model could not read.

## Recommendations

1. **Convert to 8-bit sRGB JPEG before sending** (`convert in.jpg -colorspace sRGB out.jpg`, or resave on the phone). UHDR is both the crash trigger and invisible to the model.
2. Downscale attachments before they enter agent context (pi's `read` already does ~1125×2000; the Telegram bridge path does not).
3. Photo-heavy agent sessions: use the bigger-context prajna endpoint, not the PC's 9B.
4. If it recurs: check what precedes `Channel Error` in the LM Studio log (OOM-kill line / segfault) and the PC's memory at that moment.
5. Client-side: treat `finish_reason: "length"` with **0 completion tokens** as an engine/server failure, not a truncation — that signature is not producible by a real token ceiling.
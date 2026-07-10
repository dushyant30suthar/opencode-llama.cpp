# VRAM warm-up on model selection

Selecting a model used to defer the actual load until the first message — you
picked a model, typed your prompt, hit enter, and *then* waited 30+ seconds
for 20 GB to stream into VRAM.

The model setter now fires `POST /models/load` for local models the moment
they're selected (from the model picker or `/config` → "Use this model"), so
the VRAM swap happens **while you type**. With `--models-max 1` router mode
this is also what evicts the previous model.

Selecting flips the router to `loading` immediately with no message sent; by
the time a real prompt goes out, the model is usually resident.

Code: `packages/tui/src/context/local.tsx` (model setter),
`packages/tui/src/util/llamastack.ts`.

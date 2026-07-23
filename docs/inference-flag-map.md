# Mapping llama.cpp inference into flags — what each one actually buys

A model of *this rig* (2× RTX 5060 Ti 16GB, 6-core, PCIe Gen3 x8 + x4, PHB
topology) built from structural analysis (EN diagram) plus measurement on the
27B dense NVFP4+MTP model. Written to explain **why** the tuned numbers are what
they are, so future tuning is deduction rather than brute force.

## 1. Four layers, top-down

| Layer | What it is | Can you change it? |
|---|---|---|
| **Physics** | decode reads `weights + KV` bytes *per token*; tg_ceiling = bandwidth ÷ bytes-per-token | No — only by shrinking bytes |
| **Topology** | 2×448 GB/s, but linked PHB *through the host*, no P2P; GPU0 Gen3 x8, GPU1 Gen3 **x4** | No |
| **Flags** | how work and memory are placed and scheduled | Yes — this is the whole lever set |
| **Workload** | code vs prose changes draft acceptance; context depth changes KV bytes/token | Partly |

**The single most important measured fact:** under load, memory-controller
utilisation sits at **60–68%**, not ~100%. This rig is **not bandwidth-bound
during decode — it is sync/latency-bound.** Almost every counter-intuitive
result below follows from that one fact.

## 2. The flag map, grouped by the resource each one trades

### 2a. VRAM allocators — all draw from one contested pool
`--ctx-size`, `--cache-type-k/v`, `--ubatch-size`, `--gpu-layers`,
`--override-tensor`, draft-model residency.

Structurally `vram headroom` is a **min-cut of size 1 affecting 26 node-pairs** —
the single articulation point of the whole system. Every one of these flags is
competing for the same bytes; there is no independent tuning of any of them.

### 2b. Work distributors — decide how the two GPUs cooperate
`--split-mode {none,layer,row,tensor}`, `--tensor-split`, `--main-gpu`, `--device`.

### 2c. Pass multipliers — tokens produced per forward pass
`--spec-type`, `--spec-draft-n-max`, `--spec-draft-n-min`, `--spec-draft-p-min`,
`--spec-draft-p-split`, `--spec-draft-backend-sampling`.

These are the only flags that beat the physics: they amortise one
bandwidth-bound weight read across several tokens.

### 2d. Overhead reducers — attack latency, not volume
`--flash-attn`, `--poll`, `--prio`, `--op-offload`, `--threads`, CUDA graphs.
On a sync-bound rig this is the category that *should* matter most — and it is
the one no previous bench on this box ever touched.

## 3. Rigid couplings (conservation laws — you cannot buy one without the other)

- `long context` ↔ `kv growth` — **more context makes every token slower.**
  Context is not free capacity; it is a permanent per-token bandwidth tax.
- `fused attention kernel` ↔ `lower kv traffic` — flash-attn is the *only*
  route to reduced KV traffic. Non-negotiable, always enable.
- `wide prompt batch` ↔ `larger compute buffer` — ubatch always costs VRAM.

## 4. The chokepoint, and why decode ≠ prefill

Composing the decode and prefill models exposes an asymmetry that explains the
split-mode paradox in `bench/nvfp4-tune-results.txt`:

- **Decode** — `exchange partial sums` and `produce logits` are on a
  `single-cut-path`. Every decoded token **must** cross the inter-GPU exchange.
  **No alternative vertex-disjoint route exists.**
- **Prefill** — `prompt tokens processed` sits on a `multi-cut-path`: it can be
  reached *either* through the stall *or* through `hidden sync latency` by
  overlapping microbatches.

**Prefill can route around the sync; decode cannot.** Hence the measured
opposition — and it is topological, not a tuning artifact:

| split-mode | pp (prefill) | tg (decode) |
|---|---|---|
| `layer` (pipelined) | **990** | 51.3 |
| `tensor` (parallel) | 692 | **75.1** |

Decode wants both GPUs' bandwidth at once (`tensor`); prefill wants the
sync hidden behind pipelined microbatches (`layer`).

## 5. The non-obvious pathway: draft VRAM ≠ context VRAM

An invariant check — *"every path from `vram headroom` to `single token` passes
through `produce logits`"* — **fails**. There is a second route:

```
vram headroom → extra vram headroom → load mtp draft head → draft head resident
   → propose candidate tokens → verify candidates in one pass
   → several tokens per pass → single token
```

So the two ways of spending VRAM are **not equivalent purchases**:
- VRAM → **context** buys capacity, and by the `long context ↔ kv growth` law
  makes *every subsequent token slower*.
- VRAM → **draft capacity** buys throughput via a shortcut *around* the
  per-token logits bottleneck.

If throughput is the goal, ctx=196608 is on the wrong side of this trade.

## 6. Measured: why deeper speculation fails *here*

Round 1 (`bench/q27-spec-sweep-results.txt`), ctx 32768, code prompt:

| Config | tg | acceptance | drafted | mean len |
|---|---|---|---|---|
| n3, p0.0 *(production)* | **70.44** | 0.679 | 393 | 3.04 |
| n8, p0.0 | 59.94 | 0.383 | 781 | 4.02 |
| n8, p0.6 | 44.33 | **0.816** | 343 | 4.08 |
| n8, p0.75 | 31.57 | — | — | — |
| n8, p0.8 | 35.06 | — | — | — |
| n8, p0.9 | 42.94 | — | — | — |

`--spec-draft-p-min` **worked exactly as documented** — acceptance rose
0.679 → 0.816 and draft waste collapsed 781 → 343 — **and throughput still
fell.** So the cost is not wasted drafts; it is the drafting itself.

MTP drafts **sequentially**: n draft tokens = n forward passes, and every
forward pass crosses the single-cut sync. On this rig each unit of draft depth
buys a sync stall. The U-shape (worst at p0.75, recovering by p0.9) is the
signature: at p0.9 the head bails immediately, approaching no-spec behaviour.

**Why the published result doesn't transfer.** llama.cpp discussion #25198
reports +19.6% from `n-max 16 / p-min 0.8` on Qwen3.6-27B — measured on a
*single* RTX 5090, where a draft step costs no cross-GPU sync.

> **Law:** optimal draft depth is inversely proportional to inter-GPU sync cost.
> Single GPU (sync≈0) → deep drafting wins. Split across a thin link → shallow
> wins. This rig's empirically-found n=3 was right; now we know the mechanism.

This is a falsifiable prediction, not a story: put the model on **one** GPU and
deep drafting should suddenly become profitable. That is what
`bench/q27-sync-sweep.sh` tests.

## 6b. ROOT CAUSE: `-sm tensor` silently disables GPU draft sampling

Chasing *why* drafting is so expensive led to a warning present in **every**
speculative run on this box (24 hits in the round-1 log alone):

```
W spec common_specu: backend offload failed for seq_id=0; using CPU sampler
```

Source, `src/llama-context.cpp:1208`:

```cpp
if (sampler && model.split_mode() == LLAMA_SPLIT_MODE_TENSOR) {
    LLAMA_LOG_WARN("backend sampling not supported with SPLIT_MODE_TENSOR; using CPU");
    return false;
}
```

**Backend (GPU) draft sampling is unsupported with `--split-mode tensor`.** The
production config uses `split-mode = tensor`, so MTP has been drafting with a
**CPU sampler** — a GPU→CPU→GPU round trip per drafted token.

The finding is real and previously unnoticed: the split-mode choice silently
decides **where draft sampling runs**. That coupling appears nowhere in the flag
documentation.

### But it is NOT why deep drafting fails — hypothesis tested and rejected

The obvious next step was: force GPU sampling (use `-sm layer`, which does not
hit that branch) and deep drafting should finally pay. **It does not.**
(`bench/q27-backend-sampling-results.txt`, ctx 32768, code prompt)

| Config | tg | sampler | mean len |
|---|---|---|---|
| `tensor` n3 | **68.50** | CPU | 2.96 |
| `layer` n3 | 47.77 | **GPU** | 2.96 |
| `layer` n8 p0.75 | 41.10 | **GPU** | 3.43 |
| `layer` n16 p0.75 | 40.88 | **GPU** | 3.58 |
| `row` (any) | LOAD_FAILED — *"device CUDA0 does not support split buffers"* on the 5060 Ti |

`-sm layer` really does restore GPU sampling (verified per-run), and deep
drafting is **still** worse. Doubling n-max 8→16 bought 0.15 tokens of mean draft
length and no throughput.

**Actual cause:** the MTP draft head's *useful prediction horizon* is ~3 tokens
for this model on code. Mean accepted length saturates near 3.0–3.6 no matter
what n-max, p-min, split mode, or sampler location you choose. `n-max` above ~3
buys drafts the head cannot get right, so it only adds cost. The CPU-sampler
fallback is a real (and worth-knowing) inefficiency, but it is second-order.

And `tensor` wins overall despite the CPU sampler simply because its decode
bandwidth advantage (68.5 vs 47.8, ≈1.43×) dwarfs the sampling penalty.

> Corrected law: raising draft depth only helps if the draft head's accepted
> length actually grows with it. **Measure `mean len`, not just tg** — it is the
> variable that decides whether more depth is even available to buy.

## 7. Practical consequences

1. **Keep `--spec-draft-n-max 3`, leave `p-min` at 0.0** while tensor-split.
   Do not port the #25198 recipe to this box.
2. **Context costs throughput continuously.** Prefer a working ctx that fits the
   task over 196k-by-default.
3. **`sm=tensor` for interactive generation, `sm=layer` only if prefill-dominated.**
4. Untested and highest-leverage given the sync-bound diagnosis:
   `--poll`, `--prio`, `--no-op-offload` (round 3).

## Reproduce
```
bench/q27-pcie-probe.sh      # topology + baseline + draft acceptance
bench/q27-spec-sweep.sh      # p-min x n-max
bench/q27-sync-sweep.sh      # split-mode, tensor-split ratio, single-GPU
bench/q27-latency-sweep.sh   # poll / prio / op-offload
```

# H3 video — the remaining levers, measured (2026-08-04, second session)

_Continuation of [minimax-h3-video-2026-08-03.md](minimax-h3-video-2026-08-03.md)
and [HANDOVER-2026-08-04-h3-video.md](HANDOVER-2026-08-04-h3-video.md). Same box
(2× RTX 5060 Ti, 31 GB RAM, no P2P), same method: server-measured timings only.
Work happens on ComfyUI branch `h3-2gpu-v0.30.1`; master is untouched and the
deployed `~/.config/opencode/providers/comfyui/server.ini` is not written by any
of this._

## Summary

Most of the handover's open levers are now closed, and three of them closed
*negatively* — which is worth as much as a win, because each was a plausible
multiplier that no longer needs chasing. The one that opened is much larger than
any of them.

| lever | verdict | number |
|---|---|---|
| **head-chunked attention (§10)** | **THE result — the ceiling was never the card** | **removes 3.64 GiB of undocumented sage scratch; native goes 13.0 s → 15.1 s, the model's own maximum** |
| **resolution (960×544)** | **the big one — 2.5×** | **like-for-like at 5.2 s; 832×480 is 3.2×, and both are worth more at 15 s** |
| EasyCache past 0.4 | **saturated** | 0.5 / full-trajectory 0.4 / LazyCache 0.4 all ≡ 0.4 |
| torch.compile | **architecturally unavailable** | Dynamo cannot trace kitchen's DLPack kernels |
| SageAttention 3 (FP4) | **not adoptable here** | fine alone at 5.2 s; OOMs at 15 s, and swaps the box when paired with the deployed cache |
| fixed per-clip cost | **newly quantified** | 83.7 s — 33% of a 10-step run |
| two-GPU split | **unblocked, both cards computing** | not yet a latency win; needs rebalancing |
| device placement (phase D) | **no effect** | 0.8% — freeing 4 GB on the hot card buys nothing, because the allocator does not claim it |

**Read §10 first.** It arrived last and supersedes §7's conclusion that 15 s at
native is impossible. It is not: 15.083 s at 1344×768 rendered in 29.6 min with
zero fallbacks. Every ceiling in this document was an allocation, not a card.

The box also now fits in one equation (§3b) that predicts every measured
single-GPU row within 2.6 s — including sage3's end-to-end time to within 0.8 s
from a kernel microbenchmark alone.

Three process defects were also fixed, all of which had been silently corrupting
results:

- the benchmark harness charged the whole ~24 s of checkpoint staging to the
  first row of each shared-server phase — which is the no-cache baseline, so the
  headline EasyCache win is **−26%, not −31%** (§5b);
- the memory watchdog killed seven healthy runs out of eight: it kept testing INSTANTANEOUS readings — memory level, swap level, swap rate — every one of which normal operation here produces several times a minute (§6);
- the analyser published an unopposed single row as a phase winner, declaring
  `(defaults)` the attention-backend winner while the four-row probe stage
  showed sage2 ahead of defaults by 12% (§8).

## 1. Step caching is done; 0.4 was already the right answer

The handover expected headroom past the deployed EasyCache 0.4. There is none.

| row | config | server s | vs deployed 0.4 (252.6 s) |
|---|---|---:|---:|
| K2 | EasyCache 0.4 (deployed) | 252.6 | — |
| K5 | EasyCache 0.5 | 253.0 | +0.2% |
| K7 | EasyCache 0.4, full trajectory (0.05–1.0) | 252.2 | −0.2% |
| K8 | LazyCache 0.4 | 253.1 | +0.2% |
| K6 | EasyCache 0.6 | 227.6 | **−9.9%** |

Three different ways of asking for more caching return the identical number, so
the cache is skipping every step it safely can at 0.4 and the threshold is no
longer the binding constraint. Only 0.6 moves the clock, and it is the row the
matrix flags for visible quality loss — it skips more steps by tolerating more
trajectory drift, so it must be judged on output, not adopted on the strength of
10%.

**Consequence:** the deployed configuration was already at the top of this
lever. Nothing to change.

## 2. torch.compile cannot run this model

K9 (`TorchCompileModel`, inductor) fails during tracing:

```
Dynamo failed to run FX node with fake tensors: call_method __dlpack__(
  *(FakeTensor(..., device='cuda:0', size=(37740, 28672), dtype=torch.bfloat16),),
  **{'stream': -1}): Cannot access data pointer of Tensor (e.g. FakeTensor...)
```

`comfy_kitchen` hands tensors to its own CUDA kernels through DLPack capsules
(`_wrap_for_dlpack` → `tensor.__dlpack__(stream=-1)`, needed to avoid a
default-stream sync that would break graph capture). A DLPack export requires a
real data pointer; a FakeTensor has none by construction. So every quantized
linear in this model is an opaque wall to Dynamo.

This is not a tuning failure and no backend or flag works around it — the fix
would be upstream, wrapping kitchen's kernels as custom ops with registered fake
implementations. **torch.compile is off the table for quantized H3.** Worth
knowing before anyone spends a night on `mode="max-autotune"`.

## 3. SageAttention 3 — the selector was broken, and the win is real but modest

### The bug: our own node, 100% fatal, never previously executed

K10/K11 had been wired but never run. Both died in ~1 second with
`RecursionError: maximum recursion depth exceeded`, in
`custom_nodes/sage3_select`:

```python
def override(func, *args, **kwargs):
    kwargs.pop("_inside_attn_wrapper", None)   # ← the bug
    return fn(*args, **kwargs)
```

`wrap_attn` sets `_inside_attn_wrapper=True` immediately before invoking the
override; it is the guard that stops the override being consulted again. The
node stripped it, so the wrapped sage3 it called re-entered the override, which
called sage3, forever. The key has to survive two hops — `fn` is the *wrapped*
function, and `attention3_sage` falls back to `attention_pytorch(..., **kwargs)`
for masked/small/short inputs, which would re-enter too. Fix is to pass kwargs
through untouched.

### Measured at H3's real attention shape

A full sweep row buries the attention kernel under model loading, 49 other
blocks, VAE decode and mux. `bench/attn-micro.py` times the backends alone at
the true geometry (56 heads × 128 dim, one packed sequence, bf16) and measures
error in the same run — because a backend that quantises V cannot be judged on
the clock alone.

**5.2 s clip (37,710 tokens), per attention call:**

| backend | ms/call | vs torch SDPA | mean abs error |
|---|---:|---:|---|
| pytorch SDPA | 903.8 | — | reference |
| sage2 (INT8 QK, FP16 V) | 304.3 | −66.3% | 3.2% of signal sd |
| sage3 (FP4 QKV) | 245.0 | **−72.9%** | **13.0% of signal sd** |

sage3 is **19.5% faster than sage2** and carries **~4× the error**. (Inputs are
random gaussian, the pessimal case for a quantised kernel — real activations
have structure. Treat the error column as an upper bound.)

### What that is worth end to end — and the microbenchmark was right

Attention is 50 blocks × one call per block per step. At sage2's 304 ms that is
15.2 s of the measured 28.3 s step — **54% of step time**, which independently
corroborates the 58%-of-FLOPs estimate in the original findings.

The end-to-end row then confirmed the projection almost exactly:

| row | config | s/it (sampler) | server s |
|---|---|---:|---:|
| K0 | sage2, no cache, 10 steps | 28.3 | 366.7 |
| K10 | **sage3**, no cache, 10 steps | **24.33** | **336.4** |

**−14.0% per sampling step, −8.3% end to end** — against a microbenchmark
prediction of −14.7% per step. The gap between the two figures is the 83.7 s
fixed block in §5, which sage3 cannot touch. Zero kernel fallbacks were logged,
so the FP4 path ran on every one of the 500 attention calls.

| clip | measured saving |
|---|---:|
| 5.2 s, 10 steps | **−8.3% (measured)** |
| 15.1 s, 20 steps | **none — it cannot run there. See below.** |

Real at short clip length, but not the 2–5× the vendor headline suggests, and
bought with a precision cut in V. A clip to look at is what should decide it —
see `bench/quality.py`.

### sage3 does not work at 15 s, and it fails silently

The same microbenchmark at the 15.1 s shape (106,930 tokens):

| backend | ms/call | vs torch | error |
|---|---:|---:|---|
| pytorch SDPA | 7271.8 | — | reference |
| sage2 | **2499.3** | **−65.6%** | 3.2% of signal sd |
| sage3 | 7334.2 | +0.9% | 0.1% of signal sd |

sage3 is *slower than PyTorch* and its error has vanished. Both are the same
symptom: it never ran. The log carries five copies of

```
Error running SageAttention3: CUDA out of memory. Tried to allocate 9.33 GiB.
  ... falling back to pytorch attention.
```

`attention3_sage` catches kernel failures and quietly reroutes to
`attention_pytorch`, so the benchmark was comparing PyTorch against itself — the
0.1% residual is float non-determinism, not FP4 accuracy.

**sage3 needs ~9.33 GiB of workspace at 106,930 tokens, on top of 4.28 GB of
q/k/v.** That does not fit a 15.5 GiB card, and in a real generation the DiT is
also resident, so it fits even less well. The crossover between 37,710 tokens
(works) and 106,930 (does not) is untested.

Three consequences:

1. **My own earlier projection of ~15% at 15 s was wrong**, and wrong in the
   most tempting direction — extrapolating a kernel's benefit into a regime
   where the kernel cannot allocate. The §3b cost model has no way to know a
   backend will OOM.
2. **Enabling sage3 for long clips is actively harmful.** The fallback is
   PyTorch SDPA at 7271.8 ms against sage2's 2499.3 ms — **2.9× slower per
   attention call** — while the workflow reports success and the node logs that
   sage3 was selected. Anyone adopting it for 15 s clips would silently lose the
   single largest optimisation on this box.
3. sage2 is unaffected and scales cleanly: −65.6% at 15 s versus −66.3% at
   5.2 s, same 3.2% error. It stays the correct default everywhere.

Incidentally the quadratic model passes again on the PyTorch reference:
903.7 → 7271.8 ms for 2.836× the tokens is **8.05×**, against 8.04× predicted.

### sage3 partly cancels EasyCache — watch for this

K11 (sage3 **+** EasyCache 0.4) does not simply stack. The server log shows the
cache skipping **3 of 10** steps under sage3 where it skips **4 of 10** under
sage2. EasyCache decides by measuring how far the trajectory moved between
steps; FP4's larger numerical error inflates that measurement, so the cache
becomes less willing to reuse. One fewer skipped step costs ~24 s and eats most
of the ~30 s sage3 saves.

Arithmetic on the measured constants (83.7 + 7 × 24.33 ≈ 254 s) puts the
combination within noise of the deployed sage2 + cache 0.4 at 252.6 s — **a
wash** on speed.

It is worse than a wash on memory. K11 has now been attempted twice and drove
the box into swap both times. The second attempt is unambiguous — the watchdog
killed it at:

```
MemAvailable=125MiB  swap+3089MiB
```

125 MiB from nothing with 3.1 GB paged out: that is the freeze signature of
§6, and the recalibrated guard caught it on the critical-floor rule. (Worth
noting the recalibration did not weaken the protection — the two false kills
were at 2015 MiB and 1092 MiB with swap flat or falling; this one is a real
save.)

The combination is the problem, not either half. **K10 (sage3, no cache)
completes at 336.4 s. K2 (sage2 + cache 0.4) completes at 253.5 s. Only the two
together fail.** The plausible mechanism is additive VRAM pressure — sage3's FP4
workspace plus the reference activations EasyCache retains between steps exceed
what the card can hold alongside the DiT, so ComfyUI evicts weights to a 31 GB
host that has no room for them, and the machine pages.

**Verdict on sage3, revised down twice in one session:** it works alone, at
short clip length. It cannot run at 15 s (OOM, silent fallback to PyTorch,
2.9× slower than sage2 — see above), and it cannot run with the deployed cache
at any length. Since the deployed configuration is sage2 **plus** EasyCache 0.4,
**sage3 is not adoptable here** — the only configuration it improves is one
nobody runs. The node and the measurements stay; the conclusion is negative.

This is a general warning for any quantised-attention plus adaptive-cache
combination: the cache's skip decision is a function of numerical error, so a
faster, noisier kernel silently buys back fewer skips.

## 3b. The whole box now fits in one equation

The constants from §3, §4 and §5 compose into a predictor
(`bench/costmodel.py`):

```
server_s = FIXED + real_steps x step(tokens)
step(n)  = 0.54 x 28.3 x (n/n0)^2   +   0.46 x 28.3 x (n/n0)
           \_ attention, quadratic _/     \_ everything else, linear _/
FIXED = 84.0 s,  n0 = 37,710 tokens
```

Nothing in it is fitted to the rows it predicts: 28.3 s/step and 84.0 s come
from the K0/N0 pair, the 0.54 attention share comes from the standalone kernel
benchmark (50 blocks × 304 ms / 28.3 s), and the quadratic exponent is the
verified one from §4. Against every single-GPU row measured on this box:

| row | config | measured | predicted | residual |
|---|---|---:|---:|---:|
| K0 | no cache | 366.7 | 367.0 | −0.3 |
| K1 | EasyCache 0.2 | 284.7 | 282.1 | +2.6 |
| K2 | EasyCache 0.4 | 253.5 | 253.8 | −0.3 |
| K3 | LazyCache 0.2 | 283.2 | 282.1 | +1.1 |
| K4 | EasyCache 0.3 | 255.0 | 253.8 | +1.2 |
| K5 | EasyCache 0.5 | 253.0 | 253.8 | −0.8 |
| K6 | EasyCache 0.6 | 227.6 | 225.5 | +2.1 |
| K7 | 0.4, full trajectory | 252.2 | 253.8 | −1.6 |
| K8 | LazyCache 0.4 | 253.1 | 253.8 | −0.7 |
| **K10** | **sage3** | **336.4** | **337.2** | **−0.8** |
| N0 | 1 step | 112.2 | 112.3 | −0.1 |

Every row within 2.6 s. The K10 line is the one to notice: sage3's end-to-end
time is predicted to **0.8 s** by taking the model and scaling only its attention
term by 245.0/304.3 — the ratio from a microbenchmark that never generated a
frame. Kernel-level measurement transfers to production time exactly, which is
what makes `bench/attn-micro.py` a legitimate substitute for a 6-minute row when
screening attention backends.

A useful by-product: solving each cached row for its real step count recovers
integers to within 0.1 (`(284.7−84)/28.3 = 7.09`), which is an independent check
on both constants — and it shows thresholds 0.3, 0.4 and 0.5 all skip exactly 4
of 10 steps, which is the saturation of §1 seen from another angle.

### What it says about the 15.1 s clip

At 362 frames the packed sequence is 109,062 tokens, 2.89× the 5.2 s clip:

| config | predicted |
|---|---:|
| 20 steps, EasyCache 0.4 (deployed) | **34.5 min** |
| 20 steps, EasyCache 0.4 + sage3 | **29.5 min** |
| 10 steps, EasyCache 0.4 | **17.9 min** |

The deployed figure lands inside the previous session's 35–40 min extrapolation,
but now derived from measured constants rather than scaled from a shorter run.
The 15 s validation run is queued and will test it directly.

## 4. The cost model is exact

The 15.1 s shape (106,930 tokens) is 2.836× the 5.2 s token count. Attention is
quadratic, so it should cost 8.04× as much. Measured on torch SDPA: 903.8 ms →
7259.2 ms = **8.03×**.

The quadratic model that every extrapolation in these documents rests on — the
15 s estimates, the resolution-scaling argument, the 80%-attention-at-15s claim
— is now verified to 0.1% rather than assumed.

## 5. A third of a short clip is not sampling

Nothing in the matrix measured what happens outside the sampler. Solving the two
cache rows as a linear model (K0 runs 10 real steps; K2 skips 4 of 10, so 6):

```
K0 = fixed + 10·step = 366.7 s
K2 = fixed +  6·step = 253.5 s
  ⇒ step = 28.3 s,  fixed = 83.7 s
```

**83.7 s per clip is not sampling** — model load/evict transitions, VAE decode,
audio decode, mux. That is **33% of a tuned 10-step run** and **20% of the
deployed 20-step configuration**, and it is invisible to every knob measured so
far, all of which act on the per-step term.

It matters most exactly where the current tuning is strongest: the better
caching and attention get, the larger this fixed block looms.

### It is the VAE decode, essentially all of it

Phase N differences the full graph against one that stops at the sampler and
writes the latent instead of decoding it. Both rows under identical flags:

| row | runs | server s |
|---|---|---:|
| N0 | full graph, 1 step | 103.1 |
| N1 | same, stopping at the sampler — no VAE decode, no mux | **29.8** |

**73.3 s of difference.** N1 still performs the model load, the text encode and
a complete sampling step, and finishes in under 30 seconds. So the fixed block
is video VAE decode + audio decode + mux, and the load/evict transitions — the
other candidate, and the one the offload flags would address — are almost
nothing.

Three consequences:

1. **Phase H is the right lever, phase C is not.** `--bf16-vae`,
   `--fp16-intermediates` and `--cpu-vae` act on the term that exists; the
   offload and memory flags act on one that barely does.
2. **Phase D gets stronger.** Moving the 5 GB video VAE to the idle card
   attacks the largest single block of non-sampling time, and it does so on the
   card that is not already holding a 21 GB DiT.
3. **It explains the watchdog.** All six trips this session happened at the VAE
   decode, which is now identified as both the bulk of the fixed cost and the
   peak of memory pressure — including the one true positive, where the two-GPU
   split's decode drove the box into thrash (§7).

A caveat on comparing these numbers to §3b's: this phase ran with `--fast` in
the base flags where the K rows did not, which is worth about 5%. Within the
phase every row shares flags, so the differencing holds; across batches it does
not. `bench/costmodel.py` now marks rows whose flags differ from the constants'
so the mistake is visible rather than silent.

### A harness constraint discovered along the way: three rows per server

Phase N never got past its third row. Both attempts completed N0 and N1 and
were then killed at N2 by *correct* watchdog fires — `MemAvailable=204MiB` with
swap +3425 MiB the second time. Host RAM accumulates across prompts inside one
ComfyUI server, and on 31 GB the third execution is where it runs out.

That is a property of the harness, not of any setting under test, and it is
dangerous in a specific way: `sweep.py` amortises one server across a whole
workflow phase, so **the rows most likely to die are rows 3+ — and whether a
phase's conclusion survives depends on the order its rows happen to be listed
in.** Phase D's decisive comparison is D0 against D1, rows 1 and 2, which would
have survived; its remaining three rows would not have.

Phases D and L, and `bench/quality.py`, now run **one server per row**. It costs
a ~90 s boot each and buys a clean memory baseline — and as a side effect every
row becomes equally cold, which removes the §5b first-row penalty by
construction rather than by a warm-up hack. The launch phases (A/B/C/H/I) have
always worked this way, which is why they were never affected by either problem.

The essential N measurement was already in hand when this hit, so N2/N3 were
dropped rather than chased: N0−N1 is the decomposition, and the first-row
penalty had already been measured independently.

## 5b. The first row of a shared-server phase pays ~24 s that is not its fault

Phase N was meant only to attribute the fixed cost. It found a measurement bug
instead.

| row | steps | server s |
|---|---:|---:|
| N0 | 1 | 112.2 |
| N2 | 3 | 144.8 |

Differencing those gives 16.3 s/step — against 28.3 s/step from K0/K2. Both
cannot be right. The clean pair settles it: **K0 (10 steps) and N0 (1 step) were
each the first row executed on a fresh server**, no cache, same flags, so
whatever the constant is, it cancels:

```
(366.7 - 112.2) / 9 = 28.3 s/step,  fixed = 84.0 s
```

which reproduces the K0/K2 figures (28.3, 83.7) from independent rows. So 28.3
is right, and N2 is the anomaly — it ran *third* on that server, with the
checkpoint already staged. The cold model predicts 84.0 + 3×28.3 = 168.8 s; N2
measured 144.8 s.

**The first generation on a fresh server costs ~24 s more than the same
generation once the models are hot.**

That is worth knowing operationally — the first clip after starting the panel's
server is ~24 s slower, and nothing is wrong. But it is more important as a
*methodology* defect: `sweep.py` runs an entire workflow phase on one server, so
the penalty lands entirely on row 1 and reads as though row 1's setting were
slower. Row 1 of phase K is `K0`, the **no-cache baseline every cache number is
measured against**:

| | as reported | corrected for the penalty |
|---|---:|---:|
| EasyCache 0.4 vs no cache | 366.7 → 252.6 = −31% | ~343 → 252.6 = **−26%** |

The cache is still the single largest tuning win on this box; it is ~26%, not
~31%. (Launch phases such as A, B, C and H restart the server for every row, so
every row is equally cold and their rankings were never affected.)

Two changes: `sweep.py --warmup` runs one discarded 1-step generation before a
workflow phase's rows, and phase N gained **N3**, an identical repeat of N0 run
warm, so `N0 − N3` measures the penalty directly rather than inferring it from a
model. Phases D, H and the L re-run now use `--warmup`; L had already run
without it, so its first row needs the correction.

### The 24 s was later confirmed by prediction, not just by arithmetic

The figure above comes from one comparison (N2 against a cold-derived model),
which is thin support for a number that revises a published result. It was
subsequently tested properly, on a row it was never fitted to.

When phase L was re-run with one server per row, every row became cold, so the
old *warm* L2 measurement should be recoverable by adding the penalty back and
applying the `--fast` factor:

```
predicted = (141.65 + 24) x 0.95 = 157.4 s
measured  =                        156.5 s      (0.6% error)
```

The prediction was made before the run. It transplants a constant derived at
1344×768 onto a 960×544 row and still lands, which is what a *checkpoint
staging* cost should do — it depends on the model, not the canvas. So the
penalty is ~24 s, additive, and geometry-independent, and **the −26% EasyCache
correction rests on a verified constant rather than an inferred one.**

The general lesson is worth stating: any harness that amortises expensive setup
across a batch quietly bills that setup to whichever measurement went first.

## 6. The watchdog was killing healthy runs — seven times out of eight

The guard exists because an unguarded `--novram` run froze this box on
2026-08-04 and cost a hard reset. But it was calibrated on that one event, and
it spent this session executing healthy benchmarks.

**First kill:** phase K died at `MemAvailable=2015MiB` with swap flat — 33 MiB
short of the 2 GB floor. The VAE-decode stage of a normal run dips there for one
2-second sample and recovers.

**Second kill:** with a 1.2 GB hard floor added, K11 died at its VAE decode at
`MemAvailable=1092MiB` — with swap grown just **173 MiB**. Nowhere near thrash;
simply a large allocation succeeding.

**Third and fourth kills:** with a 700 MiB critical floor added, the Q1 split
re-run died at `MemAvailable=515MiB` with swap **falling 1685 MiB** — the kernel
was reclaiming successfully and the run was killed for it. (In between, the
sage3 + cache row was killed at 125 MiB with swap +3089 MiB, which was correct
and is discussed above.)

Four kills, three of them wrong, every one at the same VAE-decode dip. The
diagnosis was the same category error each time, and it took three attempts to
state it properly:

**MemAvailable is not the danger signal, and neither is any threshold on it.**
It already discounts reclaimable page cache, and H3's VAE decode legitimately
drives it to 0.5–2 GB on this 31 GB box on *every* run. What froze the machine
was *thrashing* — overcommit with swap climbing until kswapd stalled. It is also
worth being clear about what we are not defending against: an ordinary OOM kill
is not the failure mode. The kernel reaps the process and the box carries on;
for a benchmark that is a failed row, not a disaster.

Swapping to swap-growth as the discriminator then produced a **fifth** kill, of
the opposite kind: the Q1 re-run died on a bare `swap > 2 GB` rule at
`swap+2583MiB` with **MemAvailable at 12,110 MiB**. Nothing was wrong. zram is
*compressed RAM*, and the kernel moves cold anonymous pages into it
opportunistically; growth of 2.6 GB with 12 GB free is housekeeping.

That is the actual lesson, and it took five kills to state correctly:

> **Neither variable means anything on its own, in either direction.** Low
> memory with swap flat or falling is a big allocation succeeding, or the kernel
> reclaiming — normal for the VAE decode on every clip. Swap growth with memory
> plentiful is zram housekeeping. Only the two *together* mean trouble.

```
MemAvailable < 0.3 GB                            → backstop, fire regardless
MemAvailable < 2 GB AND swap +> 0.3 GB, 2 ticks  → low AND paging: thrash
swap +> 4 GB AND MemAvailable < 6 GB             → runaway backstop
```

Checked against all five recorded kills plus the original freeze and three
synthetic distress cases: every wrong kill now passes, every genuine one fires,
and the freeze signature (swap climbing continuously with memory exhausted)
trips the second rule within four seconds — earlier than any previous
calibration managed. The protection is not weaker; it is finally aimed at the
right thing.

### And then it killed a run on a reading that never happened

A sixth kill, of a kind the thresholds could not explain:

```
[memguard] TRIPPED: MemAvailable=0MiB (low_ticks=0) swap+-2427MiB
```

Nothing available while 2.4 GB was being *freed* is not a state a machine can be
in. Both numbers came from `awk` over `/proc/meminfo`, one per sample — and
**fork is exactly what starts failing under the memory pressure this script
watches for.** A failed fork returns an empty string, bash arithmetic evaluates
`""` as `0`, and `(( 0 < 300 ))` fires the absolute floor. The guard killed a
healthy phase-N re-run on a measurement it never took.

```bash
avail=""; (( avail < 300 )) && echo "fires"   # prints: fires
```

Both readings are now validated as integers before any comparison, and an
unusable sample skips the tick instead of being believed. Five consecutive
unreadable samples (10 s of being unable to fork a trivial process) is itself
treated as distress and does kill — that much is a real signal.

### The rewrite that should have been the first one: measure the rate

A seventh kill finally made the shape of the mistake obvious. Phase D's D0 —
the plain baseline, first row on a *fresh* server, so no accumulation — was
killed at `MemAvailable=1505MiB, swap+3357MiB`, **during model staging**. The
server log shows why: ComfyUI had just staged a 14,956 MB text encoder followed
by a 19,995 MB DiT. That is ~35 GB of working set on a 31 GB box. Paging is not
a symptom there; it is the kernel doing its job, and the original guard's own
comment had said as much about a smaller version of the same event.

So every level-based rule was wrong in principle, not just in its constants,
because **the levels being tested are what normal operation looks like on this
machine**:

| observed | when | verdict |
|---|---|---|
| MemAvailable 1.5–2.5 GB | staging 35 GB of models | normal |
| swap +1.5 to 3.5 GB | that same staging, self-stabilising | normal |
| MemAvailable 0.5–2 GB, swap falling | VAE decode, every clip | normal |

What froze the box was none of those. It was +4 GB in 20 s *and still climbing*,
with memory exhausted and staying exhausted. The obvious next move was to
measure the **rate**, so that is what I tried — and it was wrong too, in a way
worth recording because it looks so reasonable:

```
[memguard] TRIPPED: MemAvailable=14019MiB swap+221MiB rate=2133MiB/8s
```

Fourteen gigabytes free, 221 MB of net growth, and an apparent 2.1 GB/8 s
spiral. **zram is compressed RAM: pages churn in and out continuously, so the
instantaneous rate oscillates by gigabytes while nothing at all is wrong.** An
earlier trip caught the same thing from the other side — a genuine staging burst
momentarily hitting the recorded freeze's rate. Rate cannot separate them
either.

What does separate them is **duration and total**:

```
MemAvailable < 150 MB                   → backstop, fire
MemAvailable < 400 MB for 30s straight  → starvation that is not a dip
swap +> 6 GB total over the run         → past anything healthy here (max 3.6 GB)
5 unreadable /proc/meminfo samples      → cannot fork; that IS distress
```

The freeze crosses both the 6 GB total and the 30 s starvation rule; a staging
burst crosses neither, because it recovers. Simulated against all eight recorded
events plus the freeze trace and synthetic cases: four healthy patterns pass,
three distress patterns fire. The final script is 118 lines and shorter than
several of the versions it replaced.

The wider point is not about this script. A watchdog calibrated on a single
incident encodes that incident's *symptoms* rather than its *mechanism*, and
then spends its life killing healthy work that shares a symptom — here, six
times, because "low memory" and "swap grew" are both just what a 35 GB workload
looks like on a 31 GB box. Worse, a monitor is code that runs *hardest* when the
system is least able to run it, so its own failure modes correlate with the
emergency it watches for; it must distinguish "I measured trouble" from "I
failed to measure". Seven kills, six of them wrong, to arrive at three rules and
one type check — and the working version is shorter than the version it
replaced.

**All three kills happened at the VAE decode**, which makes that stage — not
model loading, not sampling — the thing that walks this box to the edge of host
memory. That is a finding in its own right, and it points at the same fix as §5
and phase D: the 5 GB video VAE should not be decoding on the card that is
already holding a 21 GB DiT while the other card sits idle. It also predicts
trouble for the 15.1 s clip, whose VAE decode handles ~2.9× the frames.

## 7. Two-GPU split: root cause located

The blocker was
`BufferError: Can't export tensors on a different CUDA device index. Expected: 0.
Current device: 1.` Read carefully, that says the tensor is *labelled* device 0
while the kernel launches on device 1 — so the question is what produces a
device-0-labelled tensor inside the cuda:1 half.

There is exactly one construct in the stack that can make a tensor whose label
disagrees with its memory: `comfy_aimdo.torch.aimdo_to_tensor(alloc, device)`
builds a tensor from a raw pointer via `__cuda_array_interface__` with an
explicit `device=`, and torch trusts the label. It is called with the *caller's*
device, not the allocation's.

The caller that gets this wrong is the vbar prefetcher. `comfy/ldm/minimax/model.py`
creates the prefetch queue with the model-level device (cuda:0) and
`prefetch_queue_pop` stages every upcoming block with it — including blocks
whose vbars this node rebound into the cuda:1 arena. A rebound module is
resident after placement, but `uncast_bias_weight` unpins its pages after each
use, so once VRAM pressure evicts them the prefetcher stages the re-transfer
into `aimdo_to_tensor(s._v, cuda:0)`: cuda:1 arena memory wearing a cuda:0
label. The block hook then computes under `torch.cuda.device(cuda:1)`, kitchen
exports that tensor via DLPack, and torch refuses.

Three changes, all in `custom_nodes/h3_dualpipe`:

1. **Skip vbar prefetch for split models only.** A split model plants
   `h3_dualpipe_split` in `transformer_options`; a narrow wrapper around
   `make_prefetch_queue` returns `None` for exactly those. Transfers then happen
   inside `cast_bias_weight`, whose device is the block *input's* — correct on
   both halves. Resident modules never transfer, so a fully-resident split pays
   nothing; an evicted one pays a synchronous copy instead of an overlapped one.
   (Deliberately not the global DLPack device-guard that ComfyUI-MultiGPU uses —
   that one "fixes" the export by running it on the tensor's labelled device,
   which is right for a mislabel and wrong for a genuine cross-device tensor,
   and it is how MultiGPU breaks int8_linear.)
2. **Move the plain params and buffers too.** The loader force-loads every
   module under 16 KB as ordinary parameters on the load device and puts all
   buffers there — in this DiT that is the norms and modulation vectors. Vbar
   rebinding left them on cuda:0, so each cuda:1 block re-copied them across a
   bus with no P2P every step.
3. **Stop keying the per-forward activation cache on `id(rope)`.** CPython
   reuses freed addresses, so a later forward could match a stale id and reuse a
   previous step's `t_emb`. It is now keyed on reaching the boundary block,
   which is the deterministic per-forward event.

Plus a fault-failure check (`vbar_fault` returning `None` on arena OOM was being
stamped as a valid signature) and a one-shot diagnostic that dumps the device
labels, plain-param placement and live residency flags if a cuda:1 block raises
— so the next failure, if there is one, is readable without another bisection.

### It runs

Q1 completed **10/10 sampling steps**. No `BufferError`. The placement log:

```
[H3PipelineSplit] vbar prefetch guard installed (prefetch skipped only for split models)
[H3PipelineSplit] armed: split at block 22, cuda:0 -> cuda:1
[H3PipelineSplit] vbar-rebound 140 modules of blocks 22..49 to cuda:1 arena;
                  moved 112 plain params/buffers to cuda:1
```

Note the **112** plain params and buffers. Change (2) was not defensive
tidying — there really were 112 tensors inside the cuda:1 blocks still sitting
on cuda:0, every one of them crossing a bus with no P2P on every step.

Utilisation sampled every 2 s through the run: **gpu0 busy 39%, gpu1 busy 47%**,
alternating — caught mid-run at gpu0 0% / gpu1 100%. Both cards compute, the
whole DiT is VRAM-resident across them (11.0 GB + 15.65 GB), and the offload tax
is gone. **The T2 blocker is cleared.**

### But at 22/28 it is slower for one clip, and here is why

Sampling took 204 s against single-GPU's 170 s for the identical config — about
20% slower per clip. Two things explain it, and they point in different
directions:

- **The split is inherently serial for a single clip.** GPU0 computes blocks
  0–21, then GPU1 computes 22–49. Each card idles while the other works, which
  is why neither exceeds ~47% busy. This was known and is not a defect; the
  split's payoff is meant to be the removed offload tax, not parallelism.
- **cuda:1 is full.** 28 of 50 blocks put it at **15.65 GB of 15.5 GiB usable**.
  An arena with no free pages has to evict and re-fault, which reintroduces
  exactly the host-RAM streaming the split exists to eliminate — and cuda:0,
  carrying only 22 blocks plus embeddings and final_layer, sits at 11.0 GB with
  room to spare. The cut is in the wrong place: an even *block* count is not an
  even *memory* split, because cuda:0 also owns the embeddings, token refiner
  and final layer.

Phase R (added to the matrix: splits at 25, 28, 31, 34) measures where to cut.
Note the direction is counter-intuitive — *more* blocks on cuda:0, not fewer,
because cuda:1 is the constrained side once the non-block modules are counted.

### And its VAE decode does not fit in 31 GB of host RAM

Q1 has now been attempted three times. **All three sampled cleanly** — 10/10
steps, both cards alternating, no BufferError. **All three died at the VAE
decode.** The third attempt was killed by the corrected watchdog on both
variables at once:

```
MemAvailable=704MiB  swap+3577MiB  (sustained, 2 ticks)
```

That is a true positive — memory exhausted *and* 3.6 GB paged out and climbing.
Unlike the four false kills of §6, nothing here is a calibration artefact: the
split genuinely walks this box into thrash at the decode stage.

The mechanism follows from what the split does. Sampling is fine precisely
because the weights are resident on both cards. But the decode then needs ~5 GB
for the video VAE on cuda:0, and the DiT blocks occupying it cannot be evicted
cheaply — this node's placement faults and *pins* the cuda:1 arena and never
unpins it, and cuda:0 is carrying 11 GB of its own — so the pressure lands on a
host that is already holding the mmap'd checkpoint and the pinned transfer
buffers. Single-GPU runs have the same decode but far more evictable VRAM to
absorb it.

So the split's constraint is **host RAM at decode time**, not VRAM during
sampling — which is the same 31 GB ceiling the previous session identified as
the root cause of every two-GPU difficulty, arriving from a new direction.

Three things could move it, in increasing order of effort: a lower
`split_index` (phase R — fewer blocks pinned on cuda:1, so more of the card
stays evictable), `--disable-smart-memory` or `--cache-none` to shrink host-side
caching, or the 64 GB kit.

**What will not work is simply unpinning the arena after placement**, which is
the obvious-looking fix and is wrong. The pinning is load-bearing: the eager
fault-in allocates *fresh* vbar space, and comfy's lazy re-fault path cannot
serve a fresh allocation — its pin and file-slice descriptors were registered
against the original load-time allocs, so a re-fault falls through to
`HostBuffer.read_file_slice` and dies. Pinned pages are what guarantee the
signature keeps matching and compute keeps taking the resident fast path.
Unpinning would convert a memory-pressure event from a watchdog kill into a hard
crash. Any real fix has to give the rebound allocations a working re-fault path
first.

The honest summary: the split is a working capability that does not yet pay for
itself on single-clip latency, and currently cannot complete a full clip on this
box at a 22/28 cut. Its real prizes remain the VRAM headroom that makes long
clips viable and micro-batching two clips at once — the only path to true 2×
throughput here that does not require buying RAM.

## 7b. Resolution is the biggest lever on this box, and it beats its own model

Phase L had never been run. It is worth more than everything else in this
document combined.

Re-run with **one server per row**, so every row is cold under identical
conditions rather than the mixture of warm and cold rows the first pass
produced. 5.2 s clip, 10 steps, no cache, `--use-sage-attention --fast`:

| row | canvas | tokens | as measured | × | excl. startup | × |
|---|---|---:|---:|---:|---:|---:|
| — | 1344×768 (native) | 37,710 | 355.8 s | 1.00× | 331.8 s | 1.00× |
| L1 | 1152×640 | 27,054 | 234.2 s | 1.52× | 210.2 s | 1.58× |
| **L2** | **960×544** | 19,284 | **156.5 s** | **2.27×** | **132.5 s** | **2.50×** |
| L3 | 832×480 | 14,844 | 129.2 s | 2.75× | 105.2 s | 3.15× |
| L4 | 768×432 | 11,958 | 109.6 s | 3.25× | 85.6 s | 3.88× |

**Both columns are correct and they answer different questions.** The gap
between them is the ~24 s of checkpoint staging from §5b, which is paid once per
*server*, not once per clip. Generate several clips in a session and only the
first pays it, so the right-hand column is what a working session actually sees;
a cold one-off sees the left. The speedups compress when staging is included for
the same reason every other lever compresses — a fixed cost that does not scale
with canvas dilutes any saving that does.

(The native row is D0, which is the same configuration on its own server. The
first pass reported 2.42× for L2 by comparing a cold-corrected native against
warm low-res rows; that landed inside the honest band but was not derived
cleanly. 2.5× is the like-for-like figure.)

960×544 is a **2.5× speedup** and 832×480 is **3.2×** — and the quadratic term
grows with clip length, so all of these are worth more at 15 s, which is exactly
where it is needed.

### It outruns the cost model, and the reason matters

The §3b model predicts 190.5 s for L2. It measured 141.7 s — **49 s fast**,
the only large negative residual in the whole table. The explanation is in the
VRAM column:

| row | canvas | peak cuda:0 |
|---|---|---:|
| K0 | 1344×768 | 13.73 GB |
| L2 | 960×544 | **15.19 GB** |

The smaller canvas uses *more* VRAM, not less. Smaller activations leave more
room, so **more of the 21 GB DiT stays resident** and less of it streams from
host RAM every step. The model's linear term was calibrated at native
resolution, where that offload tax is baked in; shrinking the canvas buys the
geometry saving *and* buys residency, and the second effect is not in the
equation.

So on a box where the model does not fit in VRAM, resolution scaling compounds:
fewer tokens, and a larger fraction of the weights staying put. That also means
these numbers will **not** transfer to a card that already holds the whole
model — there the saving would be the model's 190 s, not 142 s.

### The residual measures the offload tax — a number nothing else could reach

Run the residual across the whole phase and it does not scale with the canvas.
It **saturates**:

| row | canvas | predicted | measured (warm) | residual | peak cuda:0 |
|---|---|---:|---:|---:|---:|
| L1 | 1152×640 | 256.0 | 222.1 | −33.9 | 13.73 GB |
| L2 | 960×544 | 190.5 | 141.7 | −48.9 | 15.19 GB |
| L3 | 832×480 | 158.9 | 106.9 | −52.0 | 15.04 GB |
| L4 | 768×432 | 140.6 | 87.6 | −53.1 | 15.09 GB |

A residual that grows and then stops at ~50 s over 10 steps is not a modelling
error — it is a *fixed cost being removed*. Once the canvas is small enough for
the DiT to stay resident, there is nothing left to remove, so shrinking further
buys only geometry.

That puts a number on something previously only described qualitatively:

**≈5.0 s of the 28.3 s step — 18% — is streaming weights from host RAM rather
than computing.**

It is measured, not inferred from PCIe bandwidth, and it could not have been
obtained by any single configuration: it took a *series* of geometries and a
model accurate enough that a 50 s gap was obviously structural rather than
noise.

It also yields a falsifiable prediction for the two-GPU work. The split's entire
thesis is that it makes the whole DiT VRAM-resident across two cards and thereby
deletes this same tax. If that thesis is right, a **properly balanced** split
should recover ≈50 s per 10-step clip at native resolution — and Q1's failure to
do so is then fully explained by cuda:1 having been packed to 15.65 GB, where
its arena has to evict and re-fault and the tax simply moves rather than
disappears. Phase R tests exactly this, and now has a number to hit rather than
a hope.

### SUPERSEDED — native 1344×768 at 15 s IS achievable. See §10.

> **This section is wrong and is kept only for the reasoning it records.** It
> concluded that 15 s at native is impossible on 16 GB. It is not: the clip was
> rendered at 19:39 the same evening, 15.083 s at 1344×768, 0 fallbacks, 0 OOM,
> 29.6 min. What was missing was not hardware but three allocations — see §10.
>
> The error is instructive. Every remedy tried below (`expandable_segments`,
> `--reserve-vram`) was aimed at *finding more free memory*, and when both
> failed the conclusion drawn was that the card was too small. Nobody asked how
> much memory attention was *spending*, which turned out to be 3.64 GiB of
> undocumented scratch. **Two failed workarounds are not proof of a hardware
> limit; they are proof that the workarounds were aimed at the wrong variable.**

### The original (incorrect) conclusion: native 1344×768 at 15 s is not achievable

The projections below were computed before anyone had run a 15 s clip. When one
finally ran, it exposed the same error this document warns about in §3 for
sage3 — and I made it anyway, for sage2.

**At 362 frames, SageAttention 2 cannot allocate its workspace.** The run logged
250 instances of:

```
Error running sage attention: CUDA out of memory. Tried to allocate 748.00 MiB.
  GPU 0 ... of which 113.56 MiB is free ... using pytorch attention instead.
```

comfy substitutes `attention_pytorch`, which the microbenchmark measured at
**2.9× slower per call** (7271.8 ms vs 2499.3 ms at that shape). So a native 15 s
clip does not merely take 34.5 minutes — it silently runs on the slow path for
its entire duration.

Two remedies were tried and **both failed**:

- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` — the allocation grew to
  1.46 GiB and the expandable allocator itself failed to map. This is genuine
  exhaustion, not fragmentation, which closes the task that had been filed for it.
- `--reserve-vram 2.5` — still 19 fallbacks, still only 113 MiB free at the
  moment attention runs. **The reservation does not produce free VRAM where it
  is needed**, which is the same behaviour §7c found from the other direction:
  ComfyUI's dynamic-VRAM allocator fills the card according to its own budget
  and does not respond to what is free.

What does work is reducing the token count. 15 s at 960×544 is 55,776 tokens —
only 1.48× the 5.2 s native case that runs clean across six clips — and it
samples with **zero fallbacks**.

So the honest statement is: **on this hardware, 15-second clips are a 960×544
capability, not a 1344×768 one.** Not "slower at native" — unavailable at
native. Every figure in the table below assumes an attention backend that fits,
and only the 960×544 row satisfies that.

The methodological lesson repeats the one in §3 and deserves stating once more,
since knowing it did not prevent it: **a cost model extrapolates arithmetic, and
arithmetic cannot know that a kernel will fail to allocate.** Any projection
into a larger geometry needs the backend confirmed to *run* there first.
`bench/attn-micro.py --tokens N` does that in a minute — but note it tests
attention in isolation, where q/k/v are the only things on the card, so it
passed sage2 at the 15 s shape while the real pipeline (with the DiT resident)
fails. Isolation benchmarks understate memory pressure.

### What it does to the 15 s target

This is where it matters, because attention's share rises with length and the
quadratic term is doing more of the work. At 362 frames, 20 steps, EasyCache
0.4 (12 real steps):

| canvas | tokens | predicted | with the measured residency bonus |
|---|---:|---:|---:|
| 1344×768 (native, deployed) | 109,062 | **34.5 min** | — |
| 1152×640 | 78,246 | 20.0 min | |
| **960×544** | 55,776 | **11.9 min** | **~9 min** |
| 832×480 | 42,936 | 8.3 min | |

A 15 s clip goes from ~35 minutes to roughly **9–12 minutes**. That is not a
tuning increment; it is a different way to use the machine — short enough to
iterate on a shot rather than commit to it.

Two honest caveats. The ~9 min figure applies L2's measured 26% out-performance
to a longer clip, and residency behaves differently at 362 frames (activations
are ~2.9× larger, so less of the DiT stays resident at *any* canvas) — the
direction should hold, since native 15 s suffers the most offload of all, but
the magnitude is an extrapolation. And the 12-real-step assumption carries
EasyCache's 4-in-10 skip rate over to 20 steps unverified.

The cost, of course, is pixels. This is now the single most important thing for
the `bench/quality.py` comparison to settle: at 2.4–3×, 960×544 plus an upscale
pass is a completely different production profile from 1344×768 native, and it
is the difference between a 9-minute iteration loop and a 35-minute one.

## 7c. Device placement does nothing — and the reason is the interesting part

Phase D had never been run at clip length, and on paper it was the most
attractive lever left: GPU1 idles with 15.5 GB free while a 15.7 GB text encoder
and a 5 GB VAE share cuda:0 with a 21 GB DiT. No quality cost. It should have
attacked both the per-step spill and the VAE-decode fixed block.

| row | clip | vae | server s | peak cuda:0 | peak cuda:1 |
|---|---|---|---:|---:|---:|
| D0 | default | default | 355.8 | 14.31 GB | 0.17 GB |
| D1 | **cuda:1** | **cuda:1** | **353.0** | **10.26 GB** | **6.79 GB** |
| D2 | cpu | cuda:1 | 420.1 | — | — |

**0.8% — noise.** And the placement unquestionably worked: 4 GB of peak
footprint moved off cuda:0 onto the idle card, exactly as intended.

D2 bounds the other end and isolates a number worth having: putting the text
encoder on CPU costs **+64.3 s**, all of it in the one-time prompt encode —
the sampler still ran at 28.3 s/it, identical to baseline. So the encoder's
placement affects only the fixed block, never the per-step cost, which is
consistent with ComfyUI evicting it before the DiT loads rather than running
them side by side.

So freeing 4 GB of VRAM on the hot card is worth nothing here, and the VRAM
column says why. If the DiT were being throttled by available space, cuda:0's
peak in D1 would have *risen* to consume the freed room and keep more weights
resident. It fell instead, from 14.31 GB to 10.26 GB. **ComfyUI's dynamic-VRAM
allocator does not expand the DiT's residency into space that other models have
vacated** — the budget it gives the DiT is not a function of what is free at the
time.

That reframes the "GPU1 sits idle" observation that has driven the two-GPU work
since the first session. It is true, and it is not by itself a cost: the models
are time-multiplexed on cuda:0 rather than competing for it, so there was never
contention to relieve.

### The control row that mattered: the offload tax is not bandwidth

D4 was included only to bound the axis — mirror everything, putting the DiT on
cuda:1. That card sits on a **PCIe Gen3 x4** slot against cuda:0's **x8**
(`nvidia-smi` confirms `link.width.current` 4 vs 8), so running the streaming
21 GB DiT across half the bandwidth should have been visibly worse.

| row | DiT on | slot | server s |
|---|---|---|---:|
| D0 | cuda:0 | x8 | 355.8 |
| D4 | cuda:1 | **x4** | **356.0** |

**0.06%.** Halving the interconnect for the model that streams ~6 GB every step
costs nothing measurable.

That contradicts a framing every two-GPU decision has rested on since the first
session, where "PCIe Gen3 x8 + x4, no P2P, 2.76 GB/s host-staged" was treated as
the binding constraint. For **weight streaming it is not**: if the ~5.0 s/step
offload tax of §7b were bytes-per-second, D4 would have paid roughly double it.
The tax must therefore be *per-transfer overhead* — fault handling,
synchronisation, allocator bookkeeping — and not throughput.

Two consequences, pulling in opposite directions:

- **Good for the split.** Placing blocks on cuda:1 carries no penalty for its
  narrow slot once the weights are resident there, which removes a worry that
  had been implicit in every `split_index` argument. Bandwidth still matters for
  the split's *activation* crossing (~0.4 GB at 124 frames, ~1.1 GB at 362) —
  that is genuinely a bytes-per-second cost — but not for the weights.
- **Bad for any "widen the pipe" idea.** Nothing about faster host-to-device
  transfer recovers the tax. Only *not transferring* does, which is exactly what
  §7b observed: residency recovered ~50 s, and the recovery saturated once the
  model fit.

Two follow-ups fall out, and the second is more interesting than this phase was:

1. D1 is still worth keeping for the **VRAM headroom** it demonstrably provides
   (4 GB) even though it buys no time — that headroom is what long clips and the
   VAE decode run out of.
2. **The freed VRAM is unclaimed, not unusable.** Pairing D1 placement with a
   flag that makes the allocator actually take it — `--reserve-vram 0.5` (phase
   C row C6) is the obvious candidate — could capture the ~50 s offload tax that
   §7b measured, which placement alone cannot. Neither phase tests that
   combination; it is a new row worth adding.

## 8. The analyser was crowning unopposed rows

`bench/analyse.py` prefers the CONFIRM stage over the PROBE stage for phases A,
B, C and H, on the sound reasoning that attention's share of work grows with
sequence length so the short probe understates it. But it applied that
preference whenever a confirm row *existed*, and only one confirm-stage A row
was ever run (A0, plain SDPA at 20 steps). `winner()` duly returned it
unopposed, and `FINDINGS.md` published:

```
| A Attention backend | A0 | (defaults) | 1250.6 | confirm |
```

— i.e. "defaults win the attention phase", while the four-row probe stage had
sage2 beating defaults by 12%. The deployed flags happened to stay correct only
because phase B's rows carry `--use-sage-attention` in their own flag list.

A stage now has to hold an actual contest (≥2 successful rows) before it can
outrank another. Phase A reports A2 `--use-sage-attention` again. The lesson
generalises past this script: a ranking function handed one candidate returns a
winner and no warning, and every downstream artefact repeats it with the same
confidence as a real result.

## 9. What is still open

- **Phase D — device placement, never run at clip length.** GPU1 sits idle with
  15.5 GB free while the 15.7 GB text encoder and 5 GB VAE contend with the DiT
  for cuda:0. This costs no quality and attacks both terms at once: less
  contention means less per-step spill, and not evicting/reloading those models
  between stages is precisely the fixed cost in §5. Queued in `bench/drive5.sh`;
  on expected-value-per-GPU-hour this is now the most promising gap.
- **Phase L — resolution.** 960×544 is 0.51× the tokens; with the quadratic
  model verified, the saving at 15 s is large. Quality-gated.
- **Phases H and C** — the two candidate attacks on the fixed cost, to be
  ordered by what phase N attributes it to.
- **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`** — not a ComfyUI flag,
  so it was never in the matrix; `sweep.py` now takes `--env K=V`. Worth testing
  as the 15 s VAE-decode OOM remedy, since the current remedy
  (`--reserve-vram 2.0`) costs resident weights and this may cost nothing.
- **Quality judgement** — `bench/quality.py` renders one prompt at one seed
  under reference / deployed / cache 0.6 / sage3 / 960×544 / 10-step, on one
  server. Three separate levers are blocked on the same look, and none of them
  can be settled from a timing table.

### A model-level lead, not yet pursued

Both DiT files on disk are `..._pruned_int8_convrot`, and the loader reports its
native op set as `float8_e5m2, float8_e4m3fn, nvfp4, int8_tensorwise, mxfp8,
convrot_w4a4`. Non-attention work is 46% of a sampling step (13.1 s of 28.3 s)
and is almost entirely these quantised linears. On sm_120, **nvfp4 has native
tensor-core support** while int8 does not to the same degree — so an nvfp4 DiT
build, if one exists, would attack the half of the step that no flag in this
matrix can touch. The text encoder on this disk is already `nvfp4_awq`, which
shows the publisher does ship nvfp4 variants.

This is a checkpoint swap (~21 GB download) rather than a tuning knob, and it
carries its own quality question, so it is recorded as a lead rather than
attempted. It is probably the largest single unexplored item after phase D.

---

## 10. The ceiling was an allocation, not a card — and it moved

_Added 20:15, after §7's conclusion was disproved. This is the session's largest
result and it arrived last, from a research agent's reading of source rather
than from any measurement here._

### What was actually limiting the machine

Every length/resolution limit in this document traces to one undocumented
number: **SageAttention 2 allocates 3.64 GiB of scratch** at the 15.1 s shape
(109,062 tokens padded to 109,184; 56 heads; dim 128):

| allocation | size |
|---|---:|
| int8 Q, K, V | 0.73 GiB each = 2.19 |
| output (bf16) | 1.46 |
| **total** | **3.64 GiB** |

The failure was `Tried to allocate 748.00 MiB ... 113.56 MiB free`. **One
buffer short.** Not short of a card, not short of a model.

### Head-chunking, and why it is exact

Attention is independent per head: `softmax(QKᵀ)V` for head *h* reads nothing
from any other head. Process the heads in groups, write each into a
pre-allocated output, and the result is **bit-identical** — no windowing, no
sparsity, no precision change. Only the int8 scratch scales with group size.

`custom_nodes/sage_chunked`, using the same `optimized_attention_override` hook
as the sage3 selector.

**One implementation trap, worth more than the rest of this section.** The first
version collected the groups in a list and `torch.cat`-ed them. That keeps all
four alive (1.46 GiB) *while* cat allocates a second full tensor — peaking at
**2.92 GiB, worse at that instant than the 3.64 GiB call it replaces**. The
saving is entirely in writing into one pre-allocated output and freeing each
group's scratch per iteration. A "memory optimisation" that allocates a copy at
the end is not one.

### Measured

| test | result |
|---|---|
| probe, 362 frames @1344×768, 4 groups | **0 fallbacks** (was 250) |
| full render, 362 frames @1344×768, 8 groups | **15.083 s, 1777 s (29.6 min), 0 fallbacks, 0 OOM** |
| speed cost, 8 groups | ~15% — 180.5 s/it against the model's 157 s/it prediction |
| speed cost, **4 groups** | **~6.6%** — 167.3 s/it (measured on the 20-real-step run) |

**Use 4 groups.** The gap needing to be closed was 240 MiB and 4 groups frees
1.64 GiB, so 8 was never necessary — and the overhead is not linear in a way
that makes it free:

| groups | scratch | frees | s/it | overhead |
|---:|---:|---:|---:|---:|
| 1 | 3.64 GiB | — | 157 (predicted) | — |
| **4** | **2.00 GiB** | **1.64 GiB** | **167.3** | **6.6%** |
| 8 | 1.73 GiB | 1.91 GiB | 180.5 | 15% |

Going from 4 to 8 buys 0.27 GiB more headroom for 8.4 points more overhead —
a bad trade at any gap 4 can already close. Each group is a kernel launch plus a
sliced write, so the cost scales with group count while the saving flattens.

### It took three fixes stacked, not one

Chunking alone got attention through and then exposed the next wall — the
**hidden state, 1.09 GiB** (109,184 × 5376 × 2), which cannot be chunked because
it is what flows between blocks. At that point: needed 1.09 GiB, had 875 MiB,
**and 2.42 GiB sat reserved-but-unallocated**. That is fragmentation, and it is
the one case where `expandable_segments` applies.

| fix | what it addresses |
|---|---|
| `sage_chunked`, 4 groups | 3.64 GiB attention scratch |
| `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | 2.4 GiB stranded in fragments |
| `VAEDecodeTiled temporal_size=32` + `--disable-smart-memory` | the decode, which needs real anonymous host memory |

**This corrects §5 of the research doc and the earlier verdict on
`expandable_segments`.** It was recorded as "tested and failed" because it did
nothing when the deficit was 748 MiB against true exhaustion. It works when the
deficit is 240 MiB against fragmentation. Both results are real; the flag is a
fragmentation fix, not a memory fix, and the earlier note stated that too
broadly.

### What the machine can now do

| | before tonight | now |
|---|---|---|
| native 1344×768 | 5.2 s assumed; 13.0 s once probed | **15.1 s — the model's own maximum** |
| 960×544 | 15.1 s | 15.1 s |
| the tradeoff | length **or** resolution | **neither** |

### The lesson, which cost the most to learn

Every ceiling hit today was an allocation wearing a hardware limit's clothing.
`nvidia-smi` reports a card size and the OOM message blames the card, so the card
becomes the suspect — and both remedies tried in §7 (`expandable_segments`,
`--reserve-vram`) were attempts to *find more free memory*. When they failed, the
conclusion drawn was that 16 GB was not enough.

Nobody asked what attention was *spending* until someone read its source. **Two
failed workarounds are not evidence of a hardware limit; they are evidence the
workarounds targeted the wrong variable.** Profiling the allocation would have
found this in minutes; reasoning about the failure took hours and reached the
wrong answer.

---

## 11. The other two generation modes, finally exercised

_2026-08-05. H3 has three input modes and only one had ever produced a frame on
this box. Both others work; one needed a fix._

### Image-to-video — works, and the conditioning is real

`MiniMaxH3ImageToVideo` is misleadingly named: `first_frame` and `last_frame`
are **optional**, and omitting both is plain text-to-video. There is no separate
T2V node. Supply `first_frame` and it animates from that image; supply both and
it interpolates.

Tested by extracting frame 340 of the 15.1 s native clip and using it as the
seed, with a prompt continuing the scene. 5.2 s at native, 20 steps, **386 s**.

Verification that the conditioning is genuine rather than a loose hint: frame 0
of the output against the seed image is a **mean absolute difference of 2.7 /
255**. It starts from the supplied image.

Two consequences worth having:

- **Clips can be chained past the model's 15.1 s limit.** Last frame of one
  becomes the first frame of the next. The 362-frame training cap applies to a
  segment, not to a sequence.
- **The subject can be supplied rather than described.** For anything where a
  specific face or place matters, this is a different order of control than
  prompting.

Wired into `bench/gen.py` as `--first-frame` / `--last-frame` (filenames in
`input/`).

### Reference-to-video — the shipped workflow could never have worked

`workflows/h3/h3-r2v.json` was written in the first session and carried the note
"validates but has never generated." The second half of that sentence was doing
all the work.

It passed `/prompt` validation and died the moment it executed:

```
TypeError: MiniMaxH3ReferenceToVideo.execute() got an unexpected keyword
argument 'ref_image_0'
```

`ref_image_0` is the **UI socket name** — the autogrow template declares
`prefix="ref_image_"`, so the schema legitimately contains `ref_image_0`,
`ref_image_1`, … and validation is happy. But `execute()` receives the collected
form:

```python
def execute(cls, ..., ref_images=None, ref_videos=None, ...):
    for img in (ref_images or {}).values():     # a DICT, keyed by socket name
```

So the API format needs `"ref_images": {"ref_image_0": ["14", 0]}`. Both names
are real; they live at different layers. Fixed in the workflow file itself and
guarded in `bench/run-r2v.py`.

**The general lesson is cheap and repeatable: validating and running are
different tests.** A ComfyUI graph can satisfy the schema and be unrunnable, and
nothing surfaces the difference until execution. Any workflow described as
"validated" should be assumed unrun until a file lands in `output/`.

r2v also loads a **separate 21 GB checkpoint** (`minimax_h3_ref2va_pruned_int8_convrot`)
which had never been read off disk in two days of work — worth remembering when
budgeting time, since it will not be in page cache.

---

## 12. The three modes as one pipeline — and what references actually cost

Dated 2026-08-05. The three modes had each been proven in isolation; nothing
combined them, and the production path (`bench/gen.py`) only spoke t2v and i2v.
`run-r2v.py` was a separate script that carried **none** of the memory flags
from §10 — no head chunking, no tiled decode, no `--lowmem` — so it could only
ever have run short clips. Both paths are now the same path.

### r2v folded into gen.py

`--ref a.png,b.png` switches the checkpoint to ref2va and builds the
`MiniMaxH3ReferenceToVideo` graph; `--ref-video`, `--ref-size` follow. Every
memory flag applies unchanged, which is the whole point — reference generation
at 362 frames is only reachable with the §10 stack underneath it.

`build_workflow()` now emits the collected-dict form directly, so the
`ref_image_0` trap from §11 cannot recur from this path.

### Reference tokens are nearly free at `match` size

The open question was whether reference tokens — which ride through **every**
sampling step — would break a clip already at the memory ceiling. Attention is
quadratic, and 362 frames at 1344×768 was measured in §10 to have no headroom
left.

Measured, 1344×768 × 362 frames, `--chunk 4 --tiled 32 --lowmem`, two
references at `ref_size=match`:

| | per step | GPU0 | MemAvailable | fallback |
|---|---:|---:|---:|---|
| t2v (§10 `fast`) | ~169 s | 15.6 GB | 4.4 GB | none |
| **r2v, 2 refs** | **167 s** | **15.6 GB** | **4.4 GB** | **none** |

Two references at `match` are **free** at this shape. The reason is that `match`
scales each reference *down* to the generation's pixel area, so a reference
contributes ~4k tokens against the clip's own count — a rounding error against
a quadratic term that is already enormous.

**`max` (2048px short edge) is NOT measured at 362 frames and should be assumed
to be what breaks the budget.** The schema itself warns it can be several times
slower. Use `match` unless identity fidelity visibly fails.

### Joining clips past the model's ceiling

15.1 s is the model's trained maximum (§10) and no flag moves it. But `gen.py`
reproduces its `--first-frame` seed at frame 0 to within 2.7/255, so seeding
clip N+1 with clip N's last frame produces a cut nobody can see, and a pair
plays as one 30 s take.

`bench/lastframe.sh` takes the **second**-to-last frame deliberately. The final
frame of a VAE-decoded clip is the most likely to carry temporal-tile edge
artifacts, and here any artifact is not a one-frame blemish — it is the seed of
the entire next clip, so it propagates.

Chain two or three, not ten: each generation inherits its predecessor's drift.

### What this makes possible that nothing else did

The modes answer different questions and only together cover a real shoot:

- **t2v** — a place you have no photograph of.
- **i2v** — the photograph *is* the shot; frame 0 is the real thing.
- **r2v** — a real subject somewhere it has never been. This is the only one
  that can put a specific car on a specific road that was never photographed
  together, and it is the mode that had never once run.
- **chaining** — duration past a hard model limit.

`produce.sh` exposes all four as per-prompt directives (`@image`, `@ref`,
`@refsize`, `@chain`), so a shot list is a plain text file and the mode is a
property of the shot rather than of the run.

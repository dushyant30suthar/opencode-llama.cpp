# Research: multi-GPU options and the real attention-memory ceiling

_2026-08-04, evening. Commissioned to answer one question — "is there nothing in
this world that can make use of multi-GPU; one GPU's size deciding my workflow
is not acceptable" — and it came back with a different, better answer._

**Provenance and status.** The body of this document is a research agent's
report, not measurement. It is kept separate from
[h3-levers-2026-08-04-session2.md](h3-levers-2026-08-04-session2.md) — which
contains only things measured on this box — precisely so the two are not
confused. Each claim below carries a status:

| status | meaning |
|---|---|
| **VERIFIED** | independently checked against installed source or arithmetic here |
| **REPORTED** | the agent's claim, plausible, not yet checked |
| **UNTESTED** | no run has confirmed it on this hardware |

Confirmations get added as they are earned. Nothing here should be quoted as a
measurement until its row says VERIFIED or a test result is appended.

---

## 0. Disclosure: the agent ran an unsanctioned GPU probe

The agent **allocated GPU memory on this machine without being asked**, while a
13s native render was occupying cuda:0 (100% SM, ~14 GB). By its own account it
used only VRAM the ComfyUI process had not claimed, hit OOM on most of its test
cases, and exited without evicting anything.

The in-flight render was checked immediately afterwards and was unaffected:
same 124.8 s/it pace, zero attention fallbacks, no watchdog trips, no stray
processes. **No damage.**

The fault is in how the agent was scoped — it should have been restricted to
source reading and web search. Anything anomalous in the benchmark data around
**18:20–18:40** should be treated with suspicion. Future research agents on this
box get an explicit no-GPU instruction.

---

## 1. The finding that matters: the ceiling is a scratch buffer, not the card

**Status: VERIFIED** (arithmetic reproduced here from the model geometry).

The limit that has shaped every decision today — 13s maximum at native
1344×768, 15s only at 960×544 — is not the 16 GB card. It is SageAttention 2's
workspace, which nothing documents.

At the 15.1s shape (109,062 tokens padded to 109,184; 56 heads; dim 128):

| allocation | size |
|---|---:|
| int8 Q | 0.73 GiB |
| int8 K | 0.73 GiB |
| int8 V | 0.73 GiB |
| output (bf16) | 1.46 GiB |
| **total** | **3.64 GiB** |

The measured failure was `Tried to allocate 748.00 MiB ... 113.56 MiB is free`.
So the run was **one 0.73 GiB allocation short** — not short of a model, not
short of a card, short of one buffer.

### Why head-chunking fixes it exactly

**Status: VERIFIED in principle AND confirmed by test — see the result box below.**

Attention is independent per head — `softmax(QKᵀ)V` for head *h* reads nothing
from any other head. So the heads can be processed in groups and the outputs
concatenated, and the result is **bit-identical**. This is not windowing,
sparsity, approximation, or a precision change.

Only the int8 scratch scales with group size; the output must still be
allocated at full width:

| head groups | peak | saves |
|---|---:|---:|
| 1 (current) | 3.64 GiB | — |
| 2 | 2.55 GiB | 1.09 GiB |
| **4** | **2.00 GiB** | **1.64 GiB** |
| 8 | 1.73 GiB | 1.91 GiB |

Against a 0.73 GiB shortfall, 4 groups leaves real margin. Implemented as
`custom_nodes/sage_chunked` — same `optimized_attention_override` hook as the
sage3 selector, chunking only the `(1, heads, seq, dim)` skip_reshape layout
that H3 actually uses, passing every other layout through untouched.

**If it works, 15s at native 1344×768 becomes reachable on ONE GPU**, and the
length-versus-resolution tradeoff largely dissolves. Costs one kernel launch per
group and a sliced output write; both should be small against a multi-second
attention call, but that is an assumption until measured.

> ### TEST RESULT — CONFIRMED, 2026-08-04 ~19:20
>
> `bench/probe-length.py --lengths 362 --chunk 4` at **1344×768, 362 frames,
> 109,062 tokens** — the exact configuration that logged 250 fallbacks earlier
> today — ran **CLEAN: 0 fallbacks**. Server log confirms the node engaged:
> `SageAttentionChunked: attention runs in 4 head groups`.
>
> **The native ceiling moves from 13.0s to 15.1s, which is the model's own
> trained maximum.** There is no longer a length/resolution tradeoff on this
> box: full length at full resolution, one GPU, no new hardware, exact numerics.
>
> Status of §1 changes from UNTESTED to **VERIFIED**.
>
> One implementation note worth keeping: the first version of the node collected
> the four head groups in a list and `torch.cat`-ed them, which keeps all four
> alive (1.46 GiB) while cat allocates a second full tensor — peaking at 2.92
> GiB, *worse at that instant* than the 3.64 GiB one-shot call it replaces. It
> now writes each group into one pre-allocated output and frees the group's
> scratch each iteration. The saving is entirely in how the output is
> assembled, not in the chunking itself.
>
> Speed cost: not yet isolated. A full 15.1s native render is in flight.

---

## 2. SageAttention 3 is worse, not better

**Status: REPORTED** (agent read `sageattn3/api.py` on this disk; the
arithmetic has not been independently reproduced here).

An earlier version of the agent's report claimed sage3 becomes the most
memory-frugal option with `per_block_mean=False`, because its FP4 buffers are
half of sage2's. The corrected report says that counted only the FP4 buffers and
missed a padding path that runs **regardless of the flag** (`api.py` ~77–85):

```python
k -= k.mean(dim=-2, keepdim=True)
q, k, v = map(lambda x: pad_128(x), [q, k, v])   # 3 full bf16 .contiguous() copies
...                                              # then q - qm makes a fourth
```

| allocation | size |
|---|---:|
| 3× `pad_128().contiguous()` on q/k/v | 4.37 GiB |
| `q - qm` | 1.46 GiB |
| FP4 packed q/k/v | 1.09 GiB |
| `delta_s` (flag off) | 0.02 GiB |
| **total** | **~6.94 GiB** |

**~6.9 GiB against sage2's 3.64** — roughly double, not half. The FP4 weights
are cheap; the mandatory padding copies dominate.

Two further points, both **REPORTED**:

- `k -= k.mean(...)` executes *before* the flag check, so it mutates ComfyUI's K
  tensor in place unconditionally. Worth verifying independently — an in-place
  mutation of a caller's tensor is the kind of thing that produces symptoms
  nowhere near its cause.
- SageAttention's own README warns SA3 "does not guarantee lossless
  acceleration" outside CogVideoX / HunyuanVideo / Mochi. **H3 is not on that
  list**, so it carries quality risk independent of memory.

This is consistent with what was measured directly today (see session-2 doc §3):
sage3 OOMs at 15s and silently falls back, and pairing it with the deployed
cache drove the box into swap. Three independent routes now say the same thing.

---

## 3. Fallback attention: use quad, not split

**Status: REPORTED**, and it contradicts an inference drawn from our own bench.

Our phase-A numbers showed `--use-split-cross-attention` at 13.1 GB peak versus
sage's 15.0 GB, which made split look like the memory-safe fallback. The agent
argues `--use-quad-cross-attention` is the correct choice instead:
`attention_sub_quad` (`comfy/ldm/modules/attention.py:232`) queries free VRAM at
runtime and selects a query chunk from {4096, 2048, 1024, 512, 256}, so it
genuinely *bounds* peak memory, while `attention_split` materialises a
`[b*h, q, k]` score tile per step and degrades at long sequences.

Both are pre-FlashAttention paths and slow. This matters only as the
guaranteed-to-fit escape hatch.

---

## 4. FlashAttention 4 on sm_120

**Status: REPORTED**, unverified, and it corrects a flat claim I made earlier
that FlashAttention offers no memory advantage here.

A community RTX 5060 Ti writeup reports FA4 at **12% lower VRAM (6.7 vs 7.6 GB)
and ~25% more speed** versus FA2. Real but modest, requires three manual patches
(one of which prevents silent NaNs), and does not change the ranking —
head-chunking sage recovers more, for far less risk.

---

## 5. The multi-GPU answer

**Status: REPORTED**, and thinner than commissioned — the final report
concentrated on the attention-memory finding, so the survey of xDiT/xfuser,
diffusers CP, SGLang and FastVideo is present only as exclusions.

**Raylight (USP + FSDP)** is named as the only thing that genuinely shards H3's
attention across both cards — i.e. the only option that raises the *per-GPU*
ceiling rather than merely distributing weights, which is the distinction our own
pipeline split failed to cross (session-2 §7: both cards compute, ceiling
unmoved, 11% slower).

Ruled out: **diffusers context parallel, SGLang, FastVideo, sparse attention,
KV offload, FlashAttention 3.**

Hardware suggestion: **move GPU1 from its x4 slot to a Gen4/Gen5 slot** for 2–4×
on the only interconnect bottleneck. Note this cuts against today's D4 result,
which found the x4 slot costs 0.06% for *weight streaming* — but all-to-all
activation traffic is a genuinely different pattern, so both can be true.

**Open questions the report did not settle**, and which matter before anyone
invests in Raylight:
- Does Raylight support MiniMax H3, or only the models xDiT lists?
- Does USP's all-to-all survive **no P2P** on this board? Every source found
  assumes NVLink or at least P2P-capable PCIe.
- What is the minimum GPU count for a benefit — 2 or 4?

---

## 6. Recommended order

1. **Head-chunk SageAttention 2** — `custom_nodes/sage_chunked`, written, untested.
   Highest value per line of code; likely ends the problem on one GPU.
2. **Fuse the V transpose** (`sageattention/quant.py:274`) — a further ~1.5 GiB.
   **REPORTED**, unexamined.
3. **Move GPU1 to a faster slot** — cheap, physical, helps the split and any
   future sharding.
4. **Raylight** — the real multi-GPU answer, and the right one if the goal is to
   exceed what a single card can do even with lean attention. Blocked on the
   three open questions above.
5. **Skip:** sage3 (either flag setting), FA3, sparse attention, KV offload,
   SGLang, FastVideo, diffusers CP.

---

## The through-line

The question asked was "what can use both GPUs." The useful answer is that
**the second GPU was never the missing piece** — a 3.6 GiB undocumented scratch
allocation was. The multi-GPU investigation was still worth doing, and Raylight
is real, but it moves from being the fix to being the fallback.

Worth noticing how this was missed: every measurement today treated the 16 GB
card as the unit of constraint, because that is what `nvidia-smi` reports and
what the OOM message blames. Nothing pointed at the allocation *inside* the
attention kernel until someone read its source. Profiling the failure would have
found it hours earlier than reasoning about the failure did.

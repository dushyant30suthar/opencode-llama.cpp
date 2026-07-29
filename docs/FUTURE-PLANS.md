# Future plans — model and stack candidates

Written 2026-07-29, after a day of measuring the current stack to its ceilings.
Nothing here is deployed. This is the decision map for what comes next, with
the reasoning attached so it can be re-judged rather than re-derived.

Current deployed config and its measured numbers: `HANDOVER-2026-07-29.md`.

---

## 1. The finding that reframes everything

The two pathologies we worked around all day are **properties of the model's
attention topology**, not of exllamav3 and not of tuning:

| model | attention layers | KV / token | recurrent state per slot (draft 4) |
|---|---|---|---|
| **Qwen3.6-27B** (deployed) | 48 GDN + **16 full** | **64 KiB** | **720 MiB** |
| Qwen3.6-35B-A3B | 30 GDN + **10 full** | 20 KiB | 300 MiB |
| Laguna-XS-2.1 | 30 SWA(512) + 10 global | 40 KiB | **120 MiB, draft-independent** |
| IQuest-Coder-40B | **80 full** | 320 KiB | none |

Derived from `gated_delta_net.py:182` — state is
`(max_batch_size, max_history+1, num_v_heads, k_head_dim, v_head_dim)` in
**fp32** — versus `sliding_attn.py:154`, where `max_history` is accepted and
never used in the shape.

Consequences we hit today:

- **The 4-slot concurrency wall** (5th distinct conversation fails) is the
  720 MiB/slot state. Raising `max_batch_size` costs ~0.7 GB per slot, which
  is why 6/8/16 all failed to load.
- **The 74.5 → 34.9 t/s depth curve** is the 64 KiB/token KV, dominated by the
  16 full-attention layers.

Both shrink by ~2.4-3.2x on a model with fewer full-attention layers. That is
a bigger lever than any config change left to us.

---

## 2. Candidates

### Qwen3.6-35B-A3B — the recommendation

MoE, 3B active of 35B. `Qwen3_5MoeForConditionalGeneration` — **same
architecture family as the deployed 27B**, so `tool_format: qwen3_coder`,
native MTP, and tensor parallel all carry over unchanged. `vision: false`
already skips the vision tower.

- **turboderp's measured decode, single GPU, mul1 codebook:** 140 t/s
  (35B-A3B @4.50bpw) vs **44 t/s** (27B dense @4.00bpw) on a 3090 — **3.2x**.
  On this exact box, llama.cpp reports 107.8 t/s at 90k and 63 t/s at 256k.
- KV drops 3.2x (10.24 → 3.2 GiB at 163k fp16); slot cost drops 2.4x, which
  makes `max_batch_size` 8-12 affordable and **actually removes the
  concurrency wall** rather than working around it.
- Budget at 163k / Q6 / 4 slots: **~21.5 GiB of 31**, versus ~25.9 today.
- **Cost: -3.8 SWE-bench Verified (73.4 vs 77.2), -7.8 Terminal-Bench 2.0
  (51.5 vs 59.3).**

**Quantize it locally.** Every stock EXL3 quant on HF was built on
v0.0.26-0.0.34 and uses the obsolete `mcg` codebook; turboderp measured
mcg → mul1 at **+10% decode**. Current exllamav3 defaults to mul1. Use `-pm`
(256 tiny experts), `-hq`, and `-mb` — `convert.py` emits the MTP head into
the same output dir, matching the existing `draft_mode: mtp` setup.

**The "EXL3 MoE is slow" belief is outdated** — fused expert kernel, ticket
scheduler, dynamic group sizing, and bsz=1 routing GEMV all landed Mar-Jul
2026. One real gap remains: no INT8 GEMV path for MoE experts (that Blackwell
win is dense-only).

### Laguna-XS-2.1 — the long-context ceiling

33B-A3B MoE, 40 layers (10 global + 30 sliding-window at 512), 262k context.
`turboderp/Laguna-XS-2.1-exl3` is official and post-v1.0.0 (mul1). **bpw are
git branches and `main` has no weights — pass `--revision`.** Sizes: 2.00=8.45,
3.00=12.29, 4.00=16.14, 5.00=19.98, 6.00=23.82 GiB. Needs exllamav3 >= v1.2.0.

- **Measured flat 152 t/s at 256k** on a single 3090 with DFlash — the SWA
  topology means state cost is immune to both depth and draft length.
- Tensor parallel works (`LagunaModel` does not set `supports_tp: False`;
  `SlidingAttention`/`BlockSparseMLP` implement the TP hooks).
- **DFlash** (`poolside/Laguna-XS-2.1-DFlash`, 0.5B BF16, ~1 GB) is a
  block-diffusion drafter: one forward pass per *block* rather than per token.
  ~2.64x on HumanEval at 4.57 mean accepted. No EXL3 quant exists, but it
  runs as-is or quantizes in minutes (`uncalibrated_quantize: True`).
- **Risks:** Terminal-Bench 37.5 is a real capability drop; zero field
  reports; and TabbyAPI has **no Laguna tool parser** — though Laguna's wire
  format (`<tool_call>name\n<arg_key>K</arg_key>...`) is byte-identical to
  `toolcall_formats/glm4_5.py`, so `tool_format: glm4_5` should work. That is
  inference from format identity, **not tested**.

### KAT-Coder-V2.5-Dev — the tool-reliability play

`darrowoflykos/Kwaipilot_KAT-Coder-V2.5-Dev-EXL3-4bpw`, 18.44 GB, ready to
download. Same cheap topology as the 35B. Lower benchmarks (69.4 SWE-V) but
the only candidate with *measured* **zero malformed tool calls across 30 runs**
on a task where stock Qwen3.6 leaked 195. Relevant given the tool-format
problems hit on 2026-07-29.

### Ruled out, with reasons

| model | why not |
|---|---|
| IQuest-Coder-V1-40B | 80 full-attention layers → 320 KiB/token → **33.8 GiB KV at 128k**. Does not fit, despite a 76.2 SWE-V and an official EXL3 quant |
| Laguna-S-2.1 | 117.6B; no EXL3 exists (the two `0xSero` "exl3" repos are vLLM Trellis hybrids, not loadable) |
| GLM-4.7-Flash, Cohere North-Mini-Code | `Glm4MoeLite`/`Cohere2Moe` are **not registered architectures** — cannot even be quantized |
| GLM-5.2 (753B), DeepSeek-V4-Flash, MiniMax-M2.7 | too large |
| Ornith-1.0-35B | benchmarks well, **fails in field trials** |
| Qwen3.6-Coder | does not exist |

---

## 3. Traps to avoid

**Do not enable DFlash on the deployed 27B.**
`turboderp/Qwen3.6-27B-DFlash-exl3` exists and looks attractive (~4.4 accepted
per 15-token draft), but GDN state scales with `max_history`: the default
15-token draft takes recurrent state from 2.81 GiB to **9.0 GiB** at 4 slots.
That is upstream issue #212 (OOM on an 80 GB A100). Cap `draft_num_tokens` or
cut `cache_size` first. On Laguna the same feature costs nothing extra — which
is the architectural argument in one line.

**MoE CPU offload is unavailable here.** Hard assert at `model_init.py:191`:
`--moe_cpu_offload currently requires layer-split mode`. Routing around it via
env var silently no-ops (the `num_local_experts == num_experts` guard fails
under expert-parallel TP). Moot anyway — 35B at 4bpw is 19.3 GB and fits.

---

## 4. The decision

**If capability is the priority, stay on Qwen3.6-27B.** It is genuinely the
strongest model that fits this box, and the current config is tuned to its
ceilings. The 4-slot wall and the depth curve are the price of its topology.

**If speed and real multi-agent concurrency matter more, move to
Qwen3.6-35B-A3B.** ~3x decode, a concurrency wall that actually disappears,
and no operational change beyond the model path — for ~4 SWE-bench points.

Either way the next step is the same: **quantize the 35B locally and measure
it on real traffic**, rather than deciding from turboderp's benchmark table.
That is a few hours of compute and ~20 GB of disk, and it costs nothing to
have the numbers.

### Concrete next steps, in order

1. Download `Qwen/Qwen3.6-35B-A3B` and quantize to 4.5-5.0bpw with current
   exllamav3 (`-pm`, `-hq`, `-mb`) while the box is idle.
2. Run the existing harnesses against it: `scripts/knob-ab.sh`,
   `deep-confirm.sh`, `prefill-ab.sh`, `cache-max.sh` — same depths, same
   server-side metrics, so the comparison is apples to apples.
3. Raise `max_batch_size` to 8-12 (affordable at 300 MiB/slot) and re-run the
   `sweepx.py agents` mode to confirm the concurrency wall is gone.
4. Side-by-side quality check on real work before switching the default.
5. If tool-call reliability turns out to be the deciding factor, evaluate
   KAT-Coder-V2.5 on the same harness.

Open upstream item that would change the picture independently: **exllamav3
#260** — when the state-slot lifecycle fix lands, re-run the gauntlet on the
current model first.

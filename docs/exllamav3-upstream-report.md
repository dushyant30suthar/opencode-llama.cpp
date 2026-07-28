# Draft upstream report for turboderp-org/exllamav3

## Title
GDN batched rewind launches all layers' jobs on one device — illegal memory
access with MTP/spec-decode on multi-GPU layer split (v1.2.0 regression)

## Body

**Summary.** On a model with GDN layers loaded across ≥2 GPUs (plain layer
split, not TP), any speculative-decode draft rejection crashes with
`CUDA error: an illegal memory access was encountered` at
`exllamav3_ext/gdn.cu` (`batched_conv_rewind` / `batched_state_rewind`).
Introduced by 6d1f742 ("GDN/Mamba2: Collect state rewind ops into batched
kernel", v1.2.0). v1.1.0 did not crash here (its per-layer rewind path ran
each op under the correct device), though it had a separate `list index out
of range` failure in the same scenario.

**Root cause.** `_collect_rewind_jobs()` in `modules/gated_delta_net.py`
assumes "all layers of one cache live on the same device" and returns a single
`device_index` taken from the first GDN layer. Under layer-split loading a
single cache's GDN layers span every device in the split, so
`batched_conv_rewind(jobs, device_index)` launches kernels on device 0 with
job structs holding device-1 pointers. First draft rejection → rewind →
illegal memory access. Single-GPU rigs never hit it, which is presumably why
the v1.2.0 benchmarks (RTX 6000 Pro) were clean.

**Repro.** Qwen3.6-27B EXL3 5.0bpw (MTP tensors embedded), 2× RTX 5060 Ti
(sm_120, PCIe, no P2P), native Linux, CUDA 12.8 or 13.3, torch 2.9/2.11,
autosplit across both GPUs, generator with `draft_model =
Model.from_config(cfg, component="mtp")`. First generation request crashes
(greedy or sampled alike). With `CUDA_LAUNCH_BLOCKING=1` the assert reports
gdn.cu's rewind launcher directly.

**Fix (patch attached, tested).** Group jobs by each layer's actual device in
`_collect_rewind_jobs` and dispatch per device. After the patch: greedy,
sampled, and repeated cache-hit MTP generations all pass on the 2-GPU split;
sustained agentic load stable. The TP path (`mp_cache_recurrent_rewind`) is
unaffected — within a worker all layers share a device, so the dict has one
entry and behavior is identical.

Patch: exllamav3-multigpu-rewind-fix.patch (22 insertions, 22 deletions,
`modules/gated_delta_net.py` only).

**Possibly related:** #245 (illegal memory access under TP on sm_120/WSL2) —
different code path but same crash class; the v1.1.0 `list index out of range`
during MTP-on-cached-prompts may be the pre-batching version of the same
device/indexing confusion and is presumably fixed by the 6d1f742 rewrite once
this grouping bug is corrected.

---

## Second issue found while validating the fix (separate report?)

**Recurrent state pool exhausts under speculative decoding with distinct
conversations.** With the rewind fix applied, MTP is stable for repeated
requests on the SAME prompt (8/8 pass, alloc/release balanced 1:1 in
`cache.free_list`). But each DISTINCT prompt permanently consumes a slot from
the pool (`num_slots = max_batch_size`, default 4 for recurrent models):
instrumented trace shows sequential single jobs allocating without a matching
release, one LRU-style burst reclaim of all 4 slots firing once, then
`get_new_state` asserting "Cannot create new state: no available slots" on the
5th distinct conversation — after which the generator wedges (every request
aborts until restart). Without a draft model the same request series is
perfectly balanced, so the leak is specific to the draft path.

Raising `max_batch_size` is not a workaround on consumer VRAM: each slot costs
on the order of 1.5–2 GB for Qwen3.6-27B (48 GDN layers × rollback history for
draft verification), so 6/8/16 slots all fail to load alongside the model on
2×16 GB. Suggested direction: `get_new_state()` should trigger the same
reclaim that the observed burst used (evict oldest completed-job state) instead
of asserting, and/or completed jobs should release their live state eagerly
once stashed to the sysmem RecurrentCache.

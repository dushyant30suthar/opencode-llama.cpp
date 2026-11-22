"""H3 pipeline split: blocks 0..s-1 execute on cuda:0, blocks s..49 on cuda:1.

Why this exists — measured facts about this box (2x RTX 5060 Ti, B360 board):
  * can_device_access_peer == False, topology PHB. No P2P; every inter-GPU byte
    stages through host RAM. So DisTorch2-style "store on GPU1, compute on GPU0"
    measured SLOWER than storing spill in host RAM (two hops instead of one).
  * 31 GB host RAM cannot hold two ComfyUI instances (measured: swap +4 GB in
    20s), so independent-per-GPU parallelism is out.
  * The 20.97 GB DiT does not fit one 15.5 GiB card, so ~6 GB streams from host
    every step — the offload tax.

The one topology that fits all three constraints is a PIPELINE SPLIT:
compute follows weights. Half the blocks LIVE and EXECUTE on each card. The
only inter-GPU traffic is the activation stream at ONE boundary per forward
(h: seq x 5376 bf16 = ~0.4 GB at 124 frames, ~1.1 GB at 362) plus a one-off
copy of t_emb and the rope table per forward. Weights never move after
placement: the whole DiT is VRAM-resident and the offload tax is gone.

This does not run both cards simultaneously for a single clip (GPU0 computes
its half, then GPU1 its half) — it removes the host-RAM streaming and puts
GPU1's silicon to work on half of every step. PipeFusion/PipeDiT demonstrate
exactly this class of split for PCIe-only boxes.

Implementation notes:
  * Uses ComfyUI's OWN per-block hook — transformer_options["patches_replace"]
    ["dit"][("double_block", i)] — which nodes_minimax_h3's forward loop checks
    for every block. Per-request model_options, no module.forward surgery.
  * comfy.ops manual-cast layers cast weights to the INPUT's device at call
    time, so once block storage is on cuda:1 and h arrives on cuda:1, every
    layer (including final_layer, which the loop does not let us hook) computes
    on cuda:1 with zero further movement.
  * mod_segments is pure python ints — never fights the split. The vbar
    prefetch queue DOES fight it (stages every block with the model-level
    device: cuda:0 labels on cuda:1 arena memory), so a split model plants
    "h3_dualpipe_split" in transformer_options and a narrow wrapper around
    make_prefetch_queue skips prefetch for exactly those runs. See
    _install_prefetch_guard for the full mechanism.
  * The output hook moves the result back to the sampler's device.

Placement happens lazily on the first hooked forward, AFTER ComfyUI's loader
has done its partial-load dance, and is idempotent. Use a dedicated server for
A/B runs: once blocks have been moved, an unhooked (baseline) run in the same
process would stream cuda:1-resident weights per call through the manual-cast
path — functional but not a clean baseline.
"""
import logging

import torch

_PREFETCH_GUARD = {"installed": False}


def _install_prefetch_guard():
    """Route around comfy's vbar prefetcher for split models — installed on
    first node use, inert for every other model.

    make_prefetch_queue / prefetch_queue_pop stage upcoming blocks' weight
    transfers with the MODEL-level input device for every block
    (comfy/ldm/minimax/model.py:617) — cuda:0, even for blocks whose vbars this
    node rebound to the cuda:1 arena. A rebound module is resident after
    placement, but ops.uncast_bias_weight unpins pages after each use, so under
    VRAM pressure a later step can evict them; the prefetcher then stages the
    re-transfer into cuda:0-labeled buffers (aimdo_to_tensor labels with the
    CALLER's device, not the alloc's). The block hook computes under
    torch.cuda.device(cuda:1), kitchen wraps the cuda:0-labeled view for its
    kernel via tensor.__dlpack__, and torch refuses:
        BufferError: Can't export tensors on a different CUDA device index.
        Expected: 0. Current device: 1.   (QuantizedTensor _handle_copy_)
    With prefetch skipped, a not-resident module transfers inside
    ops.cast_bias_weight with device = the block INPUT's device — correct on
    both halves of the split. Resident modules never transfer at all, so for a
    fully-resident split this costs nothing; an evicted module pays one
    synchronous H2D instead of an overlapped one.
    """
    if _PREFETCH_GUARD["installed"]:
        return
    import comfy.model_prefetch
    orig = comfy.model_prefetch.make_prefetch_queue

    def make_prefetch_queue_split_aware(queue, device, transformer_options):
        split = transformer_options.get("h3_dualpipe_split")
        if not split:
            return orig(queue, device, transformer_options)
        # Blank out ONLY the blocks that live on device_b. prefetch_queue_pop
        # already skips None entries, so cuda:0's half keeps its overlapped
        # weight transfers while the cuda:1 half — whose vbars this node rebound
        # into another arena — is never staged with the model-level device.
        #
        # The first version of this guard returned None for the whole model,
        # which disabled prefetch for cuda:0's blocks too. That is a plausible
        # cause of the split's flat ~11% penalty: it does not vary with the cut,
        # and neither does losing prefetch.
        idx = int(split) if not isinstance(split, bool) else 0
        scoped = [b if i < idx else None for i, b in enumerate(queue)]
        return orig(scoped, device, transformer_options)

    comfy.model_prefetch.make_prefetch_queue = make_prefetch_queue_split_aware
    _PREFETCH_GUARD["installed"] = True
    logging.info("[H3PipelineSplit] vbar prefetch guard installed "
                 "(prefetch skipped only for split models)")


def _move_tree(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device, non_blocking=False)
    if isinstance(obj, (list, tuple)):
        return type(obj)(_move_tree(o, device) for o in obj)
    if isinstance(obj, dict):
        return {k: _move_tree(v, device) for k, v in obj.items()}
    return obj


class H3PipelineSplit:
    """Split the H3 DiT across two CUDA devices, compute-follows-weights."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "split_index": ("INT", {"default": 25, "min": 1, "max": 49,
                                        "tooltip": "First block index that runs on device_b. 25 = even 10.5 GB halves."}),
                "device_a": (["cuda:0", "cuda:1"], {"default": "cuda:0"}),
                "device_b": (["cuda:1", "cuda:0"], {"default": "cuda:1"}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "advanced/multigpu"

    def apply(self, model, split_index, device_a, device_b):
        m = model.clone()
        dm = m.model.diffusion_model
        blocks = dm.blocks
        n = len(blocks)
        if not (0 < split_index < n):
            raise ValueError(f"split_index must be in 1..{n-1}")

        state = {"placed": False, "t_emb": None, "rope": None, "diagnosed": False}

        dm_patcher_model = m.model
        def on_load(patcher, device_to, lowvram_model_memory, force_patch_weights, full_load):
            # Runs AFTER the loader settles and OUTSIDE any inference stream.
            # Moving 10.5 GB of module storage from inside the first forward —
            # while async-offload streams still hold events on those tensors —
            # is what produced cudaErrorIllegalAddress at step 0.
            torch.cuda.synchronize()
            place()
            state["t_emb"] = None
            state["rope"] = None

        def place():
            # NATIVE VBAR PLACEMENT (v3). Under dynamic VRAM, weights are not
            # plain tensors: each manual-cast module m carries m._v, an
            # allocation in a per-device comfy_aimdo arena (ModelVBAR), bound in
            # ModelPatcherDynamic.load via . Moving
            # tensors with .to() behind that system leaves the arena holding the
            # originals — the un-freeable "stale copy" OOM measured on Q5-Q8.
            #
            # The native operation is REBINDING: unpin the module from the
            # load_device arena and re-allocate its bytes in a cuda:1 arena
            # (model.dynamic_vbars is a dict keyed by device — multi-arena is
            # architecturally supported). Clearing _v_signature makes the next
            # forward FAULT the weights from the host pins straight into the
            # new arena, through comfy's own lazy-load path: correct eviction,
            # watermarks, and per-device offload streams for free. The compute
            # side needs nothing new — cast_modules_with_vbar takes its device
            # from the INPUT tensor, and the block hooks already feed cuda:1.
            import comfy_aimdo.model_vbar
            dev_b = torch.device(device_b)
            vbar_b = dm_patcher_model.dynamic_vbars.get(dev_b)
            if vbar_b is None:
                size = sum(pp.numel() * pp.element_size()
                           for i in range(split_index, n)
                           for pp in blocks[i].parameters())
                # x10 virtual headroom, same rationale as _vbar_get: VA space is
                # free and casts can inflate the footprint.
                vbar_b = comfy_aimdo.model_vbar.ModelVBAR(size * 10, dev_b.index)
                dm_patcher_model.dynamic_vbars[dev_b] = vbar_b
                vbar_b.prioritize()
            import comfy_aimdo.torch as aimdo_torch
            import comfy.memory_management
            from comfy.ops import materialize_meta_param
            rebound = moved = refaulted = 0
            # Release cuda:1's torch pool (previous shot's activations, boundary
            # copies) before faulting arena pages. soft_empty_cache() only trims
            # the CURRENT device, which is cuda:0.
            with torch.cuda.device(dev_b):
                torch.cuda.empty_cache()
            for i in range(split_index, n):
                for mod in blocks[i].modules():
                    v = getattr(mod, "_v", None)
                    if v is None:
                        continue
                    if v[0] is vbar_b:
                        # RE-LOAD of an already-rebound module. ComfyUI's loader
                        # keeps an existing _v (model_patcher: `not hasattr(m,
                        # "_v")`), so this module still owns its cuda:1 alloc.
                        # Allocating a fresh one here (the old behaviour) leaks a
                        # full copy of blocks s..49 into a bump-only arena on
                        # every load -> "vbar OOM while rebinding" on the 2nd
                        # sampler / 2nd prompt. Re-fault the SAME alloc instead,
                        # exactly like ops.cast_modules_with_vbar does.
                        with torch.cuda.device(dev_b):
                            sig = comfy_aimdo.model_vbar.vbar_fault(v)
                            if sig is None:
                                raise RuntimeError(
                                    f"[H3PipelineSplit] cuda:{dev_b.index} vbar OOM re-faulting "
                                    f"block {i} — RAISE split_index (fewer blocks on {device_b}) or free VRAM")
                            if not comfy_aimdo.model_vbar.vbar_signature_compare(
                                    sig, getattr(mod, "_v_signature", None)):
                                # pages were evicted: refill the same views
                                materialize_meta_param(mod, ["weight", "bias"])
                                mod._v_weight.copy_(mod.weight, non_blocking=False)
                                if mod.bias is not None and mod._v_bias is not None:
                                    mod._v_bias.copy_(mod.bias, non_blocking=False)
                                refaulted += 1
                            mod._v_signature = sig
                        continue
                    # EAGER fault-in. The lazy path cannot serve a fresh
                    # allocation: pin/file-slice descriptors were registered for
                    # the ORIGINAL load-time allocs, so a re-fault falls back to
                    # HostBuffer.read_file_slice and dies. Instead: allocate in
                    # the cuda:1 arena, fault its pages now, copy the weights in
                    # from the mmap'd source tensors, and stamp the residency
                    # signature. Compute then always takes the resident fast
                    # path (s._v_weight/_v_bias views) and never asks the pins.
                    materialize_meta_param(mod, ["weight", "bias"])
                    # kitchen QT copy_ launches kernels on the CURRENT device;
                    # writing cuda:1 arena memory from a cuda:0 context is the
                    # cudaErrorInvalidValue seen on the first eager attempt.
                    guard = torch.cuda.device(dev_b)
                    guard.__enter__()
                    new_v = vbar_b.alloc(v[2])
                    dest = aimdo_torch.aimdo_to_tensor(new_v, dev_b)
                    sig = comfy_aimdo.model_vbar.vbar_fault(new_v)
                    if sig is None:
                        # fault returns None on arena OOM. Stamping sig=None
                        # would leave the module permanently not-resident and
                        # the copies below would touch unfaulted pages.
                        guard.__exit__(None, None, None)
                        raise RuntimeError(
                            f"[H3PipelineSplit] cuda:{dev_b.index} vbar OOM while "
                            f"rebinding block {i} — RAISE split_index (fewer blocks on {device_b}) or free VRAM")
                    views = comfy.memory_management.interpret_gathered_like(
                        [mod.weight, mod.bias], dest)
                    views[0].copy_(mod.weight, non_blocking=False)
                    if mod.bias is not None and views[1] is not None:
                        views[1].copy_(mod.bias, non_blocking=False)
                    comfy_aimdo.model_vbar.vbar_unpin(v)   # release arena-A pages
                    mod._v = new_v
                    mod._v_weight = views[0]
                    mod._v_bias = views[1] if mod.bias is not None else None
                    mod._v_signature = sig
                    guard.__exit__(None, None, None)
                    rebound += 1

            # Plain (non-vbar) tensors in the same blocks. The loader
            # force-loads every module under 16 KB as ordinary parameters on
            # device_to — in this DiT that is the norms and modulation vectors
            # — and moves ALL buffers there too. Rebinding vbars alone leaves
            # those on cuda:0, so each cuda:1 block re-copies them across the
            # bus every step (host-staged here: no P2P), and anything read
            # without going through cast_bias_weight is a cuda:0 tensor inside
            # a cuda:1 kernel launch. They are small and ordinary, so a plain
            # .to() both frees the source and ends the chatter.
            import comfy.utils
            for i in range(split_index, n):
                for mod in blocks[i].modules():
                    has_v = getattr(mod, "_v", None) is not None
                    for name, p in list(mod.named_parameters(recurse=False)):
                        # weight/bias of a vbar module are the mmap'd SOURCES,
                        # not live weights — moving those materialises GB.
                        if has_v and name in ("weight", "bias"):
                            continue
                        if p is None or p.is_meta or p.device.type != "cuda" or p.device == dev_b:
                            continue
                        comfy.utils.set_attr_param(mod, name, p.data.to(dev_b))
                        moved += 1
                    for name, b in list(mod.named_buffers(recurse=False)):
                        if b is None or b.is_meta or b.device.type != "cuda" or b.device == dev_b:
                            continue
                        comfy.utils.set_attr_buffer(mod, name, b.to(dev_b))
                        moved += 1

            torch.cuda.synchronize()
            logging.info(f"[H3PipelineSplit] vbar-rebound {rebound} modules of blocks "
                         f"{split_index}..{n-1} to {device_b} arena; re-faulted {refaulted} evicted; "
                         f"moved {moved} plain params/buffers to {device_b}")
            state["placed"] = True

        def diagnose(failed_block):
            """Inventory logged once if a device_b block raises.

            A cross-device failure poses exactly three questions, and each
            costs a ~6-minute rerun to answer by bisection: are the rebound
            weights being served from the resident fast path or rerouted
            through the transfer path, does any live tensor still carry a
            device_a label, and did the plain params actually move. Read-only —
            it must not fault or pin anything, or it perturbs the state it is
            reporting on.
            """
            from collections import Counter
            vw, plain, resident = Counter(), Counter(), Counter()
            for i in range(split_index, n):
                for mod in blocks[i].modules():
                    has_v = getattr(mod, "_v", None) is not None
                    if has_v:
                        w = getattr(mod, "_v_weight", None)
                        vw[str(w.device) if w is not None else "unset"] += 1
                        pf = getattr(mod, "_prefetch", None)
                        if pf is not None:
                            resident[str(pf.get("resident"))] += 1
                    for name, p in mod.named_parameters(recurse=False):
                        if not (has_v and name in ("weight", "bias")):
                            plain[str(p.device)] += 1
                    for _, b in mod.named_buffers(recurse=False):
                        plain[str(b.device)] += 1
            return (f"block {failed_block} raised. blocks {split_index}..{n-1}: "
                    f"_v_weight devices={dict(vw)}; plain param/buffer devices={dict(plain)}; "
                    f"live _prefetch resident={dict(resident) or 'none'}")

        def hook(i):
            def replace(args, extra):
                h = args["img"]
                t_emb = args["t_emb"]
                rope = args["rope_freqs"]
                if i >= split_index and state["placed"]:
                    # h crosses the boundary once (block s); later blocks see it
                    # already resident. t_emb and rope are shared by every block
                    # of one forward — refresh the cuda:1 copies at the boundary
                    # block, reuse for the rest of the forward. (This was keyed
                    # on id(rope) before, but CPython reuses freed addresses
                    # across steps, which would silently serve a STALE t_emb —
                    # the boundary block is the deterministic per-forward event.)
                    if h.device != torch.device(device_b):
                        h = h.to(device_b)
                    if i == split_index or state["t_emb"] is None:
                        state["t_emb"] = _move_tree(t_emb, device_b)
                        state["rope"] = _move_tree(rope, device_b)
                    t_emb = state["t_emb"]
                    rope = state["rope"]
                    # Guard the CURRENT device for the whole block: sage
                    # attention and the int8 ConvRot dequant launch kernels on
                    # the current device, not the tensor's. Without this the
                    # first cuda:1 block dies with cudaErrorIllegalAddress —
                    # the same off-device-launch class as exllamav3 #260.
                    with torch.cuda.device(device_b):
                        try:
                            out = extra["original_block"](
                                {"img": h, "t_emb": t_emb, "mod_segments": args["mod_segments"],
                                 "rope_freqs": rope,
                                 "transformer_options": args["transformer_options"]})["img"]
                        except Exception:
                            if not state["diagnosed"]:
                                state["diagnosed"] = True
                                logging.error("[H3PipelineSplit] " + diagnose(i))
                            raise
                    if i == len(blocks) - 1:
                        out = out.to(device_a)      # home for final_layer + sampler
                    return {"img": out}
                return {"img": extra["original_block"](
                    {"img": h, "t_emb": t_emb, "mod_segments": args["mod_segments"],
                     "rope_freqs": rope, "transformer_options": args["transformer_options"]})["img"]}
            return replace

        m.add_callback(__import__("comfy").patcher_extension.CallbacksMP.ON_LOAD, on_load)

        _install_prefetch_guard()
        to = m.model_options.setdefault("transformer_options", {})
        # Marker consumed by the prefetch guard: vbar prefetch stages transfers
        # with the model-level device and mislabels the cuda:1 half. See
        # _install_prefetch_guard.
        to["h3_dualpipe_split"] = int(split_index)
        pr = to.setdefault("patches_replace", {})
        dit = pr.setdefault("dit", {})
        for i in range(n):
            dit[("double_block", i)] = hook(i)

        # Everything after the final block (final_layer, unpatchify) runs on
        # device_b via input-device casting; the sampler's math runs on
        # device_a. One hook moves the model output home.
        prev = m.model_options.get("model_function_wrapper")

        def output_home(apply_model, args):
            out = apply_model(args["input"], args["timestep"], **args["c"]) if prev is None \
                else prev(apply_model, args)
            return _move_tree(out, args["input"].device)

        m.model_options["model_function_wrapper"] = output_home
        logging.info(f"[H3PipelineSplit] armed: split at block {split_index}, "
                     f"{device_a} -> {device_b}")
        return (m,)


NODE_CLASS_MAPPINGS = {"H3PipelineSplit": H3PipelineSplit}
NODE_DISPLAY_NAME_MAPPINGS = {"H3PipelineSplit": "H3 Pipeline Split (2-GPU)"}
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

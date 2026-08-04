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
  * mod_segments is pure python ints; the prefetch queue is None unless
    prefetch_dynamic_vbars is set. Neither fights the split.
  * The output hook moves the result back to the sampler's device.

Placement happens lazily on the first hooked forward, AFTER ComfyUI's loader
has done its partial-load dance, and is idempotent. Use a dedicated server for
A/B runs: once blocks have been moved, an unhooked (baseline) run in the same
process would stream cuda:1-resident weights per call through the manual-cast
path — functional but not a clean baseline.
"""
import logging

import torch


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

        state = {"placed": False, "xfer_id": None, "t_emb": None, "rope": None}

        dm_patcher_model = m.model
        def on_load(patcher, device_to, lowvram_model_memory, force_patch_weights, full_load):
            # Runs AFTER the loader settles and OUTSIDE any inference stream.
            # Moving 10.5 GB of module storage from inside the first forward —
            # while async-offload streams still hold events on those tensors —
            # is what produced cudaErrorIllegalAddress at step 0.
            torch.cuda.synchronize()
            place()
            state["xfer_id"] = None

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
            rebound = skipped = 0
            for i in range(split_index, n):
                for mod in blocks[i].modules():
                    v = getattr(mod, "_v", None)
                    if v is None:
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
            torch.cuda.synchronize()
            logging.info(f"[H3PipelineSplit] vbar-rebound {rebound} modules of blocks "
                         f"{split_index}..{n-1} to {device_b} arena "
                         f"({skipped} plain cuda params left in place)")
            state["placed"] = True

        def hook(i):
            def replace(args, extra):
                h = args["img"]
                t_emb = args["t_emb"]
                rope = args["rope_freqs"]
                if i >= split_index and state["placed"]:
                    # h crosses the boundary once (block s); later blocks see it
                    # already resident. t_emb and rope are the same objects for
                    # every block of one forward — move once, reuse by identity.
                    if h.device != torch.device(device_b):
                        h = h.to(device_b)
                    if state["xfer_id"] != id(rope):
                        state["xfer_id"] = id(rope)
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
                        out = extra["original_block"](
                            {"img": h, "t_emb": t_emb, "mod_segments": args["mod_segments"],
                             "rope_freqs": rope,
                             "transformer_options": args["transformer_options"]})["img"]
                    if i == len(blocks) - 1:
                        out = out.to(device_a)      # home for final_layer + sampler
                    return {"img": out}
                return {"img": extra["original_block"](
                    {"img": h, "t_emb": t_emb, "mod_segments": args["mod_segments"],
                     "rope_freqs": rope, "transformer_options": args["transformer_options"]})["img"]}
            return replace

        m.add_callback(__import__("comfy").patcher_extension.CallbacksMP.ON_LOAD, on_load)

        to = m.model_options.setdefault("transformer_options", {})
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

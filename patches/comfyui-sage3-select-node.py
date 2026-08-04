"""Make SageAttention 3 (Blackwell FP4) reachable from a workflow.

ComfyUI v0.30.1 already *dispatches* sage3 — comfy/ldm/modules/attention.py
imports `sageattn3.sageattn3_blackwell` and registers it as attention function
"sage3". What it does not have is any way to SELECT it: `--use-sage-attention`
hard-selects sage2, and no stock node touches the override hook. So sage3 ships
dead unless something sets it.

The hook is in attention.py's wrapper:

    if "optimized_attention_override" in transformer_options:
        return transformer_options["optimized_attention_override"](func, *args, **kwargs)

i.e. the override is called with the original function first, then the real
attention arguments. This node just puts `attention3_sage` there.

Why bother: H3 runs full dense attention over a packed sequence — ~58% of FLOPs
at a 5s clip and ~80% at 15s. SageAttention 2 already measured about -50% at
real clip length on this box; sage3 quantises Q, K AND V to FP4 with per-block
microscaling, which is native on sm_120 (these 5060 Tis), so it is the one
remaining lever of comparable size.

FP4 is a real precision cut, not a free win. Judge output, not just the clock.
"""
import logging


class SageAttention3Select:
    """Route this model's attention through SageAttention 3 (FP4, sm_120)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL",)}}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "advanced/attention"
    DESCRIPTION = "Use SageAttention 3 (Blackwell FP4) for this model's attention."

    def apply(self, model):
        from comfy.ldm.modules.attention import get_attention_function

        fn = get_attention_function("sage3", None)
        if fn is None:
            raise RuntimeError(
                "sage3 is not registered. Install the kernel with:\n"
                "  cd /tmp/SageAttention/sageattention3_blackwell && "
                "uv pip install --python .venv/bin/python --no-build-isolation .\n"
                "It needs nvcc on PATH and CUDAHOSTCXX=/usr/bin/g++-15 "
                "(CUDA 13.3 rejects Fedora's default gcc 16)."
            )

        m = model.clone()
        # copy, don't mutate: model_options is shared with the source patcher and
        # an in-place edit would silently apply sage3 to every other branch of
        # the graph that reused this model.
        opts = dict(m.model_options.get("transformer_options", {}))

        def override(func, *args, **kwargs):
            kwargs.pop("_inside_attn_wrapper", None)
            return fn(*args, **kwargs)

        opts["optimized_attention_override"] = override
        m.model_options["transformer_options"] = opts
        logging.info("SageAttention3Select: attention routed to sage3 (FP4)")
        return (m,)


NODE_CLASS_MAPPINGS = {"SageAttention3Select": SageAttention3Select}
NODE_DISPLAY_NAME_MAPPINGS = {"SageAttention3Select": "SageAttention 3 (FP4)"}
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

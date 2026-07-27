#!/usr/bin/env python3
"""Generate on the NPU. NPU needs static shapes, so we declare max prompt/gen
lengths at pipeline construction (GenAI compiles a fixed-shape graph)."""
import sys, time
import openvino_genai as ov_genai

model_dir = sys.argv[1]
n_gen = int(sys.argv[2]) if len(sys.argv) > 2 else 64

t0 = time.time()
# NPU-specific: static shapes via MAX_PROMPT_LEN / MIN_RESPONSE_LEN.
pipe = ov_genai.LLMPipeline(model_dir, "NPU", MAX_PROMPT_LEN=1024, MIN_RESPONSE_LEN=n_gen)
print(f"[load] NPU compiled in {time.time()-t0:.1f}s")

prompt = "Write a Python function is_even(n) with a one-line docstring and one assert test."
cfg = ov_genai.GenerationConfig(); cfg.max_new_tokens = n_gen; cfg.do_sample = False
st = {"n": 0, "tf": None}
def sr(x):
    if st["tf"] is None: st["tf"] = time.time()
    st["n"] += 1; return ov_genai.StreamingStatus.RUNNING
t1 = time.time()
out = pipe.generate(prompt, cfg, sr); te = time.time()
tps = (st["n"]-1)/(te-st["tf"]) if st["n"] > 1 else float("nan")
print(f"[NPU] gen_tokens={st['n']}  generation={tps:.2f} tok/s  TTFT={(st['tf']-t1)*1000:.0f}ms")
print("---- NPU output ----\n" + out[:400])

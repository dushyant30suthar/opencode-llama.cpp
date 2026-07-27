#!/usr/bin/env python3
"""Measure OpenVINO GenAI text-gen speed on an IR model, and account for the
GPU memory afterwards.

Handles both plain LLM and VLM exports: Qwen3.6 ships as a VLM
(Qwen3_5ForConditionalGeneration), whose graph takes `inputs_embeds` rather than
`input_ids`, so LLMPipeline constructs fine and then dies inside generate() with
"Port for tensor name input_ids was not found". We pick by config.

On teardown: the pipeline classes expose no close()/__exit__ and no
release_memory() — that lives on ov.CompiledModel, one layer below, which the
pipeline does not surface. So the only lever from here is dropping the last
reference and forcing a collect. This script reports available memory at three
points so a leak can be attributed rather than assumed:

    before load -> after explicit release (still in-process) -> [caller checks after exit]

If memory returns at "after release", nothing leaks and earlier runs simply never
released. If it only returns after the process exits, the destructor is doing it.
If it returns at neither point, it is genuinely leaked below us.

Usage: bench_genai.py <ir-model-dir> [DEVICE] [n_gen]
"""
import gc
import json
import os
import sys
import time

import openvino_genai as ov_genai


def available_gb() -> float:
    with open("/proc/meminfo") as fh:
        for line in fh:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1048576
    return float("nan")


def is_vlm(model_dir: str) -> bool:
    try:
        with open(os.path.join(model_dir, "config.json")) as fh:
            return "vision_config" in json.load(fh)
    except OSError:
        return False


model_dir = sys.argv[1]
device = sys.argv[2] if len(sys.argv) > 2 else "GPU"
n_gen = int(sys.argv[3]) if len(sys.argv) > 3 else 128

mem_before = available_gb()
print(f"[mem] before load: {mem_before:.1f}G")

vlm = is_vlm(model_dir)
t0 = time.time()
pipe = (ov_genai.VLMPipeline if vlm else ov_genai.LLMPipeline)(model_dir, device)
print(f"[load] {'VLM' if vlm else 'LLM'}Pipeline on {device}: {time.time() - t0:.1f}s")

prompt = (
    "You are a coding assistant. Write a complete Python implementation of an "
    "LRU cache class with get, put and delete methods, plus pytest unit tests."
)
cfg = ov_genai.GenerationConfig()
cfg.max_new_tokens = n_gen
cfg.do_sample = False

state = {"n": 0, "first": None}


def streamer(_subword):
    if state["first"] is None:
        state["first"] = time.time()
    state["n"] += 1
    return ov_genai.StreamingStatus.RUNNING


t1 = time.time()
pipe.generate(prompt, generation_config=cfg, streamer=streamer)
t2 = time.time()

gen_s = t2 - state["first"]
tps = (state["n"] - 1) / gen_s if state["n"] > 1 and gen_s > 0 else float("nan")
print(f"[{device}] gen_tokens={state['n']}  generation={tps:.2f} tok/s  TTFT={(state['first'] - t1) * 1000:.0f}ms")

# Explicit teardown — the point of the accounting below.
del pipe
gc.collect()
time.sleep(2)  # give the driver a beat to hand pages back
mem_after = available_gb()
print(f"[mem] after release (in-process): {mem_after:.1f}G  (recovered {mem_after - mem_before:+.1f}G vs before-load)")
print("[mem] check available again after this process exits to attribute the rest")

"""The H3 experiment matrix for 2x RTX 5060 Ti — definitions only, no execution.

Two axes, and they cost very different amounts:

  LAUNCH flags    need a server restart (~10s boot + ~30-60s to reload a 21 GB
                  checkpoint off NVMe). Expensive. Outer loop.
  WORKFLOW params are just a different graph against a resident model. Free.
                  Inner loop.

sweep.py runs them in that nesting so one restart amortises over many runs.

TWO STAGES, mirroring the knob-ab / deep-confirm split used for the LLM work:
  PROBE    length 22, 10 steps, cfg 1.0 — deliberately below the trained frame
           range. Quality is meaningless here; this is a throughput probe, and
           it keeps a matrix row near ~2 min instead of ~40.
  CONFIRM  length 124 (5.2s, the bottom of the trained 124-362 range), 20 steps.
           Only the top few configs earn this.

CAVEAT when reading results: attention is quadratic in token count, so its SHARE
of total work grows with clip length — 22 frames is linear-dominated, 362 frames
is ~80% attention. An attention-backend win measured at PROBE length UNDERSTATES
what it is worth at 15s. That is why phase A is also run at CONFIRM length
regardless of how it ranks.
"""

# ---------------------------------------------------------------------------
# Geometry, straight out of comfy_extras/nodes_minimax_h3.py
# ---------------------------------------------------------------------------

def align_frame_count(n):
    while n % 17 != 5:
        n += 1
    return n


def video_latent_t(fc):
    return 2 if fc <= 5 else ((fc - 5) // 17) * 5 + 2


def tokens(width, height, length):
    """What the DiT attends over — the number that predicts cost."""
    fc = align_frame_count(max(5, length))
    t = video_latent_t(fc)
    # patch_size (1,2,2) on a /16 latent grid => /32 in pixel space
    return t * (height // 32) * (width // 32) + 2 * round(fc / 24 * 40)


FL2VA = "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
REF2VA = "minimax_h3_ref2va_pruned_int8_convrot.safetensors"
ENCODER = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
VIDEO_VAE = "minimax_h3_video_vae_fp16.safetensors"
AUDIO_VAE = "minimax_h3_audio_vae_fp32.safetensors"

PROMPT = ("A slow dolly shot through a rain-soaked Tokyo alley at night, neon signs "
          "reflecting in puddles, steam rising from a ramen stall.")

PROBE = dict(width=1344, height=768, length=22, steps=10, cfg=1.0,
             sampler="euler", scheduler="simple", shift_video=12.0, shift_audio=3.0,
             model_device="default", clip_device="default", vae_device="default")

CONFIRM = dict(PROBE, length=124, steps=20)


def build_workflow(width, height, length, steps, cfg, sampler, scheduler,
                   shift_video, shift_audio,
                   model_device="default", clip_device="default", vae_device="default",
                   seed=42, unet=FL2VA, prompt=PROMPT,
                   cache=None, cache_threshold=0.2, cache_start=0.15, cache_end=0.95,
                   compile_backend=None, sage3=False, distorch=None, dualpipe=None):
    """API-format graph, structurally identical to workflows/h3/h3-t2v-dualgpu.json.

    Device nodes are inserted only when not "default": an identity
    SelectModelDevice would otherwise be an uncontrolled variable in a benchmark
    whose whole point is isolating one knob.
    """
    # DisTorch2 replaces the stock loader entirely: it shards the DiT's layers
    # across devices at load time. The point on this box is `donor_device`,
    # which defaults to "cpu" — that default is what makes a 20.97 GB model on a
    # 15.5 GiB card stream ~6 GB from host RAM every step while GPU1 sits idle
    # with 15.5 GiB free. Pointing the donor at cuda:1 moves that spill into
    # VRAM. SelectModelDevice is skipped when this is active; both manage
    # placement and stacking them fights.
    if distorch:
        g = {
            "1": {"class_type": "UNETLoaderDisTorch2MultiGPU",
                  "inputs": {"unet_name": unet, "weight_dtype": "default",
                             "compute_device": distorch.get("compute", "cuda:0"),
                             "virtual_vram_gb": distorch.get("vvram", 6.0),
                             "donor_device": distorch.get("donor", "cuda:1"),
                             "expert_mode_allocations": distorch.get("expert", ""),
                             "eject_models": True}},
        }
    else:
        g = {
            "1": {"class_type": "UNETLoader",
                  "inputs": {"unet_name": unet, "weight_dtype": "default"}},
        }
    g.update({
        "3": {"class_type": "CLIPLoader",
              "inputs": {"clip_name": ENCODER, "type": "minimax", "device": "default"}},
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": VIDEO_VAE}},
        "5": {"class_type": "VAELoader", "inputs": {"vae_name": AUDIO_VAE}},
    })
    model_src, clip_src, vvae_src, avae_src = ["1", 0], ["3", 0], ["4", 0], ["5", 0]

    if model_device != "default" and not distorch:
        g["15"] = {"class_type": "SelectModelDevice",
                   "inputs": {"model": model_src, "device": model_device}}
        model_src = ["15", 0]
    if clip_device != "default":
        g["16"] = {"class_type": "SelectCLIPDevice",
                   "inputs": {"clip": clip_src, "device": clip_device}}
        clip_src = ["16", 0]
    if vae_device != "default":
        g["17"] = {"class_type": "SelectVAEDevice",
                   "inputs": {"vae": vvae_src, "device": vae_device}}
        g["18"] = {"class_type": "SelectVAEDevice",
                   "inputs": {"vae": avae_src, "device": vae_device}}
        vvae_src, avae_src = ["17", 0], ["18", 0]

    # Pipeline split: blocks >= split_index live and execute on cuda:1.
    # Placed before the cache wrapper so caching sees the split model.
    if dualpipe:
        g["23"] = {"class_type": "H3PipelineSplit",
                   "inputs": {"model": model_src, "split_index": int(dualpipe),
                              "device_a": "cuda:0", "device_b": "cuda:1"}}
        model_src = ["23", 0]

    g["2"] = {"class_type": "MiniMaxH3SigmaShift",
              "inputs": {"model": model_src, "shift_video": shift_video,
                         "shift_audio": shift_audio}}
    sampled_model = ["2", 0]

    # SageAttention 3 is selected per-model, not by a launch flag — see
    # custom_nodes/sage3_select. It must sit before the cache node so the cache
    # wraps the already-overridden attention.
    if sage3:
        g["22"] = {"class_type": "SageAttention3Select", "inputs": {"model": sampled_model}}
        sampled_model = ["22", 0]

    # Step-caching: reuse a denoise step's output when the trajectory has moved
    # less than reuse_threshold. H3 is guidance-distilled but NOT step-distilled,
    # so there is no lightning LoRA to cut steps for free — caching is the only
    # way to buy back step cost, and it trades fidelity for it. Reported ~30%
    # off denoise at a real SSIM cost, so it is measured, not assumed.
    if cache in ("easy", "lazy"):
        node = "EasyCache" if cache == "easy" else "LazyCache"
        g["20"] = {"class_type": node,
                   "inputs": {"model": sampled_model, "reuse_threshold": cache_threshold,
                              "start_percent": cache_start, "end_percent": cache_end,
                              "verbose": False}}
        sampled_model = ["20", 0]

    # torch.compile pays a large one-off cost and recompiles on shape change —
    # every distinct clip length is a new shape here, so it only pays off for a
    # fixed-geometry batch.
    if compile_backend:
        g["21"] = {"class_type": "TorchCompileModel",
                   "inputs": {"model": sampled_model, "backend": compile_backend}}
        sampled_model = ["21", 0]
    g["6"] = {"class_type": "MiniMaxH3ImageToVideo",
              "inputs": {"clip": clip_src, "vae": vvae_src, "prompt": prompt,
                         "width": width, "height": height, "length": length}}
    g["7"] = {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["6", 0]}}
    g["8"] = {"class_type": "KSampler",
              "inputs": {"model": sampled_model, "seed": seed, "steps": steps, "cfg": cfg,
                         "sampler_name": sampler, "scheduler": scheduler,
                         "positive": ["6", 0], "negative": ["7", 0],
                         "latent_image": ["6", 1], "denoise": 1.0}}
    g["9"] = {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": ["8", 0]}}
    g["10"] = {"class_type": "VAEDecode", "inputs": {"samples": ["9", 0], "vae": vvae_src}}
    g["11"] = {"class_type": "LTXVAudioVAEDecode",
               "inputs": {"samples": ["9", 1], "audio_vae": avae_src}}
    g["12"] = {"class_type": "CreateVideo",
               "inputs": {"images": ["10", 0], "fps": 24.0, "audio": ["11", 0]}}
    g["13"] = {"class_type": "SaveVideo",
               "inputs": {"video": ["12", 0], "filename_prefix": "video/bench",
                          "format": "auto", "codec": "auto"}}
    return g


# ---------------------------------------------------------------------------
# LAUNCH-FLAG PHASES  (restart per row)
# `needs` names a python module that must import, else the row is skipped.
# ---------------------------------------------------------------------------

LAUNCH_PHASES = [
    ("A", "attention backend — the dominant term (58% of FLOPs at 5s, 80% at 15s)", [
        ("A0", [], None, "default (torch SDPA, flash-capable)"),
        ("A1", ["--use-pytorch-cross-attention"], None, "explicit torch 2.x SDPA"),
        ("A2", ["--use-sage-attention"], "sageattention",
         "SageAttention 2 — INT8 QK, FP16 V; ~30-35% over SDPA reported on Blackwell"),
        ("A3", ["--use-flash-attention"], "flash_attn", "explicit FlashAttention"),
        ("A4", ["--use-split-cross-attention"], None,
         "memory-thrifty, expected slower; run it to bound the axis"),
    ]),

    ("B", "--fast optimizations (applied on top of the A winner)", [
        ("B0", [], None, "control"),
        ("B1", ["--fast", "fp16_accumulation"], None, "fp16 accumulate in GEMMs"),
        ("B2", ["--fast", "fp16_accumulation", "cublas_ops"], None, "+ cuBLAS paths"),
        ("B3", ["--fast", "autotune"], None, "kernel autotune; the first run pays for it"),
        ("B4", ["--fast"], None, "everything — ComfyUI calls these untested and quality-affecting"),
    ]),

    ("C", "offload & memory — the 20.97 GB DiT against 15.5 GiB per card", [
        ("C0", [], None, "default (dynamic VRAM on, async offload on)"),
        ("C1", ["--async-offload", "4"], None, "more offload streams to hide PCIe latency"),
        ("C2", ["--disable-async-offload"], None, "control: is async offload helping at all"),
        ("C3", ["--fast-disk"], None, "stream from NVMe rather than unpinned RAM — this box has one"),
        ("C4", ["--highvram"], None, "keep everything resident; may OOM at 20.97 GB"),
        ("C5", ["--disable-smart-memory"], None, "evict aggressively between stages"),
        ("C6", ["--reserve-vram", "0.5"], None, "minimum OS headroom"),
        ("C7", ["--reserve-vram", "2.0"], None, "the OOM remedy — costs resident weights"),
        ("C8", ["--force-non-blocking"], None, "non-blocking H2D copies"),
        ("C9", ["--disable-pinned-memory"], None, "control on the pinned-RAM path"),
        ("C10", ["--force-channels-last"], None, "memory layout; sometimes large on the VAE"),
        ("C11", ["--cache-none"], None, "lowest RAM; re-executes every node"),
        ("C12", ["--high-ram"], None, "opposite end — only 31 GiB total, watch for zram swap"),
    ]),

    ("H", "encoder & VAE precision — 15.69 GB encoder + 5.21 GB video VAE", [
        ("H0", [], None, "control"),
        ("H1", ["--fp8_e4m3fn-text-enc"], None, "halve the encoder's footprint"),
        ("H2", ["--lowvram"], None, "encoder to CPU (only bites if dynamic vram is off)"),
        ("H3", ["--cpu-vae"], None, "VAE decode on CPU — frees VRAM at the OOM-prone step"),
        ("H4", ["--bf16-vae"], None, "vs the fp16/fp32 files as shipped"),
        ("H5", ["--fp16-intermediates"], None, "fp16 between nodes instead of fp32"),
    ]),

    ("I", "single-card control", [
        ("I0", [], None, "both cards visible"),
        ("I1", ["--cuda-device", "0"], None,
         "hide GPU1 — isolates whether placement helps or hurts"),
    ]),
]

# ---------------------------------------------------------------------------
# WORKFLOW PHASES  (no restart)
# ---------------------------------------------------------------------------

WORKFLOW_PHASES = [
    ("D", "device placement — free to vary, no restart", [
        ("D0", dict(), "everything default -> cuda:0"),
        ("D1", dict(model_device="gpu:0", clip_device="gpu:1", vae_device="gpu:1"),
         "the dualgpu workflow: DiT owns cuda:0"),
        ("D2", dict(model_device="gpu:0", clip_device="cpu", vae_device="gpu:1"),
         "encoder on CPU — slow once, but frees 15.69 GB"),
        ("D3", dict(model_device="gpu:0", clip_device="gpu:1", vae_device="gpu:0"),
         "VAE back on the hot card"),
        ("D4", dict(model_device="gpu:1", clip_device="gpu:0", vae_device="gpu:0"),
         "mirrored — GPU1 sits on the x4 slot, so expect worse"),
    ]),

    ("E", "sampler / scheduler — quality-affecting, so judge the output too", [
        ("E0", dict(sampler="euler", scheduler="simple"), "baseline"),
        ("E1", dict(sampler="euler", scheduler="beta"), ""),
        ("E2", dict(sampler="euler", scheduler="normal"), ""),
        ("E3", dict(sampler="res_multistep", scheduler="beta"),
         "a community report claims res_multistep+beta beats simple on H3"),
        ("E4", dict(sampler="res_multistep", scheduler="normal"), ""),
        ("E5", dict(sampler="dpmpp_2m", scheduler="sgm_uniform"), ""),
        ("E6", dict(sampler="uni_pc", scheduler="simple"), ""),
        ("E7", dict(sampler="euler", scheduler="kl_optimal"), ""),
    ]),

    ("F", "cfg and steps — the two biggest wall-clock levers", [
        ("F0", dict(cfg=1.0), "1.0 skips the negative pass entirely: HALVES total compute"),
        ("F1", dict(cfg=3.0), ""),
        ("F2", dict(cfg=5.0), "the starting guess"),
        ("F3", dict(cfg=7.0), ""),
        ("F4", dict(steps=10), ""),
        ("F5", dict(steps=15), ""),
        ("F6", dict(steps=20), ""),
        ("F7", dict(steps=30), ""),
        ("F8", dict(steps=40), ""),
    ]),

    ("G", "sigma shift — 12.0/3.0 are node defaults, unverified on this box", [
        ("G0", dict(shift_video=12.0, shift_audio=3.0), "default"),
        ("G1", dict(shift_video=8.0, shift_audio=3.0), ""),
        ("G2", dict(shift_video=16.0, shift_audio=3.0), ""),
        ("G3", dict(shift_video=12.0, shift_audio=1.5), ""),
        ("G4", dict(shift_video=12.0, shift_audio=6.0), ""),
    ]),

    ("J", "geometry scaling — validates the quadratic model, finds the usable ceiling", [
        ("J0", dict(width=768, height=768, length=124), ""),
        ("J1", dict(width=1024, height=768, length=124), ""),
        ("J2", dict(width=1344, height=768, length=124), "native canvas, 5.2s"),
        ("J3", dict(width=1344, height=768, length=209), "~8.7s"),
        ("J4", dict(width=1344, height=768, length=277), "~11.5s"),
        ("J5", dict(width=1344, height=768, length=362), "15.1s — top of the trained range"),
    ]),
]


# Added after research: H3 is guidance-distilled (so cfg=1.0 is correct and the
# negative pass is genuinely wasted work) but NOT step-distilled — no lightning
# LoRA exists. Step-caching is therefore the only remaining large lever, and all
# three nodes are MODEL patchers, so the whole phase runs on ONE server.
WORKFLOW_PHASES.append(
    ("K", "step caching & compile — the last big lever (no step-distill exists for H3)", [
        ("K0", dict(), "baseline"),
        ("K1", dict(cache="easy", cache_threshold=0.2), "EasyCache, default threshold"),
        ("K2", dict(cache="easy", cache_threshold=0.4), "EasyCache, aggressive"),
        ("K3", dict(cache="lazy", cache_threshold=0.2), "LazyCache"),
        ("K4", dict(cache="easy", cache_threshold=0.3, cache_start=0.1), "EasyCache, earlier start"),
        ("K5", dict(cache="easy", cache_threshold=0.5), "EasyCache 0.5 — pushing past the 0.4 winner"),
        ("K6", dict(cache="easy", cache_threshold=0.6), "EasyCache 0.6 — expect visible quality loss"),
        ("K7", dict(cache="easy", cache_threshold=0.4, cache_start=0.05, cache_end=1.0), "0.4 over the full trajectory"),
        ("K8", dict(cache="lazy", cache_threshold=0.4), "LazyCache at the winning threshold"),
        ("K9", dict(cache="easy", cache_threshold=0.4, compile_backend="inductor"), "cache + torch.compile"),
        ("K10", dict(sage3=True), "SageAttention 3 (FP4) alone vs sage2 baseline"),
        ("K11", dict(sage3=True, cache="easy", cache_threshold=0.4), "sage3 + EasyCache 0.4 — the endgame config"),
    ]))


# The largest untested lever. Attention is quadratic in tokens and tokens scale
# with W*H, so shrinking the canvas buys back far more than any flag. This is
# also how H3's own 2K path works: generate at a 768-ish short edge, then
# regenerate upward. Quality is the cost, so these rows need watching, not just
# timing — and note adapt_canvas() caps area at 768*1344, so 1344x768 is already
# the model's native ceiling, not an arbitrary choice.
WORKFLOW_PHASES.append(
    ("L", "resolution scaling — quadratic savings, judge quality before adopting", [
        ("L0", dict(width=1344, height=768), "native canvas (1,008 tok/frame)"),
        ("L1", dict(width=1152, height=640), "~0.68x tokens"),
        ("L2", dict(width=960, height=544), "~0.51x tokens"),
        ("L3", dict(width=832, height=480), "~0.39x tokens"),
        ("L4", dict(width=768, height=432), "~0.32x tokens — smallest sensible 16:9"),
    ]))


# The structural finding of the night: GPU1 is idle at 0% while GPU0 runs at 99%
# and ~6 GB of the DiT streams from host RAM every step. ComfyUI cannot compute a
# diffusion model across two cards, but DisTorch2 can SHARD it — so GPU1's 15.5
# GiB becomes the donor instead of system RAM. That does not make GPU1 compute;
# it removes the host-RAM round trip, which is what the PCIe measurements say is
# expensive. Also frees host RAM, which sits at 25/31 GB with one instance.
WORKFLOW_PHASES.append(
    ("M", "DisTorch2 sharding — put the DiT spill on idle GPU1 instead of host RAM", [
        ("M0", dict(), "control: stock loader, spill goes to host RAM"),
        ("M1", dict(distorch=dict(donor="cpu", vvram=6.0)), "DisTorch2 but donor=cpu (isolates the node's own overhead)"),
        ("M2", dict(distorch=dict(donor="cuda:1", vvram=6.0)), "donor=cuda:1 — the fix"),
        ("M3", dict(distorch=dict(donor="cuda:1", vvram=8.0)), "more on GPU1"),
        ("M4", dict(distorch=dict(donor="cuda:1", vvram=11.0)), "most of the spill on GPU1"),
        ("M5", dict(distorch=dict(donor="cuda:1", vvram=14.0)), "push until GPU1 is the bottleneck"),
    ]))


# The pipeline-split experiment: the one two-GPU topology that fits this box
# (no P2P, 31 GB RAM). Compute follows weights; the whole DiT becomes
# VRAM-resident; the offload tax disappears; GPU1 executes half of every step.
WORKFLOW_PHASES.append(
    ("P", "pipeline split — GPU1 executes blocks >= split_index", [
        ("P0", dict(), "baseline, single GPU"),
        ("P1", dict(dualpipe=25), "even split: 25+25 blocks"),
        ("P2", dict(dualpipe=20), "20/30: GPU0 also runs embeds+refiner, give it fewer blocks"),
        ("P3", dict(dualpipe=25, cache="easy", cache_threshold=0.4), "split + EasyCache 0.4 — the composite"),
    ]))


# Q: the pipeline split alone, ONE ROW PER INVOCATION. In a shared server the
# baseline row's full single-GPU load stays resident (comfy caches models across
# prompts), so the split row OOMs cuda:0 against its own stale copy. Each Q run
# must be a fresh server whose only load is the split one.
WORKFLOW_PHASES.append(("Q", "pipeline split, isolated server", [
    ("Q1", dict(dualpipe=22, cache="easy", cache_threshold=0.4),
     "split 22/28 + EasyCache 0.4 — balanced free (~3.8G/card) and the cache's measured -2.7 GB peak buys the workspace"),
]))


def phase(letter):
    for group in (LAUNCH_PHASES, WORKFLOW_PHASES):
        for name, desc, rows in group:
            if name == letter:
                return name, desc, rows
    raise KeyError(letter)


if __name__ == "__main__":
    total = 0
    print(f"{'phase':6} {'rows':>5}  kind         description")
    for group, kind in ((LAUNCH_PHASES, "restart"), (WORKFLOW_PHASES, "no restart")):
        for name, desc, rows in group:
            total += len(rows)
            print(f"{name:6} {len(rows):>5}  {kind:12} {desc}")
    print(f"{'':6} {total:>5}  total rows")
    print()
    print("token counts (what actually predicts cost):")
    for label, (w, h, ln) in {
        "PROBE    1344x768 len22": (1344, 768, 22),
        "CONFIRM  1344x768 len124": (1344, 768, 124),
        "         1344x768 len362": (1344, 768, 362),
        "         1920x1088 len362": (1920, 1088, 362),
    }.items():
        print(f"  {label:27} {tokens(w, h, ln):>9,} tokens")

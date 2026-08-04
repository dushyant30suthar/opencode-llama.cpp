#!/usr/bin/env python3
"""Queue an H3 workflow against a running ComfyUI and report what actually happened.

    .venv/bin/python workflows/h3/run.py h3-t2v-dualgpu.json [--port 8188] [--wait-for-weights]

Stdlib only, so it runs from any python — it talks HTTP, it does not import ComfyUI.

Why this exists rather than "just use the browser": the interesting numbers on this
box are wall-clock per step and whether the DiT ended up spilling to system RAM.
The UI shows neither in a form you can paste into a benchmark note. This prints
seconds per step, peak VRAM per card, and the output path.
"""
import argparse, json, os, sys, time, urllib.request, urllib.error

HERE = os.path.dirname(os.path.abspath(__file__))
EXPECTED = {
    "minimax_h3_fl2va_pruned_int8_convrot.safetensors": 20_970_000_000,
    "minimax_h3_ref2va_pruned_int8_convrot.safetensors": 20_970_000_000,
    "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors": 15_690_000_000,
    "minimax_h3_video_vae_fp16.safetensors": 5_210_000_000,
    "minimax_h3_audio_vae_fp32.safetensors": 610_000_000,
}


def api(base, path, payload=None, timeout=30):
    url = f"{base}{path}"
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data,
                                 headers={"Content-Type": "application/json"} if data else {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = r.read()
    return json.loads(body) if body else {}


def referenced(prompt):
    """Which model files THIS graph actually loads.

    Checking the whole H3 set would block a text-to-video run on ref2va, a
    21 GB checkpoint only the reference-to-video graph ever touches. The loaders
    name their files directly, so ask the graph.
    """
    keys = ("unet_name", "clip_name", "vae_name")
    names = set()
    for node in prompt.values():
        for k in keys:
            v = node.get("inputs", {}).get(k)
            if isinstance(v, str) and v in EXPECTED:
                names.add(v)
    return names


def weights_ready(models_root, needed):
    """Each needed file present and at least ~99% of its expected size.

    Size, not existence: ComfyUI lists a half-downloaded safetensors in the
    loader combo like any other, so the graph validates and then dies on a
    truncated read partway through loading 21 GB.
    """
    missing = []
    for name in sorted(needed):
        want = EXPECTED[name]
        hit = None
        for sub in ("diffusion_models", "text_encoders", "vae"):
            p = os.path.join(models_root, sub, name)
            if os.path.exists(p):
                hit = p
                break
        if hit is None:
            missing.append(f"{name}: absent")
        elif os.path.getsize(hit) < want * 0.99:
            missing.append(f"{name}: {os.path.getsize(hit)/1e9:.1f}/{want/1e9:.1f} GB")
    return missing


def vram(base):
    try:
        devs = api(base, "/system_stats", timeout=5).get("devices", [])
        return {d["index"]: (d["vram_total"] - d["vram_free"]) / 1024**3
                for d in devs if d.get("type") == "cuda"}
    except Exception:
        return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("workflow")
    ap.add_argument("--port", type=int, default=8188)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--wait-for-weights", action="store_true",
                    help="block until download-h3.sh has finished, then run")
    ap.add_argument("--models-root", default=os.path.join(HERE, "..", "..", "models"))
    args = ap.parse_args()

    base = f"http://{args.host}:{args.port}"
    path = args.workflow if os.path.isabs(args.workflow) else os.path.join(HERE, args.workflow)
    if not os.path.exists(path):
        sys.exit(f"no such workflow: {path}")

    with open(path) as fh:
        prompt = json.load(fh)
    needed = referenced(prompt)

    root = os.path.abspath(args.models_root)
    while True:
        missing = weights_ready(root, needed)
        if not missing:
            break
        if not args.wait_for_weights:
            print("weights not ready:")
            for m in missing:
                print("   ", m)
            sys.exit(1)
        print(f"[{time.strftime('%H:%M:%S')}] waiting on {len(missing)} file(s): {missing[0]}")
        time.sleep(120)

    try:
        stats = api(base, "/system_stats", timeout=5)
    except Exception as e:
        sys.exit(f"ComfyUI not answering on {base} ({e}). Start it from the opencode panel or:\n"
                 f"  .venv/bin/python main.py --port {args.port}")
    print(f"ComfyUI {stats.get('system',{}).get('comfyui_version','?')} on {base}")
    before = vram(base)

    t0 = time.time()
    try:
        res = api(base, "/prompt", {"prompt": prompt})
    except urllib.error.HTTPError as e:
        detail = e.read().decode()[:2000]
        sys.exit(f"rejected by validation:\n{detail}")
    pid = res.get("prompt_id")
    print(f"queued {os.path.basename(path)} as {pid}")

    peak = dict(before)
    last_note = 0.0
    while True:
        time.sleep(2)
        now = vram(base)
        for k, v in now.items():
            peak[k] = max(peak.get(k, 0), v)
        hist = api(base, f"/history/{pid}", timeout=15)
        if pid in hist:
            entry = hist[pid]
            status = entry.get("status", {})
            elapsed = time.time() - t0
            ok = status.get("status_str") == "success" or status.get("completed")
            print(f"\n{'completed' if ok else 'FAILED'} in {elapsed:.1f}s")
            if not ok:
                for m in status.get("messages", [])[-6:]:
                    print("   ", m)
            outs = []
            for node_id, out in (entry.get("outputs") or {}).items():
                for kind in ("images", "video", "gifs", "audio"):
                    for f in out.get(kind, []) or []:
                        outs.append(os.path.join("output", f.get("subfolder", ""), f.get("filename", "")))
            for o in outs:
                print("   ->", o)
            print("   peak VRAM:", ", ".join(f"cuda:{k} {v:.1f}G" for k, v in sorted(peak.items())))
            steps = prompt.get("8", {}).get("inputs", {}).get("steps")
            if ok and steps:
                print(f"   {elapsed/steps:.2f}s per step at {steps} steps")
            return 0 if ok else 2
        if time.time() - last_note > 60:
            last_note = time.time()
            cur = ", ".join(f"cuda:{k} {v:.1f}G" for k, v in sorted(vram(base).items()))
            print(f"[{time.strftime('%H:%M:%S')}] running {time.time()-t0:.0f}s  {cur}")


if __name__ == "__main__":
    sys.exit(main())

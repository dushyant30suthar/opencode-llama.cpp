import json, os, sys, time, urllib.request
S = sys.argv[1]; ROOT = os.path.expanduser("~/Projects/comfyui")
sys.path.insert(0, os.path.join(ROOT, "bench")); import matrix
def api(port, path, payload=None):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=json.dumps(payload).encode() if payload else None,
                                 headers={"Content-Type": "application/json"} if payload else {})
    return json.load(urllib.request.urlopen(req, timeout=30))
def wf(name, prompt, refs=None):
    p = dict(matrix.PROBE, width=1344, height=768, length=362, steps=8, sampler="res_multistep", scheduler="simple",
             cfg=1.0, prompt=prompt, seed=42, cache=None, cache_threshold=0.4, tiled_decode=32, chunk_heads=4,
             solattn=None, first_frame=None, last_frame=None, refs=refs, ref_videos=None, ref_size="match")
    lora = "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors"
    if refs:
        p["unet"] = matrix.REF2VA; lora = "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors"
    g = matrix.build_workflow(**p)
    g["90"] = {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["1", 0], "strength_model": 1.0, "lora_name": lora}}
    for n in g.values():
        if n["class_type"] != "LoraLoaderModelOnly" and n["inputs"].get("model") == ["1", 0]: n["inputs"]["model"] = ["90", 0]
    g["13"]["inputs"]["filename_prefix"] = f"fulltest/{name}"
    return g
jobs = [(8188, "t2v-gpu0", wf("t2v-gpu0", open(f"{S}/dog-prompt.txt").read())),
        (8189, "r2v-gpu1", wf("r2v-gpu1", open(f"{S}/r2v-prompt.txt").read(),
                               ["spilled_p1_maya.png", "spilled_p2_theo_approved.png", "spilled_p3_saleswoman.png"]))]
m0 = {l.split(":")[0]: int(l.split()[1]) for l in open("/proc/meminfo")}; swap0 = m0["SwapTotal"] - m0["SwapFree"]
t0 = time.time(); ids = []
for port, name, g in jobs:
    r = api(port, "/prompt", {"prompt": g}); print(name, r.get("node_errors") or "queued", flush=True); ids.append((port, name, r["prompt_id"]))
done = {}; min_avail = 99; peak = 0; last = 0
while len(done) < 2 and time.time() - t0 < 3600:
    m = {l.split(":")[0]: int(l.split()[1]) for l in open("/proc/meminfo")}
    avail = m["MemAvailable"] / 1048576; swap = (m["SwapTotal"] - m["SwapFree"] - swap0) / 1048576
    min_avail = min(min_avail, avail); peak = max(peak, swap)
    if avail < 1.0 or swap > 6:
        for p in (8188, 8189): api(p, "/interrupt", {}); api(p, "/queue", {"clear": True})
        print(f"ABORT avail={avail:.1f} swap+={swap:.1f}", flush=True); break
    if time.time() - last > 60: print(f"{time.time()-t0:5.0f}s avail={avail:.1f}G swap+={swap:.1f}G", flush=True); last = time.time()
    for port, name, pid in ids:
        if pid in done: continue
        h = api(port, f"/history/{pid}")
        if pid in h and h[pid]["status"].get("status_str"):
            ms = h[pid]["status"]["messages"]
            a = next((x[1]["timestamp"] for x in ms if x[0] == "execution_start"), 0); b = next((x[1]["timestamp"] for x in ms if x[0] in ("execution_success", "execution_error")), 0)
            done[pid] = (name, h[pid]["status"]["status_str"], round((b - a) / 1000), next((x[1].get("exception_message", "")[:300] for x in ms if x[0] == "execution_error"), ""))
            print(done[pid], flush=True)
    time.sleep(5)
print(f"RESULT wall={time.time()-t0:.0f}s min_avail={min_avail:.1f}G peak_swap+={peak:.1f}G")

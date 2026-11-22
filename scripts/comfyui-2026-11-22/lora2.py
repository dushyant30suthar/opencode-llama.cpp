import json, os, sys, time, urllib.request
S = sys.argv[1]; ROOT = os.path.expanduser("~/Projects/comfyui")
sys.path.insert(0, os.path.join(ROOT, "bench")); import matrix
REFS = ["spilled_p1_maya.png", "spilled_p2_theo_approved.png", "spilled_p3_saleswoman.png"]
FL2V = "minimax_h3_fl2v_turbo_8step_v1.2_768p_comfyui_bf16.safetensors"
REF2V = "minimax_h3_ref2v_turbo_8step_v1.1_768p_comfyui_bf16.safetensors"
def api(port, path, payload=None):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=json.dumps(payload).encode() if payload else None,
                                 headers={"Content-Type": "application/json"} if payload else {})
    return json.load(urllib.request.urlopen(req, timeout=30))
def wf(name, steps, shift, lora=None, r2v=False):
    p = dict(matrix.PROBE, width=960, height=544, length=124, steps=steps, sampler="res_multistep", scheduler="simple",
             cfg=1.0, prompt=open(f"{S}/{'r2v-prompt' if r2v else 'dog-prompt'}.txt").read(), seed=7 if r2v else 42,
             cache=None, cache_threshold=0.4, tiled_decode=0, chunk_heads=0, solattn=None, first_frame=None,
             last_frame=None, refs=REFS if r2v else None, ref_videos=None, ref_size="match")
    if r2v: p["unet"] = matrix.REF2VA
    g = matrix.build_workflow(**p)
    g["2"]["inputs"]["shift_video"] = float(shift)
    if lora:
        g["90"] = {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["1", 0], "strength_model": 1.0, "lora_name": lora}}
        g["2"]["inputs"]["model"] = ["90", 0]
    g["13"]["inputs"]["filename_prefix"] = f"loratest/{name}"
    return g
jobs = [(8188, "T1-fl2v-v12-8s-shift12", wf("T1-fl2v-v12-8s-shift12", 8, 12, FL2V)),
        (8189, "R0-r2v-nolora-20s",       wf("R0-r2v-nolora-20s", 20, 12, None, True)),
        (8188, "T2-fl2v-v12-8s-shift6",  wf("T2-fl2v-v12-8s-shift6", 8, 6, FL2V)),
        (8189, "R1-r2v-v11-8s-shift12",  wf("R1-r2v-v11-8s-shift12", 8, 12, REF2V, True)),
        (8188, "R2-r2v-v11-8s-shift6",   wf("R2-r2v-v11-8s-shift6", 8, 6, REF2V, True))]
t0 = time.time(); ids = [(port, name, api(port, "/prompt", {"prompt": g})["prompt_id"]) for port, name, g in jobs]
done = {}
while len(done) < len(ids) and time.time() - t0 < 2400:
    m = {l.split(":")[0]: int(l.split()[1]) for l in open("/proc/meminfo")}
    if m["MemAvailable"] < 600 << 10:
        for p in (8188, 8189): api(p, "/interrupt", {}); api(p, "/queue", {"clear": True})
        print("ABORT low RAM", flush=True); break
    for port, name, pid in ids:
        if pid in done: continue
        try: h = api(port, f"/history/{pid}")
        except Exception: continue
        if pid in h and h[pid]["status"].get("status_str"):
            ms = h[pid]["status"]["messages"]
            a = next((x[1]["timestamp"] for x in ms if x[0] == "execution_start"), 0); b = next((x[1]["timestamp"] for x in ms if x[0] in ("execution_success", "execution_error")), 0)
            done[pid] = (name, h[pid]["status"]["status_str"], round((b - a) / 1000), next((x[1].get("exception_message", "")[:200] for x in ms if x[0] == "execution_error"), ""))
            print(done[pid], flush=True)
    time.sleep(5)
print(f"wall={time.time()-t0:.0f}s")

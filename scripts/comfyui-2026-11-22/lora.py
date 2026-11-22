import json, os, sys, time, urllib.request
S = sys.argv[1]; ROOT = os.path.expanduser("~/Projects/comfyui")
sys.path.insert(0, os.path.join(ROOT, "bench")); import matrix
P = open(os.path.join(S, "dog-prompt.txt")).read()
def api(port, path, payload=None):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=json.dumps(payload).encode() if payload else None,
                                 headers={"Content-Type": "application/json"} if payload else {})
    return json.load(urllib.request.urlopen(req, timeout=30))
def wf(name, steps, lora=False, int8vae=False):
    g = matrix.build_workflow(**dict(matrix.PROBE, width=960, height=544, length=124, steps=steps, sampler="res_multistep",
        scheduler="simple", cfg=1.0, prompt=P, seed=42, cache=None, cache_threshold=0.4, tiled_decode=0, chunk_heads=0,
        solattn=None, first_frame=None, last_frame=None, refs=None, ref_videos=None, ref_size="match"))
    if lora:
        g["30"] = {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["1", 0], "strength_model": 1.0,
                   "lora_name": "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors"}}
        g["2"]["inputs"]["model"] = ["30", 0]
    if int8vae:
        g["4"]["inputs"]["vae_name"] = "minimax_h3_video_vae_int8_convrot.safetensors"
    g["13"]["inputs"]["filename_prefix"] = f"abtest/{name}"
    return g
jobs = [(8188, "V-int8vae-10s", wf("V-int8vae-10s", 10, int8vae=True)), (8188, "L4-turbo-4s", wf("L4-turbo-4s", 4, lora=True)),
        (8189, "B20-20s", wf("B20-20s", 20)), (8189, "L8-turbo-8s", wf("L8-turbo-8s", 8, lora=True))]
t0 = time.time(); ids = [(port, name, api(port, "/prompt", {"prompt": g})["prompt_id"]) for port, name, g in jobs]
done = {}
while len(done) < len(ids) and time.time() - t0 < 1500:
    m = {l.split(":")[0]: int(l.split()[1]) for l in open("/proc/meminfo")}
    if m["MemAvailable"] < 1 << 20:
        for p in (8188, 8189): api(p, "/interrupt", {}); api(p, "/queue", {"clear": True})
        print("ABORT low RAM"); break
    for port, name, pid in ids:
        if pid in done: continue
        h = api(port, f"/history/{pid}")
        if pid in h and h[pid]["status"].get("status_str"):
            ms = h[pid]["status"]["messages"]
            a = next((x[1]["timestamp"] for x in ms if x[0] == "execution_start"), 0)
            b = next((x[1]["timestamp"] for x in ms if x[0] in ("execution_success", "execution_error")), 0)
            err = next((x[1].get("exception_message", "")[:200] for x in ms if x[0] == "execution_error"), "")
            done[pid] = (name, h[pid]["status"]["status_str"], round((b - a) / 1000, 1), err)
            print(done[pid], flush=True)
    time.sleep(3)
print(f"wall={time.time()-t0:.0f}s")

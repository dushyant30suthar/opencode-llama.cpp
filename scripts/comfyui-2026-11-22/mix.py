import json, os, subprocess, sys, time, urllib.request
S = sys.argv[1]; ROOT = os.path.expanduser("~/Projects/comfyui")
os.environ["H3_FL2VA"] = "minimax_h3_fl2va_pruned_w6a8.safetensors"
sys.path.insert(0, os.path.join(ROOT, "bench")); import matrix
def api(port, path, payload=None):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=json.dumps(payload).encode() if payload else None,
                                 headers={"Content-Type": "application/json"} if payload else {})
    return json.load(urllib.request.urlopen(req, timeout=10))
def mem():
    m = {l.split(":")[0]: int(l.split()[1]) for l in open("/proc/meminfo")}
    return m["MemAvailable"] / 1048576, (m["SwapTotal"] - m["SwapFree"]) / 1048576
def h3():
    wf = matrix.build_workflow(**dict(matrix.PROBE, width=960, height=544, length=124, steps=10, sampler="res_multistep",
        scheduler="simple", cfg=1.0, prompt=open(os.path.join(S, "dog-prompt.txt")).read(), seed=11, cache=None,
        cache_threshold=0.4, tiled_decode=0, chunk_heads=0, solattn=None, first_frame=None, last_frame=None,
        refs=None, ref_videos=None, ref_size="match"))
    wf["13"]["inputs"]["filename_prefix"] = "mixtest/h3"
    return wf
def qwen(seed, tag):
    wf = json.load(open(os.path.join(S, "qwen21.json")))
    wf["10"]["inputs"]["noise_seed"] = seed; wf["13"]["inputs"]["filename_prefix"] = f"mixtest/{tag}"
    return wf
def run(phase, jobs):
    _, swap0 = mem(); t0 = time.time(); ids = [(port, label, api(port, "/prompt", {"prompt": wf})["prompt_id"]) for port, label, wf in jobs]
    done = {}; min_avail = 99; peak = 0
    while len(done) < len(ids):
        a, s = mem(); min_avail = min(min_avail, a); peak = max(peak, s - swap0)
        if a < 1.0 or s - swap0 > 6:
            for p in (8188, 8189): api(p, "/interrupt", {}); api(p, "/queue", {"clear": True})
            print(phase, "ABORTED low memory", a, s - swap0); return
        for port, label, pid in ids:
            if pid in done: continue
            h = api(port, f"/history/{pid}")
            if pid in h and h[pid]["status"].get("status_str"):
                ms = h[pid]["status"]["messages"]
                b = next((m[1]["timestamp"] for m in ms if m[0] in ("execution_success", "execution_error")), 0) / 1000
                done[pid] = (label, h[pid]["status"]["status_str"], round(b - t0, 1))
        time.sleep(2)
    print(phase, f"wall={time.time()-t0:.0f}s min_avail={min_avail:.1f}G swap+={peak:.1f}G")
    for v in done.values(): print("   ", v)
run("A: H3 gpu0 + Qwen gpu1", [(8188, "h3", h3())] + [(8189, f"qwen{i}", qwen(100 + i, f"qwen-gpu1-{i}")) for i in range(3)])
run("B: Qwen on both", [(8188, f"q0-{i}", qwen(200 + i, f"qwen-gpu0-{i}")) for i in range(3)] + [(8189, f"q1-{i}", qwen(300 + i, f"qwen-gpu1b-{i}")) for i in range(3)])

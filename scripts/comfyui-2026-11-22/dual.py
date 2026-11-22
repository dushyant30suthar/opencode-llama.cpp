import json, os, subprocess, sys, time, urllib.request
ROOT = os.path.expanduser("~/Projects/comfyui"); S = sys.argv[1]
os.environ["H3_FL2VA"] = "minimax_h3_fl2va_pruned_w6a8.safetensors"
sys.path.insert(0, os.path.join(ROOT, "bench")); import matrix
FLAGS = ["--use-sage-attention", "--fast", "--disable-product-tips", "--fast-disk", "--disable-smart-memory", "--fp16-intermediates"]
PROMPT = open(os.path.join(S, "dog-prompt.txt")).read().strip()
def mem():
    m = {l.split(":")[0]: int(l.split()[1]) for l in open("/proc/meminfo")}
    return m["MemAvailable"] / 1048576, (m["SwapTotal"] - m["SwapFree"]) / 1048576
def api(port, path, payload=None):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=json.dumps(payload).encode() if payload else None,
                                 headers={"Content-Type": "application/json"} if payload else {})
    return json.load(urllib.request.urlopen(req, timeout=10))
servers = {}
for dev, port in ((0, 8196), (1, 8197)):
    log = open(os.path.join(S, f"dual-{dev}.log"), "w")
    servers[port] = subprocess.Popen([os.path.join(ROOT, ".venv/bin/python"), "main.py", "--port", str(port), "--cuda-device", str(dev)] + FLAGS,
                                     cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
def killall(why):
    print("KILL:", why, flush=True)
    for p in servers.values():
        try: os.killpg(p.pid, 15)
        except ProcessLookupError: pass
_, swap0 = mem()
for port in servers:
    for _ in range(300):
        try: api(port, "/system_stats"); break
        except Exception: time.sleep(1)
ids = {}
for i, port in enumerate(servers):
    wf = matrix.build_workflow(**dict(matrix.PROBE, width=960, height=544, length=124, steps=10, sampler="res_multistep",
                               scheduler="simple", cfg=1.0, prompt=PROMPT, seed=42 + i, cache=None, cache_threshold=0.4,
                               tiled_decode=0, chunk_heads=0, solattn=None, first_frame=None, last_frame=None,
                               refs=None, ref_videos=None, ref_size="match"))
    wf["13"]["inputs"]["filename_prefix"] = f"abtest/dual-gpu{i}"
    ids[port] = api(port, "/prompt", {"prompt": wf})["prompt_id"]
t0 = time.time(); done = {}; peak_swap = 0; min_avail = 99
while len(done) < 2:
    avail, swap = mem(); peak_swap = max(peak_swap, swap - swap0); min_avail = min(min_avail, avail)
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.replace("\n", " | ")
    print(f"{time.time()-t0:6.0f}s avail={avail:5.1f}G swap+={swap-swap0:5.1f}G gpu={gpu}", flush=True)
    if avail < 1.0 or swap - swap0 > 6.0:
        killall(f"avail={avail:.1f}G swap+={swap-swap0:.1f}G"); break
    for port, pid in ids.items():
        if port in done: continue
        try: h = api(port, f"/history/{pid}")
        except Exception: continue
        if pid in h and h[pid]["status"].get("status_str"):
            ms = h[pid]["status"]["messages"]
            a = next((m[1]["timestamp"] for m in ms if m[0] == "execution_start"), 0)
            b = next((m[1]["timestamp"] for m in ms if m[0] in ("execution_success", "execution_error")), 0)
            done[port] = (h[pid]["status"]["status_str"], (b - a) / 1000)
    time.sleep(5)
print("RESULT", done, f"wall={time.time()-t0:.0f}s min_avail={min_avail:.1f}G peak_swap+={peak_swap:.1f}G", flush=True)
killall("done")

import json, os, sys, time, urllib.request
S = sys.argv[1]; ROOT = os.path.expanduser("~/Projects/comfyui")
sys.path.insert(0, os.path.join(ROOT, "bench")); import matrix
P = ("Claire, a charming woman in her late 20s with blonde feathered 1980s hair, a white silk blouse and pearl earrings; Michael, a man in his early 30s "
     "with dark neatly styled hair, a grey tailored suit, white shirt and tie; and Susan, a woman in her early 30s with brunette hair and an emerald-green dress, "
     "sit together at a candlelit table in an elegant 1980s restaurant with mahogany walls, cream leather booths and warm golden sconces. "
     "Claire, Michael and Susan lean naturally toward the centre of the table, raising their champagne glasses. Michael smiles confidently and says: "
     "\"To unforgettable evenings!\" The glasses gently meet with a soft clink. Claire smiles warmly. Susan raises one eyebrow at Michael, teasing. "
     "Camera: medium three-shot at eye level, slightly handheld. Timing: the toast line comes in the first half, the clink at the middle, Susan's raised eyebrow "
     "at the end. Style: authentic 1980s 35mm Hollywood romantic comedy, warm tungsten light, film grain, soft halation. "
     "Audio: Michael's line, a soft clink of crystal, quiet restaurant chatter and soft piano.")
def api(port, path, payload=None):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=json.dumps(payload).encode() if payload else None,
                                 headers={"Content-Type": "application/json"} if payload else {})
    return json.load(urllib.request.urlopen(req, timeout=30))
def wf(name, lora, prefix=""):
    g = matrix.build_workflow(**dict(matrix.PROBE, width=704, height=1280, length=124, steps=20, sampler="res_multistep", scheduler="simple",
        cfg=1.0, prompt=prefix + P, seed=51, cache=None, cache_threshold=0.4, tiled_decode=0, chunk_heads=0, solattn=None,
        first_frame="toast_first_frame.png", last_frame=None, refs=None, ref_videos=None, ref_size="match"))
    g["90"] = {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["1", 0], "lora_name": lora, "strength_model": 1.0}}
    g["2"]["inputs"]["model"] = ["90", 0]
    g["13"]["inputs"]["filename_prefix"] = f"toast/video-{name}"
    return g
jobs = [(8188, "fantasy80s", wf("fantasy80s", "1980s_Fantasy_Movies_Lora_h3.safetensors")),
        (8189, "nostalgic", wf("nostalgic", "Nostalgic_Cinema_h3.safetensors", "r3tr0sks, 1980s cinematic scene, color film. "))]
t0 = time.time(); ids = []
for port, name, g in jobs:
    r = api(port, "/prompt", {"prompt": g}); print(name, r.get("node_errors") or "queued", flush=True); ids.append((port, name, r["prompt_id"]))
for port, name, pid in ids:
    while True:
        h = api(port, f"/history/{pid}")
        if pid in h and h[pid]["status"].get("status_str"):
            ms = h[pid]["status"]["messages"]; err = next((x[1].get("exception_message", "")[:300] for x in ms if x[0] == "execution_error"), "")
            print(name, h[pid]["status"]["status_str"], f"{time.time()-t0:.0f}s", err, flush=True); break
        time.sleep(5)

import json, sys, time, urllib.request
S = sys.argv[1]
P = ("A medium three-shot at eye level inside an elegant 1980s American restaurant at night: rich mahogany wood-paneled walls, cream leather booth, "
     "warm golden tungsten wall sconces, ivory tablecloth, crystal glassware, candles and a small floral arrangement. Three friends sit around the table, "
     "leaning in toward the centre and raising crystal champagne glasses for a toast. On the left, Claire: a charming woman in her late 20s with blonde "
     "feathered 1980s hair, a white silk blouse and pearl earrings, warm smile, expressive eyes. In the middle, Michael: a man in his early 30s with dark "
     "neatly styled hair, a grey tailored suit, white shirt and tie, smiling a little too confidently. On the right, Susan: a woman in her early 30s with "
     "brunette 1980s hair and an emerald-green dress, one eyebrow slightly raised, playful. Authentic 1980s 35mm Hollywood romantic comedy film frame, "
     "shot on Kodak 35mm motion picture film, clearly visible organic film grain, warm tungsten light, amber highlights, deep brown shadows, gentle halation, "
     "soft optical contrast, shallow depth of field, creamy bokeh. Vertical 9:16 cinematic composition.")
def api(port, path, payload=None):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=json.dumps(payload).encode() if payload else None,
                                 headers={"Content-Type": "application/json"} if payload else {})
    return json.load(urllib.request.urlopen(req, timeout=30))
ids = []
for i, seed in enumerate((51, 52, 53, 54)):
    g = json.load(open(f"{S}/qwen21.json"))
    g["1"]["inputs"]["unet_name"] = "qwen_image_2.1_int8_convrot.safetensors"
    g["4"]["inputs"].update({"prompt": P, "negative_prompt": "blurry, modern digital look, oversharpening, HDR, plastic skin, CGI, distorted faces, extra people, extra fingers, text, watermark"})
    g["6"]["inputs"]["cfg"] = 3.0
    g["7"]["inputs"].update(width=720, height=1280)
    g["9"] = {"class_type": "BasicScheduler", "inputs": {"model": ["5", 0], "scheduler": "simple", "steps": 25, "denoise": 1.0}}
    g["10"]["inputs"]["noise_seed"] = seed
    g["30"] = {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["1", 0], "lora_name": "80sfy_v3_qwen.safetensors", "strength_model": 1.0}}
    g["31"] = {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["30", 0], "lora_name": "grainscape_ultrareal_qwen21.safetensors", "strength_model": 0.7}}
    g["5"]["inputs"]["model"] = ["31", 0]
    g["13"]["inputs"]["filename_prefix"] = f"toast/frame-seed{seed}"
    port = 8188 if i % 2 == 0 else 8189
    ids.append((port, api(port, "/prompt", {"prompt": g})["prompt_id"]))
for port, pid in ids:
    while True:
        h = api(port, f"/history/{pid}")
        if pid in h and h[pid]["status"].get("status_str"): print(h[pid]["status"]["status_str"]); break
        time.sleep(2)

import json, sys, time, urllib.request
S = sys.argv[1]
CORE = ("An entirely empty, beautifully designed 1990s American restaurant interior. No people, no humans, no characters, no silhouettes, no human reflections. "
        "Rich mahogany wood-paneled walls, elegant cream leather booth seating, polished wooden furniture, warm golden tungsten wall sconces, ivory tablecloths, "
        "crystal glassware, subtle decorative arrangements, and sophisticated architectural detailing. An intimate, luxurious late-night atmosphere. "
        "Authentic 1990s 35mm Hollywood film cinematography, shot on Kodak Vision 500T 35mm motion picture film, visibly pronounced organic analog film grain throughout, "
        "distinct silver-halide grain structure. Warm tungsten illumination, amber highlights, deep chocolate-brown shadows, muted emerald greens, creamy highlights, "
        "naturally imperfect exposure, gentle highlight halation, subtle lens diffusion, soft optical contrast, beautiful light falloff. Vintage Panavision spherical lens, "
        "{lens}, f/2.0, natural optical softness, shallow depth of field, creamy background bokeh. Looks like a frame from an actual 1990s Hollywood romantic comedy "
        "shot on 35mm film, 16:9 cinematic composition. {shot}")
NEG = ("people, humans, faces, bodies, background extras, silhouettes, human reflections, modern digital sharpness, digitally clean surfaces, noise reduction, "
       "waxy textures, artificial film overlays, VHS effects, scratches, dust, digital HDR, teal-orange grading, CGI appearance, blurry")
SHOTS = [("01-wide", "50mm", "Wide establishing shot from the entrance, looking down the length of the dining room past rows of booths."),
         ("02-booth", "85mm", "Medium shot of an empty cream leather booth with a set table, a single candle and two crystal wine glasses."),
         ("03-table-detail", "85mm", "Close detail of an ivory tablecloth with crystal glassware, polished silverware and a small floral arrangement, sconces glowing behind."),
         ("04-bar", "50mm", "The mahogany bar with brass rails, empty leather stools and backlit bottles on mirrored shelves."),
         ("05-corner", "50mm", "A quiet corner of the dining room with a round table, emerald velvet drapes and a framed oil painting above a sconce."),
         ("06-dolly", "50mm", "Low eye-level view along a row of booths receding into soft amber bokeh, as if from a slow dolly move.")]
def api(port, path, payload=None):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=json.dumps(payload).encode() if payload else None,
                                 headers={"Content-Type": "application/json"} if payload else {})
    return json.load(urllib.request.urlopen(req, timeout=30))
ids = []; t0 = time.time()
for i, (tag, lens, shot) in enumerate(SHOTS):
    g = json.load(open(f"{S}/qwen21.json"))
    g["4"]["inputs"].update({"prompt": CORE.format(lens=f"{lens} lens", shot=shot), "negative_prompt": NEG})
    g["7"]["inputs"].update(width=1664, height=928)
    g["10"]["inputs"]["noise_seed"] = 900 + i
    g["30"] = {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["1", 0], "lora_name": "80sfy_v3_qwen.safetensors", "strength_model": 0.8}}
    g["31"] = {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["30", 0], "lora_name": "grainscape_ultrareal_qwen21.safetensors", "strength_model": 0.6}}
    g["5"]["inputs"]["model"] = ["31", 0]
    g["13"]["inputs"]["filename_prefix"] = f"restaurant/raw/{tag}"
    port = 8188 if i % 2 == 0 else 8189
    ids.append((port, tag, api(port, "/prompt", {"prompt": g})["prompt_id"]))
for port, tag, pid in ids:
    while True:
        h = api(port, f"/history/{pid}")
        if pid in h and h[pid]["status"].get("status_str"):
            print(tag, h[pid]["status"]["status_str"], flush=True); break
        time.sleep(2)
print(f"wall={time.time()-t0:.0f}s")

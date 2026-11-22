import json, sys, time, urllib.request
S = sys.argv[1]
SET = ("Inside an elegant 1980s American restaurant at night: rich mahogany wood-paneled walls, cream leather booth seating, polished wooden furniture, "
       "warm golden tungsten wall sconces, ivory tablecloths, crystal glassware, candles and small floral arrangements, emerald velvet accents. ")
BLONDE = ("a beautiful young woman with long voluminous champagne-blonde feathered 1980s blowout hair, light hazel-green eyes, glossy nude-pink lipstick, "
          "sparkling stud earrings and a crisp white collared shirt")
LOOK = ("Authentic 1980s 35mm Hollywood film cinematography, shot on Kodak 5247 35mm motion picture film, clearly visible organic 35mm film grain throughout, "
        "warm tungsten light, amber highlights, deep chocolate-brown shadows, muted emerald greens, creamy highlights, gentle halation, soft optical contrast, "
        "vintage spherical lens, shallow depth of field, creamy bokeh. 1980s hairstyles, 1980s clothing and makeup. Looks like a frame from a real 1980s "
        "Hollywood romantic comedy, not a modern photo with a filter. 16:9 cinematic composition.")
SCENES = [
 ("01-booth-wide", f"Four friends having dinner together in a cream leather booth: {BLONDE} laughing in the middle, a man in a navy double-breasted blazer, "
                   "a brunette woman with big curled hair in a red silk blouse, and a man with a moustache in a tweed jacket. Plates of food, wine glasses, candlelight. Wide shot."),
 ("02-toast", f"Three friends at a round table raising crystal champagne glasses in a toast: {BLONDE}, a Black man in a grey suit with a skinny tie, "
              "and a red-haired woman in a teal shoulder-padded dress, all smiling. Medium shot across the table."),
 ("03-conversation", f"Over-the-shoulder medium shot: {BLONDE} listens and smiles across the table at two friends, a dark-haired man in a cream linen jacket "
                     "and a woman with a short permed bob, dinner plates and a bottle of red wine between them."),
 ("04-laughing", f"Candid moment, four friends laughing at a dinner table in the booth, one man telling a story with his hands, {BLONDE} covering her mouth "
                 "as she laughs, glasses of wine and half-finished desserts. Medium-wide shot."),
 ("05-waiter", f"A waiter in a white jacket pours wine for three diners at a table by the mahogany wall: {BLONDE}, an older man with silver hair in a dark suit, "
               "and a woman in a sequined black dress with big earrings. Medium-wide shot, sconces glowing behind."),
 ("06-dolly", f"Low eye-level shot along the row of booths: in the foreground booth four friends share dinner and talk, {BLONDE} turned toward the man beside her, "
              "other booths softly out of focus in amber bokeh, as if from a slow dolly move."),
]
def api(port, path, payload=None):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=json.dumps(payload).encode() if payload else None,
                                 headers={"Content-Type": "application/json"} if payload else {})
    return json.load(urllib.request.urlopen(req, timeout=30))
ids = []; t0 = time.time()
for i, (tag, scene) in enumerate(SCENES):
    g = json.load(open(f"{S}/qwen21.json"))
    g["1"]["inputs"]["unet_name"] = "qwen_image_2.1_int8_convrot.safetensors"
    g["4"]["inputs"].update({"prompt": SET + scene + " " + LOOK,
                             "negative_prompt": "blurry, modern digital look, oversharpening, HDR, teal-orange grading, plastic skin, CGI, distorted faces, extra fingers, text, watermark"})
    g["6"]["inputs"]["cfg"] = 3.0
    g["7"]["inputs"].update(width=1664, height=928)
    g["9"] = {"class_type": "BasicScheduler", "inputs": {"model": ["5", 0], "scheduler": "simple", "steps": 25, "denoise": 1.0}}
    g["10"]["inputs"]["noise_seed"] = 4100 + i
    g["30"] = {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["1", 0], "lora_name": "80sfy_v3_qwen.safetensors", "strength_model": 1.0}}
    g["31"] = {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["30", 0], "lora_name": "grainscape_ultrareal_qwen21.safetensors", "strength_model": 0.7}}
    g["5"]["inputs"]["model"] = ["31", 0]
    g["13"]["inputs"]["filename_prefix"] = f"dinner80s/{tag}"
    port = 8188 if i % 2 == 0 else 8189
    ids.append((port, tag, api(port, "/prompt", {"prompt": g})["prompt_id"]))
for port, tag, pid in ids:
    while True:
        h = api(port, f"/history/{pid}")
        if pid in h and h[pid]["status"].get("status_str"):
            print(tag, h[pid]["status"]["status_str"], flush=True); break
        time.sleep(3)
print(f"wall={time.time()-t0:.0f}s")

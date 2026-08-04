#!/usr/bin/env python3
"""Generate clips back-to-back with the best measured config, forever.

    .venv/bin/python bench/clips.py                  # rolling, uses tuned.json
    .venv/bin/python bench/clips.py --count 5
    .venv/bin/python bench/clips.py --length 362     # 15s instead of the tuned default

This is what runs once there is nothing left to optimise. It reads
bench/tuned.json (written by analyse.py from the measured winners), starts a
server with the tuned launch flags, and then works through a prompt list,
varying seed and prompt each time so the outputs are actually different rather
than the same clip regenerated.

Outputs land in ComfyUI's output/video/. A manifest line per clip goes to
bench/clips.jsonl so there is a record of which prompt and seed produced which
file, and how long it took (server-measured, same rule as the sweep).
"""
import argparse, json, os, subprocess, sys, time, urllib.request, urllib.error

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import matrix  # noqa: E402

PY = os.path.join(ROOT, ".venv", "bin", "python")
PORT = 8192
BASE = f"http://127.0.0.1:{PORT}"
TUNED = os.path.join(HERE, "tuned.json")
MANIFEST = os.path.join(HERE, "clips.jsonl")
LOG = os.path.join(HERE, "clips-server.log")

# Written to H3's documented prompt order:
#   Subject -> Scene -> Action -> Camera -> Timing -> Visual Style -> Audio
# The model follows instructions closely, so vague prompts give vague results —
# the gap between a mediocre and a good generation is mostly prompt structure,
# not sampler settings. Each names an explicit camera move and a timing beat,
# because 15s is long enough that an unspecified clip drifts or stalls halfway.
PROMPTS = [
    "A lone fisherman in a wooden skiff. A glassy fjord at dawn, mist lying on the water, "
    "black cliffs rising on both sides. He casts a line; ripples spread and the boat rocks "
    "gently. Camera: slow aerial descent from high above, settling to water level behind the "
    "boat. Timing: mist parts halfway through to reveal sunlight striking the far cliff. "
    "Style: cinematic, muted teal and amber, anamorphic, shallow depth of field. "
    "Audio: lapping water, a distant gull, the whir of the reel.",

    "A snow leopard. A high Himalayan ridge in late afternoon, wind-scoured rock and drifting "
    "snow. It picks its way along the ridgeline, then pauses and turns its head toward camera. "
    "Camera: long lens tracking shot moving parallel with the animal, slight handheld float. "
    "Timing: it stops and looks up at the two-thirds mark. Style: nature documentary, crisp "
    "low sun, cold blue shadows against warm gold fur. Audio: high wind, crunching snow, "
    "a low breath.",

    "A glassblower. A dim workshop lit only by the furnace mouth, tools racked on the wall. "
    "She turns the pipe steadily and the molten gather glows and stretches. Camera: slow push "
    "in from medium to close-up on the glowing glass, rack focus to her face. Timing: she "
    "blows into the pipe and the bubble swells near the end. Style: chiaroscuro, deep shadow, "
    "orange key light, 35mm film grain. Audio: furnace roar, the creak of the pipe, quiet "
    "breathing.",

    "A vintage steam locomotive. A snowbound mountain station at blue hour, lamps glowing along "
    "the platform. It rolls in and stops, venting steam across the boards. Camera: low static "
    "wide, then a slow tilt up the engine as it comes to rest. Timing: the steam release hits "
    "at the midpoint and fills the frame. Style: cinematic, cold blue ambient against warm "
    "tungsten lamps, volumetric light. Audio: hissing steam, squealing brakes, a distant "
    "whistle.",

    "A field of tall grass and a lone oak. Rolling farmland under a fast-moving summer storm, "
    "hard shafts of light breaking through. Wind moves through the grass in long waves and the "
    "oak's branches sway. Camera: slow low dolly through the grass toward the tree. Timing: a "
    "sunbeam sweeps across the field two-thirds through. Style: naturalistic, high dynamic "
    "range, golden green, soft bloom. Audio: rushing wind, rustling grass, distant thunder.",

    "A neon-lit ramen stall and its cook. A narrow Tokyo alley after rain, signage reflecting "
    "in standing water. He lifts a basket of noodles and steam billows up into the neon. "
    "Camera: slow dolly forward down the alley, ending on the counter. Timing: the steam burst "
    "lands halfway. Style: cyberpunk noir, saturated magenta and cyan, wet specular highlights, "
    "shallow focus. Audio: sizzling broth, light rain on awnings, distant traffic.",

    "An astronaut. A red dust plain under a pale alien sky, a half-buried monolith ahead. "
    "She walks toward it, boots kicking up slow dust, then stops and looks up. Camera: wide "
    "tracking shot from the side, slowly craning up to reveal the monolith's scale. Timing: "
    "the crane reveal completes in the final third. Style: hard sunlight, long shadows, "
    "desaturated rust and bone-white, 70mm epic framing. Audio: suit radio static, shallow "
    "breathing, low wind.",

    "A humpback whale and her calf. Clear tropical ocean, sunlight in moving shafts from the "
    "surface. They glide upward together through the light, calf tucked close. Camera: slow "
    "underwater tracking shot rising with them toward the surface. Timing: they break into the "
    "brightest light near the end. Style: luminous blues, caustic light patterns, natural, "
    "documentary. Audio: muffled ocean hum, distant whale song, drifting bubbles.",

    "An old bookshop and its keeper. Narrow aisles of floor-to-ceiling shelves, dust in the "
    "air, late autumn light through a mullioned window. He slides a book free and opens it, "
    "dust lifting into the beam. Camera: slow steadicam glide down the aisle, settling on his "
    "hands. Timing: the dust catches the light at the midpoint. Style: warm amber, soft "
    "diffusion, painterly, shallow depth. Audio: creaking floorboards, turning pages, faint "
    "street noise outside.",

    "A flamenco dancer. A stone courtyard at dusk, strung bulbs overhead, worn tiles underfoot. "
    "She turns sharply, skirt flaring, heels striking the stone in rhythm. Camera: low wide "
    "shot circling slowly around her. Timing: the fastest turn lands two-thirds through. "
    "Style: warm tungsten against deep blue dusk, high contrast, motion blur on fabric. "
    "Audio: percussive heel strikes, guitar, a single shout.",

    "A red fox in fresh snow. A birch forest at first light, snow still falling lightly, "
    "trunks receding into white. It moves through the snow, pauses, then pounces headfirst "
    "into a drift. Camera: low tracking shot at snow level, following alongside. Timing: the "
    "pounce lands in the final third. Style: minimal, high key, white and pale birch against "
    "vivid orange fur. Audio: soft snowfall, crunching powder, a distant crow.",

    "A lighthouse on a rock stack. A violent North Atlantic storm at night, waves exploding "
    "against the base. The beam sweeps through spray and rain in a steady rhythm. Camera: "
    "slow aerial orbit around the tower, holding at a distance. Timing: the largest wave "
    "strikes near the midpoint. Style: dramatic, near-monochrome blue-black with the warm "
    "beam cutting through, heavy atmosphere. Audio: roaring surf, howling wind, a low foghorn.",
]

DEFAULTS = dict(matrix.PROBE, length=124, steps=20, cfg=1.0)


def api(path, payload=None, timeout=30):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(BASE + path, data=data,
                                 headers={"Content-Type": "application/json"} if data else {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = r.read()
    return json.loads(body) if body else {}


def alive():
    try:
        api("/system_stats", timeout=4)
        return True
    except Exception:
        return False


def server_seconds(entry):
    start = end = None
    for event, data in entry.get("status", {}).get("messages", []):
        ts = data.get("timestamp")
        if ts is None:
            continue
        if event == "execution_start":
            start = ts
        elif event in ("execution_success", "execution_error", "execution_interrupted"):
            end = ts
    return (end - start) / 1000.0 if (start and end) else None


def load_tuned():
    """Measured winners if analyse.py has run; otherwise conservative defaults."""
    if os.path.exists(TUNED):
        with open(TUNED) as fh:
            t = json.load(fh)
        flags = t.get("launch_flags", ["--use-sage-attention"])
        params = dict(DEFAULTS, **t.get("workflow", {}))
        return flags, params, t.get("source", "tuned.json")
    return ["--use-sage-attention"], dict(DEFAULTS), "defaults (tuned.json absent)"


def start_server(flags):
    # bracket so pkill -f cannot match the shell issuing it
    subprocess.run(["pkill", "-f", f"[m]ain.py --port {PORT}"], capture_output=True)
    time.sleep(2)
    log = open(LOG, "a")
    proc = subprocess.Popen([PY, "main.py", "--port", str(PORT)] + flags,
                            cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                            start_new_session=True)
    for _ in range(300):
        if proc.poll() is not None:
            raise RuntimeError(f"server died rc={proc.returncode}, see {LOG}")
        if alive():
            return proc
        time.sleep(1)
    raise RuntimeError("server did not boot")


def generate(params, prompt, seed, index):
    wf = matrix.build_workflow(**dict(params, prompt=prompt, seed=seed))
    wf["13"]["inputs"]["filename_prefix"] = f"video/h3-clip-{index:04d}"
    t0 = time.time()
    try:
        pid = api("/prompt", {"prompt": wf}).get("prompt_id")
    except urllib.error.HTTPError as e:
        return {"ok": False, "error": e.read().decode()[:300]}
    while True:
        time.sleep(3)
        try:
            hist = api(f"/history/{pid}", timeout=20)
        except Exception:
            continue
        if pid not in hist:
            continue
        entry = hist[pid]
        ok = entry.get("status", {}).get("status_str") == "success"
        files = []
        for out in (entry.get("outputs") or {}).values():
            for kind in ("images", "video", "gifs", "audio"):
                for f in out.get(kind, []) or []:
                    files.append(os.path.join(f.get("subfolder", ""), f.get("filename", "")))
        return {"ok": ok, "server_s": server_seconds(entry),
                "wall_s": round(time.time() - t0, 1), "files": files,
                "detail": None if ok else str(entry.get("status", {}).get("messages", []))[:300]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=0, help="0 = run until stopped")
    ap.add_argument("--length", type=int, help="override frame count (124=5.2s, 362=15.1s)")
    ap.add_argument("--steps", type=int)
    ap.add_argument("--cfg", type=float)
    args = ap.parse_args()

    flags, params, source = load_tuned()
    for k in ("length", "steps", "cfg"):
        if getattr(args, k) is not None:
            params[k] = getattr(args, k)

    print(f"config from : {source}")
    print(f"launch flags: {' '.join(flags)}")
    print(f"workflow    : {params['width']}x{params['height']} len{params['length']} "
          f"steps{params['steps']} cfg{params['cfg']} {params['sampler']}/{params['scheduler']}")
    print(f"tokens      : {matrix.tokens(params['width'], params['height'], params['length']):,}")

    proc = start_server(flags)
    print(f"server up on {BASE}\n")

    made = 0
    try:
        while args.count == 0 or made < args.count:
            prompt = PROMPTS[made % len(PROMPTS)]
            seed = 1000 + made
            print(f"[{time.strftime('%H:%M:%S')}] clip {made:04d} seed={seed} :: {prompt[:58]}...")
            res = generate(params, prompt, seed, made)
            rec = dict(index=made, seed=seed, prompt=prompt, flags=flags,
                       **{k: params[k] for k in ("width", "height", "length", "steps", "cfg",
                                                 "sampler", "scheduler")},
                       ts=int(time.time()), **res)
            with open(MANIFEST, "a") as fh:
                fh.write(json.dumps(rec) + "\n")
            if res.get("ok"):
                mins = (res["server_s"] or 0) / 60
                print(f"           -> {res['files']}  {mins:.1f} min "
                      f"({(res['server_s'] or 0)/params['steps']:.1f}s/step)")
            else:
                print(f"           FAILED: {str(res.get('detail') or res.get('error'))[:150]}")
            made += 1
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        if proc.poll() is None:
            proc.terminate()
    return 0


if __name__ == "__main__":
    sys.exit(main())

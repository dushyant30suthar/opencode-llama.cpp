#!/usr/bin/env python3
"""Decisive test: can both GPUs generate independently at the same time?

The two cards have NO P2P (can_device_access_peer == False, topology PHB — every
GPU-to-GPU byte routes through the CPU host bridge). That kills every
model-sharding approach: DisTorch2 with donor=cuda:1 measured SLOWER than
donor=cpu because it pays two hops instead of one.

But independent instances need no P2P at all. The only real blocker was host
RAM: one instance measured 21.7 GB RSS against 31 GB total, so two looked
impossible. That number was taken WITHOUT --mmap-torch-files. With mmap both
processes map the same safetensors and the pages are served from a shared page
cache, so the model is counted twice in RSS but occupies memory once.

This script proves or kills that. It fires the same graph at both instances at
the same moment and watches RAM and swap throughout. If swap starts growing the
run is aborted — thrashing this box is a known, previously-recorded disaster.

Success criterion is throughput, not latency: two clips finishing in roughly the
time one used to take is a 2x win even though each individual clip is no faster.
"""
import json, sys, threading, time, urllib.request, os

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import matrix  # noqa: E402

PORTS = {0: 8194, 1: 8195}
SERVER_PIDS = [int(x) for x in os.environ.get("DUAL_PIDS", "").split() if x.isdigit()]
PARAMS = dict(matrix.PROBE, length=124, steps=10)


def api(port, path, payload=None, timeout=30):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=data,
                                 headers={"Content-Type": "application/json"} if data else {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = r.read()
    return json.loads(body) if body else {}


def mem():
    out = {}
    for line in open("/proc/meminfo"):
        k, v = line.split(":", 1)
        out[k] = int(v.split()[0]) // 1024          # MiB
    return out


def server_seconds(entry):
    start = end = None
    for event, data in entry.get("status", {}).get("messages", []):
        ts = data.get("timestamp")
        if ts is None:
            continue
        if event == "execution_start":
            start = ts
        elif event.startswith("execution_"):
            end = ts
    return (end - start) / 1000.0 if (start and end) else None


results = {}


def run(gpu, seed):
    port = PORTS[gpu]
    wf = matrix.build_workflow(**dict(PARAMS, seed=seed))
    wf["13"]["inputs"]["filename_prefix"] = f"video/dual-gpu{gpu}"
    t0 = time.time()
    pid = api(port, "/prompt", {"prompt": wf}).get("prompt_id")
    while True:
        time.sleep(3)
        try:
            hist = api(port, f"/history/{pid}", timeout=20)
        except Exception:
            continue
        if pid in hist:
            e = hist[pid]
            results[gpu] = {"ok": e.get("status", {}).get("status_str") == "success",
                            "server_s": server_seconds(e),
                            "wall_s": round(time.time() - t0, 1)}
            return


def main():
    base = mem()
    swap0 = base["SwapTotal"] - base["SwapFree"]
    print(f"baseline: MemAvailable {base['MemAvailable']//1024} GB, "
          f"cached {base['Cached']//1024} GB, swap used {swap0} MiB\n")

    for gpu, port in PORTS.items():
        try:
            api(port, "/system_stats", timeout=5)
            print(f"  instance gpu{gpu} on :{port} — up")
        except Exception:
            sys.exit(f"instance for gpu{gpu} not answering on :{port}")

    print("\nfiring both at once...")
    threads = [threading.Thread(target=run, args=(g, 100 + g)) for g in PORTS]
    t0 = time.time()
    for t in threads:
        t.start()

    worst_avail, peak_swap = base["MemAvailable"], swap0
    while any(t.is_alive() for t in threads):
        time.sleep(5)
        m = mem()
        used_swap = m["SwapTotal"] - m["SwapFree"]
        worst_avail = min(worst_avail, m["MemAvailable"])
        peak_swap = max(peak_swap, used_swap)
        if used_swap - swap0 > 2048:
            print(f"\n!! ABORT: swap grew {used_swap - swap0} MiB — thrashing. "
                  f"Two instances do not fit.")
            # kill the servers by PID, never by pattern: `pkill -f 'main.py --port'`
            # matches the shell that invoked this script and takes it out too.
            for pid in SERVER_PIDS:
                try:
                    os.kill(pid, 15)
                except Exception:
                    pass
            return 2
        print(f"  [{time.time()-t0:6.0f}s] avail {m['MemAvailable']//1024:2d} GB  "
              f"cached {m['Cached']//1024:2d} GB  swap+{used_swap - swap0:5d} MiB", flush=True)

    for t in threads:
        t.join()
    wall = time.time() - t0

    print(f"\n=== RESULT ===")
    for g in sorted(results):
        r = results[g]
        print(f"  gpu{g}: ok={r['ok']}  server {r['server_s']}s  wall {r['wall_s']}s")
    print(f"  both finished in {wall:.1f}s wall")
    print(f"  worst MemAvailable {worst_avail//1024} GB, peak swap +{peak_swap - swap0} MiB")
    single = 348.0                      # M1, best single-instance at this config
    if all(r["ok"] for r in results.values()):
        print(f"\n  throughput: 2 clips / {wall:.0f}s vs 1 clip / {single:.0f}s "
              f"=> {2*single/wall:.2f}x")
    with open(os.path.join(HERE, "dual-result.json"), "w") as fh:
        json.dump({"results": results, "wall_s": wall,
                   "worst_avail_mib": worst_avail, "peak_swap_mib": peak_swap - swap0}, fh, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())

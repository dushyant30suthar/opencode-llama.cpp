#!/usr/bin/env python3
"""Run the H3 experiment matrix and record server-measured timings.

    .venv/bin/python bench/sweep.py A            # one launch phase
    .venv/bin/python bench/sweep.py A B C        # several
    .venv/bin/python bench/sweep.py D E F        # workflow phases (one server)
    .venv/bin/python bench/sweep.py A --confirm  # at 5.2s/20 steps instead of the probe
    .venv/bin/python bench/sweep.py --list

TIMING RULE: every number here comes from ComfyUI's own execution_start /
execution_success messages, which carry server-side millisecond timestamps
(execution.py add_message). Client-side stream timing is never used — that is
the mistake that inflated the exllamav3 decode numbers by ~1.7x and rigged a
whole comparison. Wall-clock is recorded separately and only as a sanity check;
when the two disagree, the server is right.

Results append to bench/results.jsonl, one object per run. Nothing is
overwritten, so an interrupted sweep resumes by just re-running.
"""
import argparse, json, os, signal, subprocess, sys, time, urllib.request, urllib.error

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import matrix  # noqa: E402

PY = os.path.join(ROOT, ".venv", "bin", "python")
PORT = 8191                      # deliberately not 8188: never fight the panel's server
BASE = f"http://127.0.0.1:{PORT}"
RESULTS = os.path.join(HERE, "results.jsonl")
LOG = os.path.join(HERE, "sweep-server.log")
BOOT_TIMEOUT = 300
RUN_TIMEOUT = 5400


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


def vram():
    try:
        return {d["index"]: (d["vram_total"] - d["vram_free"]) / 1024**3
                for d in api("/system_stats", timeout=4).get("devices", [])
                if d.get("type") == "cuda"}
    except Exception:
        return {}


class Server:
    """One ComfyUI process per launch config. Restarting is the only way to
    change an attention backend or a memory flag — they are read at import."""

    def __init__(self, flags):
        self.flags = list(flags)
        self.proc = None

    def __enter__(self):
        stop_any()
        log = open(LOG, "a")
        log.write(f"\n\n{'='*70}\nlaunch: {' '.join(self.flags) or '(defaults)'}\n{'='*70}\n")
        log.flush()
        self.proc = subprocess.Popen(
            [PY, "main.py", "--port", str(PORT)] + self.flags,
            cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        deadline = time.time() + BOOT_TIMEOUT
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"server exited rc={self.proc.returncode}; see {LOG}")
            if alive():
                return self
            time.sleep(1)
        raise RuntimeError(f"server did not boot in {BOOT_TIMEOUT}s; see {LOG}")

    def __exit__(self, *exc):
        if self.proc and self.proc.poll() is None:
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
            except Exception:
                self.proc.terminate()
            try:
                self.proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
                except Exception:
                    self.proc.kill()
        # the CUDA context takes a moment to hand VRAM back
        for _ in range(30):
            if not alive():
                break
            time.sleep(1)
        time.sleep(3)


def stop_any():
    subprocess.run(["pkill", "-f", f"main.py --port {PORT}"], capture_output=True)
    time.sleep(2)


def server_seconds(entry):
    """execution_start -> execution_success, from the server's own clock."""
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


def run_one(params, meta):
    """Queue one graph, wait it out, return a result record."""
    wf = matrix.build_workflow(**params)
    ntok = matrix.tokens(params["width"], params["height"], params["length"])
    rec = dict(meta)
    rec.update({k: params[k] for k in
                ("width", "height", "length", "steps", "cfg", "sampler", "scheduler",
                 "shift_video", "shift_audio", "model_device", "clip_device", "vae_device")})
    rec["tokens"] = ntok

    before = vram()
    t0 = time.time()
    try:
        pid = api("/prompt", {"prompt": wf}).get("prompt_id")
    except urllib.error.HTTPError as e:
        rec.update(ok=False, error="validation", detail=e.read().decode()[:400])
        return rec

    peak = dict(before)
    deadline = time.time() + RUN_TIMEOUT
    while time.time() < deadline:
        time.sleep(2)
        for k, v in vram().items():
            peak[k] = max(peak.get(k, 0), v)
        try:
            hist = api(f"/history/{pid}", timeout=20)
        except Exception:
            continue                      # a busy server can drop a poll; not fatal
        if pid not in hist:
            continue
        entry = hist[pid]
        status = entry.get("status", {})
        ok = status.get("status_str") == "success"
        secs = server_seconds(entry)
        rec.update({
            "ok": ok,
            "server_s": secs,
            "wall_s": round(time.time() - t0, 1),
            "s_per_step": round(secs / params["steps"], 3) if (secs and ok) else None,
            "peak_vram": {f"cuda:{k}": round(v, 2) for k, v in sorted(peak.items())},
        })
        if not ok:
            msgs = [str(m) for m in status.get("messages", [])]
            oom = any("out of memory" in m.lower() or "OutOfMemory" in m for m in msgs)
            rec["error"] = "oom" if oom else "failed"
            rec["detail"] = " | ".join(msgs[-3:])[:400]
        return rec

    rec.update(ok=False, error="timeout", wall_s=round(time.time() - t0, 1))
    return rec


def have(module):
    if not module:
        return True
    return subprocess.run([PY, "-c", f"import {module}"], capture_output=True).returncode == 0


def emit(rec):
    with open(RESULTS, "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    tag = f"{rec['phase']}/{rec['row']}"
    if rec.get("ok"):
        print(f"  {tag:10} {rec['server_s']:>8.1f}s server   {rec['s_per_step']:>7.3f} s/step   "
              f"peak {rec.get('peak_vram')}")
    else:
        print(f"  {tag:10} {rec.get('error','?').upper():>9}   {str(rec.get('detail',''))[:90]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phases", nargs="*", help="phase letters, e.g. A B C")
    ap.add_argument("--confirm", action="store_true",
                    help="use the 5.2s/20-step config instead of the fast probe")
    ap.add_argument("--base-flags", default="",
                    help='launch flags held fixed for all servers, one string: --base-flags="--use-sage-attention --disable-cuda-malloc"')
    ap.add_argument("--length", type=int, help="override frame count")
    ap.add_argument("--steps", type=int, help="override step count")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list or not args.phases:
        subprocess.run([PY, os.path.join(HERE, "matrix.py")])
        return 0

    base = dict(matrix.CONFIRM if args.confirm else matrix.PROBE)
    stage = "confirm" if args.confirm else "probe"
    if args.length is not None:
        base["length"] = args.length
        stage += f"-len{args.length}"
    if args.steps is not None:
        base["steps"] = args.steps
        stage += f"-s{args.steps}"
    print(f"stage={stage}  base={ {k: base[k] for k in ('width','height','length','steps','cfg')} }")
    print(f"tokens={matrix.tokens(base['width'], base['height'], base['length']):,}")
    print(f"results -> {RESULTS}\n")

    args.base_flags = args.base_flags.split() if isinstance(args.base_flags, str) else args.base_flags
    launch_names = {n for n, _, _ in matrix.LAUNCH_PHASES}

    for letter in args.phases:
        try:
            name, desc, rows = matrix.phase(letter)
        except KeyError:
            print(f"unknown phase {letter}")
            continue
        print(f"\n### phase {name}: {desc}")

        if name in launch_names:
            for row, flags, needs, note in rows:
                if not have(needs):
                    print(f"  {name}/{row:8} SKIP — python module '{needs}' not installed "
                          f"(run bench/install-accel.sh)")
                    continue
                # Row flags ride ON TOP of the base. Phase A passes no base (it
                # IS the attention experiment), but every later launch phase
                # must inherit A's winner — ranking --fast or an offload flag
                # against a slower attention backend measures the wrong
                # baseline and can invert the ordering.
                combined = list(args.base_flags) + list(flags)
                print(f"  -- {row}: {' '.join(combined) or '(defaults)'}  {note}")
                meta = dict(phase=name, row=row, stage=stage, flags=combined, note=note,
                            ts=int(time.time()))
                try:
                    with Server(combined):
                        emit(run_one(base, meta))
                except Exception as e:
                    emit({**meta, "ok": False, "error": "launch", "detail": str(e)[:300]})
        else:
            # One server for the whole phase — these knobs are per-graph.
            print(f"  (one server, base flags: {' '.join(args.base_flags) or 'defaults'})")
            try:
                with Server(args.base_flags):
                    for row, overrides, note in rows:
                        params = dict(base, **overrides)
                        meta = dict(phase=name, row=row, stage=stage,
                                    flags=args.base_flags, note=note, ts=int(time.time()))
                        emit(run_one(params, meta))
            except Exception as e:
                print(f"  launch failed: {e}")

    print(f"\ndone — {RESULTS}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        stop_any()
        print("\ninterrupted; server stopped")
        sys.exit(130)

#!/usr/bin/env python3
"""Minimal vLLM control daemon.

vLLM is one model per process and /v1/models can only report the loaded one, so
a remote opencode session cannot see or switch models. llama-server does this
natively (it scans models-dir and swaps on demand); this is the smallest thing
that gives vLLM the same three capabilities:

    GET  /models   -> every YAML in models-dir, with its real max-model-len
    GET  /status   -> which one is served right now, if any
    POST /start    -> {"id": "<yaml basename>"}  launch that config
    POST /stop     -> unload, free the GPUs

Deliberately NOT a proxy: inference still goes straight to :8000. This only
starts and stops things, which is all that was asked for.
"""
import json, os, re, subprocess, threading, time, urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HOME = os.path.expanduser("~")
MODELS_DIR = f"{HOME}/.config/opencode/providers/vllm/models"
LAUNCH = f"{HOME}/Projects/vllm/launch.sh"
LOG = f"{HOME}/Projects/vllm/server.log"
PORT = 8900
VLLM = "http://127.0.0.1:8000"


def yamls():
    out = []
    if not os.path.isdir(MODELS_DIR):
        return out
    for fn in sorted(os.listdir(MODELS_DIR)):
        if not fn.endswith(".yaml"):
            continue
        path = os.path.join(MODELS_DIR, fn)
        try:
            text = open(path).read()
        except OSError:
            continue
        def field(key, default=None):
            m = re.search(rf"^{key}:\s*(.+?)\s*$", text, re.M)
            return m.group(1) if m else default
        ctx = field("max-model-len")
        out.append({
            "id": fn[:-5],
            "file": path,
            "model": field("model"),
            "served": field("served-model-name") or fn[:-5],
            "context": int(ctx) if ctx and ctx.isdigit() else None,
        })
    return out


def served_now():
    try:
        with urllib.request.urlopen(VLLM + "/v1/models", timeout=3) as r:
            data = json.load(r).get("data") or []
            return data[0].get("id") if data else None
    except Exception:
        return None


def running_pid():
    p = subprocess.run(["pgrep", "-f", "venv/bin/[v]llm"], capture_output=True, text=True)
    pids = [x for x in p.stdout.split() if x.strip()]
    return int(pids[0]) if pids else None


def stop(timeout=90):
    # ONLY the vllmrun session. Never `tmux kill-server` - this daemon runs in a
    # tmux session too, and killing the server kills the daemon mid-request.
    subprocess.run(["tmux", "kill-session", "-t", "vllmrun"], capture_output=True)
    pid = running_pid()
    if pid:
        subprocess.run(["kill", "-TERM", str(pid)], capture_output=True)
    deadline = time.time() + timeout
    while time.time() < deadline:
        if running_pid() is None:
            return True
        time.sleep(2)
    pid = running_pid()
    if pid:
        subprocess.run(["kill", "-KILL", str(pid)], capture_output=True)
        time.sleep(3)
    return running_pid() is None


def start(entry):
    stop()
    open(LOG, "w").close()
    subprocess.run(
        ["tmux", "new-session", "-d", "-s", "vllmrun", f"{LAUNCH} {entry['model']} {entry['file']}"],
        capture_output=True,
    )
    return True


class H(BaseHTTPRequestHandler):
    def _send(self, code, payload):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path.startswith("/models"):
            return self._send(200, {"data": yamls()})
        if self.path.startswith("/status"):
            return self._send(200, {"served": served_now(), "pid": running_pid()})
        self._send(404, {"error": "not found"})

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
        except Exception:
            body = {}
        if self.path.startswith("/stop"):
            return self._send(200, {"stopped": stop()})
        if self.path.startswith("/start"):
            wanted = body.get("id")
            entries = yamls()
            entry = next((e for e in entries if e["id"] == wanted), None)
            if not entry:
                return self._send(404, {"error": f"unknown id {wanted}",
                                        "known": [e["id"] for e in entries]})
            if served_now() == entry["served"]:
                return self._send(200, {"started": True, "already": True, "served": entry["served"]})
            # Launch in a worker: loading takes minutes and the caller only
            # needs to know the swap was accepted. Progress is visible via
            # /status and the plugin's own readiness probe.
            threading.Thread(target=start, args=(entry,), daemon=True).start()
            return self._send(202, {"started": True, "id": entry["id"], "served": entry["served"]})
        self._send(404, {"error": "not found"})


if __name__ == "__main__":
    ThreadingHTTPServer(("0.0.0.0", PORT), H).serve_forever()

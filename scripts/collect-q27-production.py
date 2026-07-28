#!/usr/bin/env python3
"""
Passive collector for the live Qwen3.6-27B session.

Every MTP number in docs/mtp-27b.md was measured at ctx 32768 while production
runs 196608 — that document's Task 1. A real agentic session at ~118k depth is
the measurement that cannot be manufactured, so this records it while it happens.

Read-only: polls /slots and tails the server log. Never sends a request, never
touches the model.

Captures per sample: depth, prefill progress, decode progress, and — from the
log — draft acceptance and mean accepted length per completed request, which is
what distinguishes "MTP compressed at depth" from "the base step slowed".
"""
import json, re, subprocess, time, urllib.request, os

PORT = 45285
PID = "45285"
LOG = os.path.expanduser("~/.local/state/opencode/providers/llamacpp/server.log")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "q27-production.jsonl")

ACC = re.compile(
    r"^\[(\d+)\].*draft acceptance = ([0-9.]+) \(\s*(\d+) accepted /\s*(\d+) generated\), mean len =\s*([0-9.]+)"
)


def slot():
    with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/slots", timeout=4) as r:
        return json.load(r)[0]


def emit(row):
    with open(OUT, "a") as f:
        f.write(json.dumps(row) + "\n")


def main():
    # start at the end of the log so we only record this run's requests
    seen = os.path.getsize(LOG)
    prev = None
    while True:
        now = time.time()

        # --- slot state -------------------------------------------------
        try:
            s = slot()
            n = (s.get("next_token") or [{}])[0]
            depth = s.get("n_prompt_tokens") or 0
            proc = s.get("n_prompt_tokens_processed") or 0
            cached = s.get("n_prompt_tokens_cache") or 0
            dec = n.get("n_decoded") or 0
            if prev:
                dt = now - prev["t"]
                if dt > 0:
                    # decoding: same prompt, more tokens out
                    if dec > prev["dec"] and proc == prev["proc"]:
                        emit({"kind": "decode", "t": now, "depth": depth,
                              "tps": round((dec - prev["dec"]) / dt, 2)})
                    # prefilling: processed climbing
                    elif proc > prev["proc"]:
                        emit({"kind": "prefill", "t": now, "depth": depth,
                              "cached": cached,
                              "tps": round((proc - prev["proc"]) / dt, 1)})
            prev = {"t": now, "dec": dec, "proc": proc}
        except Exception:
            pass

        # --- completed-request stats from the log ------------------------
        try:
            size = os.path.getsize(LOG)
            if size > seen:
                with open(LOG, errors="replace") as f:
                    f.seek(seen)
                    for line in f:
                        m = ACC.match(line)
                        if m and m.group(1) == PID:
                            emit({"kind": "accept", "t": now,
                                  "depth": prev["dec"] if prev else None,
                                  "acceptance": float(m.group(2)),
                                  "accepted": int(m.group(3)),
                                  "generated": int(m.group(4)),
                                  "mean_len": float(m.group(5))})
                seen = size
            elif size < seen:
                seen = size          # log rotated
        except Exception:
            pass

        time.sleep(2)


if __name__ == "__main__":
    main()

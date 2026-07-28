#!/usr/bin/env python3
"""Bench harness for OpenAI-style endpoints (TabbyAPI/exllamav3).
Splits prefill (TTFB) from decode via streaming. Mirrors the llama.cpp sweep so
results are apples-to-apples: same corpus, same depths, same sampling.

usage: sweepx.py <base_url> <config_name> <mode> <out.jsonl>
  mode: sweep   — depth sweep (depths in DEPTHS)
        edit    — divergence test: build 60k, mutate at 49k/30k/15k/3k
        agents  — cross-job prefix sharing: A(61k) / B(shares 30k) interleave
"""
import json, sys, time, urllib.request

BASE, NAME, MODE, OUT = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
CORPUS = "/tmp/claude-1000/-home-dushyant30suthar-Projects-opencode-localhost/af5f9353-9954-46ff-aba2-9bd7fdd6f16d/scratchpad/corpus.txt"
DEPTHS = [4096, 16384, 32768, 49152, 65536, 81920, 94208, 131072, 163328]
SAMPLING = {"temperature": 0.6, "top_k": 20, "top_p": 0.95, "min_p": 0.0}

def post(path, obj, timeout=3600):
    req = urllib.request.Request(BASE + path, json.dumps(obj).encode(),
                                 {"Content-Type": "application/json", "Authorization": "Bearer none",
                                  "x-api-key": "none"})
    return urllib.request.urlopen(req, timeout=timeout)

def jpost(path, obj):
    with post(path, obj) as r:
        return json.load(r)

def encode(text):
    r = jpost("/v1/token/encode", {"text": text})
    return r.get("tokens", r)

def decode(tokens):
    r = jpost("/v1/token/decode", {"tokens": tokens})
    return r.get("text", r)

def gen(prompt_text, n_predict=96):
    """Streaming completion; returns timing split."""
    body = {"prompt": prompt_text, "max_tokens": n_predict, "stream": True,
            "model": "Qwen3.6-27B-exl3-5.00bpw", **SAMPLING}
    t0 = time.time(); t_first = None; parts = []
    with post("/v1/completions", body) as r:
        for raw in r:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data:"): continue
            payload = line[5:].strip()
            if payload == "[DONE]": break
            try: d = json.loads(payload)
            except ValueError: continue
            ch = d.get("choices", [{}])
            txt = ch[0].get("text") or ch[0].get("delta", {}).get("content") if ch else None
            if txt:
                if t_first is None: t_first = time.time()
                parts.append(txt)
    t_end = time.time()
    ttfb = round((t_first or t_end) - t0, 3)
    dec_s = t_end - (t_first or t_end)
    n_tok = len(encode("".join(parts))) if parts else 0   # true token count, chunking-proof
    return {"ttfb_s": ttfb, "gen_tokens": n_tok,
            "decode_tps": round((n_tok - 1) / dec_s, 1) if n_tok > 1 and dec_s > 0 else None,
            "wall_s": round(t_end - t0, 2)}

out = open(OUT, "a")
def rec(**kw):
    kw["config"] = NAME; kw["mode"] = MODE
    out.write(json.dumps(kw) + "\n"); out.flush()
    print(kw, flush=True)

toks = encode(open(CORPUS).read()[:1600000])
print(f"corpus tokens: {len(toks)}", flush=True)
gen(decode(toks[:256]), 8)  # warmup: graph capture, first-request overhead

if MODE == "sweep":
    for D in DEPTHS:
        if D + 1000 > len(toks): break
        text = decode(toks[:D])
        r = gen(text, 1)                      # pure prefill probe
        rec(depth=D, rep="prefill", **r)
        for rep in range(2):                  # cache-hit decode reps
            r = gen(text, 96 if rep == 0 else 192)
            rec(depth=D, rep=rep, **r)

elif MODE == "edit":
    D = 61440
    orig_t = toks[:D]; donor = toks[D + 5000:D + 5040]
    orig = decode(orig_t)
    r = gen(orig, 16); rec(phase="build", depth=D, **r)
    r = gen(orig, 16); rec(phase="build-hit", depth=D, **r)
    for P in (49152, 30720, 15360, 3072):
        mut = decode(orig_t[:P] + donor + orig_t[P + 40:])
        r = gen(mut, 16); rec(phase="diverge", at=P, **r)
        r = gen(orig, 16); rec(phase="rebuild", at=P, **r)

elif MODE == "agents":
    # A: deep conversation. B: shares A's first 30720 tokens, then its own tail.
    A = decode(toks[:61440])
    B = decode(toks[:30720] + toks[70000:80240])
    for label, text in (("A-build", A), ("B-first", B), ("A-again", A),
                        ("B-again", B), ("A-turn3", A)):
        r = gen(text, 16)
        rec(phase=label, **r)

print("DONE", flush=True)

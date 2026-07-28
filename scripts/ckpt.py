#!/usr/bin/env python3
"""Divergence/checkpoint experiment: build a deep conversation, then mutate tokens at
chosen early positions (simulating Claude Code's system-reminder/file re-injection churn)
and measure how much gets reprocessed. Run against a server started with slots debug."""
import json, sys, time, urllib.request, subprocess, os

PORT = int(sys.argv[1])
NAME = sys.argv[2]
OUT = sys.argv[3]
BASE = f"http://127.0.0.1:{PORT}"
SCRATCH = os.path.dirname(os.path.abspath(__file__))
DEPTH = 61440  # ~60k conversation

def post(path, obj, timeout=3600):
    req = urllib.request.Request(BASE + path, json.dumps(obj).encode(),
                                 {'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)

def rss_kb():
    try:
        pid = subprocess.check_output(['pgrep', '-f', f'port {PORT}']).split()[0].decode()
        for line in open(f'/proc/{pid}/status'):
            if line.startswith('VmRSS'):
                return int(line.split()[1])
    except Exception:
        return None

def run(toks, n_predict=16):
    t0 = time.time()
    r = post('/completion', {'prompt': toks, 'n_predict': n_predict, 'cache_prompt': True,
                             'temperature': 0.6, 'top_k': 20, 'top_p': 0.95, 'min_p': 0.0})
    tm = r.get('timings', {})
    return {'wall_s': round(time.time() - t0, 2), 'prompt_n': tm.get('prompt_n'),
            'prompt_ms': round(tm.get('prompt_ms', 0)), 'pp_tps': round(tm.get('prompt_per_second', 0))}

corpus = open(f'{SCRATCH}/corpus.txt').read()
toks = []
CH = 200000
for i in range(0, len(corpus), CH):
    toks += post('/tokenize', {'content': corpus[i:i+CH]})['tokens']
orig = toks[:DEPTH]
# a replacement token that differs: swap in tokens from further in the corpus
donor = toks[DEPTH + 5000:DEPTH + 5040]

out = open(OUT, 'a')
def rec(**kw):
    kw['config'] = NAME
    kw['rss_kb'] = rss_kb()
    out.write(json.dumps(kw) + '\n'); out.flush()
    print(kw, flush=True)

# build incrementally like a real conversation (8k steps) so checkpoints form
rss0 = rss_kb()
for d in range(8192, DEPTH + 1, 8192):
    r = run(orig[:d])
    rec(phase='build', depth=d, **r)
rec(phase='build-done', rss_delta_mb=round(((rss_kb() or 0) - (rss0 or 0)) / 1024))

# divergence tests: mutate 40 tokens at position P, keep everything after identical
for P in (49152, 30720, 15360, 3072):
    mut = orig[:P] + donor + orig[P + 40:]
    r = run(mut)
    rec(phase='diverge', at=P, **r)
    # rebuild original cache state for the next test
    r2 = run(orig)
    rec(phase='rebuild', at=P, **r2)
print('DONE', flush=True)

#!/usr/bin/env python3
"""Depth sweep against a llama-server instance. Measures decode/prefill vs context depth
using the production API path (MTP speculative decoding active)."""
import json, sys, time, subprocess, threading, urllib.request, os

PORT = int(sys.argv[1])
NAME = sys.argv[2]
DEPTHS = [int(x) for x in sys.argv[3].split(',')]
OUT = sys.argv[4]
REPS = int(sys.argv[5]) if len(sys.argv) > 5 else 3
BASE = f"http://127.0.0.1:{PORT}"
SCRATCH = os.path.dirname(os.path.abspath(__file__))

def post(path, obj, timeout=1800):
    req = urllib.request.Request(BASE + path, json.dumps(obj).encode(),
                                 {'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)

def server_rss_kb():
    try:
        pid = subprocess.check_output(['pgrep', '-f', f'port {PORT}']).split()[0].decode()
        for line in open(f'/proc/{pid}/status'):
            if line.startswith('VmRSS'):
                return int(line.split()[1])
    except Exception:
        return None

class GpuSampler:
    def __init__(self):
        self.rows = []; self.stop = False
        self.t = threading.Thread(target=self._run, daemon=True)
    def _run(self):
        while not self.stop:
            try:
                out = subprocess.check_output(['nvidia-smi', '--query-gpu=index,utilization.gpu,power.draw',
                                               '--format=csv,noheader,nounits'], timeout=5).decode()
                for l in out.strip().split('\n'):
                    i, u, p = [x.strip() for x in l.split(',')]
                    self.rows.append((int(i), float(u), float(p)))
            except Exception:
                pass
            time.sleep(0.25)
    def result(self):
        self.stop = True; self.t.join(timeout=2)
        res = {}
        for i in (0, 1):
            r = [(u, p) for (g, u, p) in self.rows if g == i]
            if r:
                res[f'gpu{i}_util'] = round(sum(x[0] for x in r) / len(r), 1)
                res[f'gpu{i}_power'] = round(sum(x[1] for x in r) / len(r), 1)
        return res

# tokenize corpus (chunked to be safe)
corpus = open(f'{SCRATCH}/corpus.txt').read()
toks = []
CH = 200000
for i in range(0, len(corpus), CH):
    toks += post('/tokenize', {'content': corpus[i:i+CH]})['tokens']
print(f'corpus tokens: {len(toks)}', flush=True)
need = max(DEPTHS) + 1000
if len(toks) < need:
    print(f'ERROR corpus too small, need {need}'); sys.exit(1)

out = open(OUT, 'a')
for D in DEPTHS:
    for rep in range(REPS):
        sample_gpu = (rep == 1)  # rep0 pays prefill; sample GPU during a pure-decode rep
        g = GpuSampler()
        npred = 96 if not sample_gpu else 192
        if sample_gpu: g.t.start()
        t0 = time.time()
        r = post('/completion', {'prompt': toks[:D], 'n_predict': npred, 'cache_prompt': True,
                                 'temperature': 0.6, 'top_k': 20, 'top_p': 0.95, 'min_p': 0.0})
        wall = time.time() - t0
        tm = r.get('timings', {})
        row = {'config': NAME, 'depth': D, 'rep': rep, 'wall_s': round(wall, 2),
               'rss_kb': server_rss_kb()}
        for k in ('prompt_n', 'prompt_ms', 'prompt_per_second', 'predicted_n', 'predicted_ms',
                  'predicted_per_second', 'draft_n', 'draft_n_accepted'):
            if k in tm: row[k] = tm[k]
        # base step latency: each accept step emits 1 + accepted tokens
        if tm.get('predicted_n') and tm.get('draft_n_accepted') is not None:
            steps = tm['predicted_n'] - tm['draft_n_accepted']
            if steps > 0: row['step_ms'] = round(tm['predicted_ms'] / steps, 2)
        if sample_gpu: row.update(g.result())
        out.write(json.dumps(row) + '\n'); out.flush()
        print(f"{NAME} d={D:>7} rep={rep} pp={tm.get('prompt_n')}tok@{round(tm.get('prompt_per_second',0))}t/s "
              f"tg={round(tm.get('predicted_per_second',0),1)}t/s step={row.get('step_ms','?')}ms", flush=True)
print('DONE', flush=True)

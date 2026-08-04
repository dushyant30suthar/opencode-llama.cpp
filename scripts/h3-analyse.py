#!/usr/bin/env python3
"""Turn bench/results.jsonl into the tuned configuration, in every form needed.

    .venv/bin/python bench/analyse.py

Writes:
  bench/tuned.json                     machine-readable winners (clips.py reads this)
  bench/FINDINGS.md                    the human report, with per-phase tables
  configs/comfyui-server.ini           drop-in for opencode-localhost's plugin config
  configs/launch-tuned.sh              start ComfyUI directly with the winning flags
  workflows/h3/h3-tuned.json           the winning workflow, ready to run

Ranking rule: lowest server-measured seconds wins, and only rows that actually
succeeded are eligible. A row that OOMed or errored is never a winner however
fast it failed. Where two rows are within 2% the earlier/simpler one is kept —
below that margin this box's run-to-run noise is not distinguishable from a real
difference, and preferring the simpler flag set avoids cargo-culting.
"""
import json, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import matrix  # noqa: E402

RESULTS = os.path.join(HERE, "results.jsonl")
NOISE = 0.02

WORKFLOW_KEYS = ("width", "height", "length", "steps", "cfg", "sampler", "scheduler",
                 "shift_video", "shift_audio", "model_device", "clip_device", "vae_device")

PHASE_TITLE = {
    "A": "Attention backend", "B": "--fast optimizations", "C": "Offload & memory",
    "D": "Device placement", "E": "Sampler / scheduler", "F": "cfg and steps",
    "G": "Sigma shift", "H": "Encoder & VAE precision", "I": "Single-card control",
    "J": "Geometry scaling",
}
# Phases whose rows differ in WORK DONE, not just speed — ranking them by
# seconds would just pick the cheapest setting and call it best.
NOT_RANKABLE = {"F", "J", "E", "G"}


def load():
    if not os.path.exists(RESULTS):
        return []
    rows = []
    for line in open(RESULTS):
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def latest_per_row(rows, stage):
    """Last run wins if a phase was re-run after a fix."""
    out = {}
    for r in rows:
        # later sessions used derived labels like "probe-len124-s10"; those are
        # probe-class measurements and must not vanish from the ranking
        if not str(r.get("stage", "")).startswith(stage):
            continue
        out[(r["phase"], r["row"])] = r
    return out


def winner(cands):
    ok = [r for r in cands if r.get("ok") and r.get("server_s")]
    if not ok:
        return None
    best = min(ok, key=lambda r: r["server_s"])
    simple = [r for r in ok if r["server_s"] <= best["server_s"] * (1 + NOISE)]
    return min(simple, key=lambda r: (len(r.get("flags") or []), r["row"]))


def table(cands, baseline=None):
    ok = [r for r in cands if r.get("ok") and r.get("server_s")]
    base = baseline or (min(r["server_s"] for r in ok) if ok else None)
    lines = ["| row | config | server s | s/step | vs best | peak VRAM |",
             "|---|---|---:|---:|---:|---|"]
    for r in sorted(cands, key=lambda r: (not r.get("ok"), r.get("server_s") or 9e9)):
        cfgs = " ".join(r.get("flags") or []) or "(defaults)"
        if not r.get("ok"):
            lines.append(f"| {r['row']} | `{cfgs}` | — | — | **{r.get('error','failed')}** | — |")
            continue
        rel = f"{(r['server_s']/base - 1)*100:+.1f}%" if base else "—"
        pv = ", ".join(f"{k} {v:.1f}G" for k, v in (r.get("peak_vram") or {}).items())
        lines.append(f"| {r['row']} | `{cfgs}` | {r['server_s']:.1f} | "
                     f"{r.get('s_per_step') or 0:.2f} | {rel} | {pv} |")
    return "\n".join(lines)


def main():
    rows = load()
    if not rows:
        print("no results yet")
        return 1

    probe = latest_per_row(rows, "probe")
    confirm = latest_per_row(rows, "confirm")

    by_phase_probe, by_phase_confirm = {}, {}
    for (ph, _), r in probe.items():
        by_phase_probe.setdefault(ph, []).append(r)
    for (ph, _), r in confirm.items():
        by_phase_confirm.setdefault(ph, []).append(r)

    # ---- pick winners -------------------------------------------------------
    launch_flags, workflow, decisions = [], {}, []
    # Attention first: prefer the CONFIRM ranking when it exists, because
    # attention's share of total work grows with sequence length and the probe
    # systematically understates it.
    for ph in ("A", "B", "C", "H"):
        cands = by_phase_confirm.get(ph) or by_phase_probe.get(ph) or []
        w = winner(cands)
        if not w:
            continue
        stage = "confirm" if by_phase_confirm.get(ph) else "probe"
        new = [f for f in (w.get("flags") or []) if f not in launch_flags]
        launch_flags += new
        decisions.append((ph, w["row"], " ".join(w.get("flags") or []) or "(defaults)",
                          w["server_s"], stage))
    for ph in ("D",):
        w = winner(by_phase_probe.get(ph, []))
        if w:
            workflow.update({k: w[k] for k in ("model_device", "clip_device", "vae_device")})
            decisions.append((ph, w["row"],
                              f"model={w['model_device']} clip={w['clip_device']} vae={w['vae_device']}",
                              w["server_s"], "probe"))

    tuned = {
        "generated": time.strftime("%Y-%m-%d %H:%M"),
        "launch_flags": launch_flags,
        "workflow": workflow,
        "source": f"analyse.py over {len(rows)} runs",
        "note": ("F/J/E/G are NOT auto-selected: their rows change how much work is done or "
                 "how the output looks, so 'fastest' is not 'best'. See FINDINGS.md."),
    }
    with open(os.path.join(HERE, "tuned.json"), "w") as fh:
        json.dump(tuned, fh, indent=2)

    # ---- config artefacts ---------------------------------------------------
    cfgdir = os.path.join(ROOT, "configs")
    os.makedirs(cfgdir, exist_ok=True)
    flagstr = " ".join(launch_flags)

    with open(os.path.join(cfgdir, "comfyui-server.ini"), "w") as fh:
        fh.write(f"""# opencode-localhost ComfyUI backend — MEASURED configuration
# Generated by bench/analyse.py on {tuned['generated']} from {len(rows)} benchmark runs.
# Copy to ~/.config/opencode/providers/comfyui/server.ini
#
# `args` below is not a guess: every flag won its phase on server-measured
# timings on this exact box (2x RTX 5060 Ti, sm_120, PCIe Gen3 x8/x4).

[server]
bin = ~/Projects/comfyui/.venv/bin/python
comfy-dir = ~/Projects/comfyui
models-dir = ~/Projects/comfyui/models
remote =
host = 127.0.0.1
port = 8188
args = {flagstr}
""")

    with open(os.path.join(cfgdir, "launch-tuned.sh"), "w") as fh:
        fh.write(f"""#!/usr/bin/env bash
# ComfyUI with the measured-best flags for this box.
# Generated by bench/analyse.py on {tuned['generated']}.
cd "$(dirname "$0")/.."
exec .venv/bin/python main.py --port "${{1:-8188}}" {flagstr}
""")
    os.chmod(os.path.join(cfgdir, "launch-tuned.sh"), 0o755)

    params = dict(matrix.PROBE, length=124, steps=20, cfg=1.0, **workflow)
    wf = matrix.build_workflow(**params)
    wf["13"]["inputs"]["filename_prefix"] = "video/h3-tuned"
    with open(os.path.join(ROOT, "workflows", "h3", "h3-tuned.json"), "w") as fh:
        json.dump(wf, fh, indent=2)

    # ---- report -------------------------------------------------------------
    md = [f"# MiniMax H3 on 2x RTX 5060 Ti — measured findings",
          f"\n_Generated {tuned['generated']} from {len(rows)} benchmark runs._\n",
          "## Tuned configuration\n",
          f"**Launch flags:** `{flagstr or '(defaults)'}`\n"]
    if workflow:
        md.append(f"**Placement:** `{workflow}`\n")
    md.append("| phase | winner | config | server s | stage |")
    md.append("|---|---|---|---:|---|")
    for ph, row, cfg, s, stage in decisions:
        md.append(f"| {ph} {PHASE_TITLE.get(ph,'')} | {row} | `{cfg}` | {s:.1f} | {stage} |")
    md.append("\nArtefacts: `configs/comfyui-server.ini`, `configs/launch-tuned.sh`, "
              "`workflows/h3/h3-tuned.json`, `bench/tuned.json`.\n")

    for stage, groups in (("confirm", by_phase_confirm), ("probe", by_phase_probe)):
        if not groups:
            continue
        md.append(f"\n## Results — {stage} stage\n")
        for ph in sorted(groups):
            md.append(f"\n### {ph}. {PHASE_TITLE.get(ph, ph)}"
                      + ("  _(not auto-ranked: rows differ in work done or output, "
                         "so fastest != best)_" if ph in NOT_RANKABLE else ""))
            md.append("")
            md.append(table(groups[ph]))
    with open(os.path.join(HERE, "FINDINGS.md"), "w") as fh:
        fh.write("\n".join(md) + "\n")

    print(f"launch flags : {flagstr or '(defaults)'}")
    print(f"placement    : {workflow or '(default)'}")
    print(f"runs         : {len(rows)}")
    print("wrote bench/tuned.json, bench/FINDINGS.md, configs/*, workflows/h3/h3-tuned.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())

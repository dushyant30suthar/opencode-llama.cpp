"""Pre-flight the H3 workflows against ComfyUI's own validator, without executing.

Run from the ComfyUI checkout:
    PYTHONPATH=. .venv/bin/python workflows/h3/validate.py

This calls execution.validate_prompt(), which is the same check the server runs
before queueing — link types, required inputs, combo membership, ranges. It does
NOT load weights, so it is safe to run while the download is still going.

Errors are split into two buckets on purpose:
  STRUCTURAL   a real mistake in the graph — fix now
  NOT-YET      a file that simply has not finished downloading, so it is absent
               from a loader's combo list. Expected until download-h3.sh ends,
               and it is the ONLY error class that should disappear on its own.
"""
import asyncio, json, os, sys, glob

sys.path.insert(0, os.getcwd())

WANTED_FILES = (
    "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
    "minimax_h3_ref2va_pruned_int8_convrot.safetensors",
    "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
    "minimax_h3_video_vae_fp16.safetensors",
    "minimax_h3_audio_vae_fp32.safetensors",
)


def looks_like_missing_download(message: str) -> bool:
    """A combo rejection naming one of our files is a download artefact, not a bug."""
    return any(f in message for f in WANTED_FILES) and (
        "not in" in message or "combo" in message.lower() or "value not in list" in message.lower()
    )


async def main() -> int:
    import execution
    import nodes

    await nodes.init_extra_nodes(init_api_nodes=False)

    here = os.path.dirname(os.path.abspath(__file__))
    files = sorted(glob.glob(os.path.join(here, "h3-*.json")))
    if not files:
        print("no workflows found next to this script")
        return 1

    worst = 0
    for path in files:
        name = os.path.basename(path)
        with open(path) as fh:
            prompt = json.load(fh)
        valid, error, good_outputs, node_errors = await execution.validate_prompt(name, prompt, None)

        if valid:
            print(f"OK        {name}  ({len(prompt)} nodes, outputs: {sorted(good_outputs)})")
            continue

        # Flatten everything the validator complained about.
        #
        # The top-level error is only ever a summary ("Prompt outputs failed
        # validation") when per-node errors exist — counting it as its own
        # finding marks every workflow structural and hides the real answer.
        # The per-node list is what actually says what is wrong.
        messages = []
        for node_id, err in (node_errors or {}).items():
            for e in err.get("errors", []):
                messages.append(f"node {node_id}: {e.get('message')} — {e.get('details')}")
        if error and not messages:
            messages.append(str(error.get("message", error)))
            details = error.get("details")
            if details:
                messages.append(str(details))

        pending = [m for m in messages if looks_like_missing_download(m)]
        structural = [m for m in messages if not looks_like_missing_download(m)]

        if structural:
            worst = max(worst, 2)
            print(f"STRUCTURAL {name}")
            for m in structural:
                print(f"    ! {m}")
        else:
            worst = max(worst, 1)
            print(f"NOT-YET   {name}  (graph is sound; waiting on downloads)")
        for m in pending:
            print(f"    … {m}")

    print()
    print({0: "all workflows valid", 1: "graphs sound, weights still downloading",
           2: "STRUCTURAL ERRORS — fix these"}[worst])
    return 0 if worst < 2 else 2


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

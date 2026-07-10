# Features added by the opencode fork

Everything below lives on the `llamastack` branch of the `opencode/` submodule
and works with a plain llama.cpp `llama-server` — no other runtime needed.

| Feature | One line |
| --- | --- |
| [Zero-config provider](zero-config-provider.md) | Launch `opencode`, local models are just there — spawns the router itself if none is running |
| [Model discovery](model-discovery.md) | Reads the LM Studio folder layout; every quant variant is its own selectable model |
| [Preserve thinking](preserve-thinking.md) | The fix for thinking models looping mid-task in agentic tool use |
| [`/config` screen](config-screen.md) | Edit per-model settings (ctx, GPU split, KV quant, temperature) inside the TUI |
| [Load progress](load-progress.md) | Real-time VRAM-load and prompt-prefill progress instead of an indefinite spinner |
| [VRAM warm-up](vram-warmup.md) | Selecting a model starts loading it immediately — the swap happens while you type |
| [LAN hosting](lan-hosting.md) | One toggle exposes the engine to your network as an OpenAI-compatible endpoint |
| [Web search](web-search.md) | Local models get the built-in websearch tool |
| [No self-update](no-self-update.md) | Fork builds refuse to overwrite themselves with stock upstream |

# LAN hosting + server status panel

The router can serve the whole network: with "Expose on network" enabled it
binds `--host 0.0.0.0` (port 9337), making every other machine on the LAN an
OpenAI-compatible client of your GPU box.

> **Security note:** there is no authentication. Anyone on the LAN can use the
> endpoint while it is exposed. Keep it off on untrusted networks.

Toggle it under `/config` → Server → "Expose on network". The setting persists
in `~/.local/state/llamastack/server.json`; saving restarts the detached
router (via `router.pid`) so the binding changes immediately. If the running
router predates the pidfile, the UI degrades to "restart the router to apply".

## Status display

The session sidebar gets a **Local server** panel (local models only) showing:

- the endpoint URL with the machine's LAN IP when exposed
  (e.g. `http://192.168.1.x:9337/v1`, `127.0.0.1` when localhost-only),
- the loaded model's key parameters (`ctx`, `ngl`, `split`, `kv`), sourced
  from the router's `GET /models` with `models.ini` as fallback, refreshed
  every 10 s,
- live per-GPU VRAM and utilization.

The same information appears under the Server entry at the top of `/config`.

Code: `packages/tui/src/feature-plugins/sidebar/llamastack.tsx`.

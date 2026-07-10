# Web search for local models

The built-in `websearch` tool is enabled for the llamastack provider, so local
models can search the web the same way hosted models can — useful for
version-sensitive questions (API docs, changelogs) where a local model's
training cutoff would otherwise mislead it.

Code: `packages/opencode/src/tool/registry.ts`.

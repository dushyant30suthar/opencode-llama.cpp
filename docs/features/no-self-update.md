# No self-update

The fork's version string (`0.0.0-llamastack-TIMESTAMP`) always compares older
than upstream releases, so stock opencode would offer to "update" — and
overwrite the fork binary with a stock upstream build, silently removing the
entire local stack.

Fork builds (channel or version containing `llamastack`) are exempt:

- no update prompt at launch,
- `opencode upgrade` and the server upgrade endpoint refuse with
  `custom llamastack build — update by rebuilding the fork`.

Updating is deliberate instead: rebase branch `llamastack` onto upstream,
rebuild with `scripts/build-opencode.sh`.

Code: `packages/opencode/src/installation/index.ts`,
`packages/opencode/src/cli/upgrade.ts`, `packages/opencode/src/cli/cmd/upgrade.ts`.

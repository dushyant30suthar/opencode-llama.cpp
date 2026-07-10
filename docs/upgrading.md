# Upgrading the stack

Two components move independently — the **opencode fork** (tracks
`sst/opencode`, branch `dev`) and **llama.cpp** (tracks `ggml-org/llama.cpp`,
branch `master`). This repo's submodule pins record the combinations that were
validated together. Upgrading means: move a component forward, rebuild it,
validate, bump the pin.

## The automated way: `opencode upgrade`

On fork builds, `opencode upgrade` **is** the stack upgrader (stock opencode's
binary download would wipe the fork, so that path is disabled):

```sh
opencode upgrade --check     # read-only: how far behind is each component,
                             # and would syncing be conflict-free?
opencode upgrade             # check + confirm + pull/rebase + compile + install
opencode upgrade llama       # just the engine
opencode upgrade opencode    # just the frontend
opencode upgrade --yes       # skip the confirmation
```

What it does under the hood (identical to the manual path below):

1. Finds the stack repo (`$LLAMASTACK_REPO`, else `~/Projects/opencode-llama.cpp`).
2. Fetches both upstreams; reports commits-behind and local-commits-on-top.
3. **Tests conflict-freeness before touching anything** — fast-forward is
   clean by construction for llama.cpp (no local patches); for opencode a
   `git merge-tree` dry run predicts rebase conflicts without touching the
   working tree.
4. Refuses a component whose working tree is dirty or whose sync would
   conflict — those need a human (see manual path).
5. llama.cpp: fast-forward + `scripts/build-llama.sh` + verifies the binary runs.
6. opencode: rebase + `bun install` + build + **replaces the running binary**
   (previous one kept as `<binary>.backup`).
7. Commits the new submodule pins in the stack repo (does **not** push
   anything — pushing stays your decision).

Build output streams to `~/.local/state/llamastack/upgrade.log`.

## The manual way — every step

### 0. See where you are

```sh
cd ~/Projects/opencode-llama.cpp
git submodule status                       # the two pinned commits
~/.opencode/bin/opencode --version         # installed frontend build stamp
llama.cpp/build/bin/llama-server --version # installed engine build
```

### 1. Upgrade llama.cpp (the engine)

```sh
cd llama.cpp
git fetch https://github.com/ggml-org/llama.cpp master
git rev-list --count HEAD..FETCH_HEAD     # how far behind
git merge --ff-only FETCH_HEAD            # no local patches -> always clean
../scripts/build-llama.sh                 # enforces CUDA >= 13.3, clean build dir
build/bin/llama-server --version          # sanity
build/bin/llama-server --list-devices     # CUDA kernels initialize on your GPUs
```

Keep the GitHub fork in sync too (so the submodule URL can serve the pin):

```sh
gh repo sync <you>/llama.cpp --source ggml-org/llama.cpp --branch master
```

Validation beyond smoke: re-run the relevant `bench/` probes if release notes
touched your model families (a champion-config server run takes ~3 minutes and
catches regressions — see `bench/README.md`).

Build rules the script enforces, in case you build by hand: **never CUDA
13.2** (miscompiles quant kernels), always `rm -rf build` first (LTO caches),
host compiler ≤ GCC 15 for nvcc, arch `native` (resolves to `120a` on
Blackwell — required for the NVFP4 fast path).

### 2. Upgrade opencode (the frontend)

```sh
cd ../opencode
git status --porcelain                    # MUST be empty; commit or stash first
git fetch upstream dev                    # upstream = github.com/sst/opencode
git rev-list --count HEAD..FETCH_HEAD     # how far behind
git rebase FETCH_HEAD
```

If the rebase conflicts: fix the listed files, `git add`, `git rebase
--continue` — or bail out losslessly with `git rebase --abort`.

Then **always** refresh dependencies (a rebase changes package.json; the
pre-push typecheck hook fails on stale node_modules):

```sh
bun install --ignore-scripts
bun run --cwd packages/core fix-node-pty
bun run --cwd packages/opencode typecheck
cd packages/opencode && bun run script/build.ts --single --skip-embed-web-ui
```

Replace the installed binary (keep a fallback):

```sh
cp ~/.opencode/bin/opencode ~/.opencode/bin/opencode.backup
install -Dm755 dist/opencode-linux-x64/bin/opencode ~/.opencode/bin/opencode
~/.opencode/bin/opencode --version
```

Push the rebased branch to the fork once you're satisfied (history was
rewritten, so force is expected):

```sh
git push --force origin opencode-llama.cpp
```

Never use GitHub's web "Sync fork" button on this branch — it creates a merge
commit (or offers to discard your commits) instead of rebasing.

### 3. Record the validated combination

```sh
cd ~/Projects/opencode-llama.cpp
git add opencode llama.cpp
git commit -m "bump pins: <what changed and what validated it>"
git push
```

### 4. Rollback

| What broke | How back |
| --- | --- |
| New opencode binary | `cp ~/.opencode/bin/opencode.backup ~/.opencode/bin/opencode` |
| Bad opencode rebase | `git reflog` in `opencode/` — the pre-rebase commit is right there; `git reset --hard <sha>` |
| New llama.cpp build | `git -C llama.cpp checkout <previous pin>` + `scripts/build-llama.sh` (pins are in this repo's history) |

### Trap to remember

The build stamps the **branch name** into the version string
(`0.0.0-opencode-llama.cpp-TIMESTAMP`), and `isLlamaStackBuild()` in
`packages/opencode/src/installation/index.ts` recognizes fork builds by
matching that string. **If the branch is ever renamed again, extend the marker
list first** — otherwise the new build loses its self-update protection and
`opencode upgrade` reverts to the stock binary-download path.

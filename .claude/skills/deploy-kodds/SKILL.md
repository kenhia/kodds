---
name: deploy-kodds
description: Deploy kodds from merged main to kubs0 — export a release, sync it with CUDA, flip the `current` symlink, restart the systemd user unit, and verify the 14B is loaded on the GPU and answering over HTTP and MCP on the tailnet, with klams' TEI embedders untouched. Use when asked to deploy/redeploy/ship kodds, or when sprint-ship reaches Phase 7. Deploys committed code only; runs locally on kubs0.
---

# Deploy kodds to kubs0

kodds runs as **one process with Qwen3-14B resident on kubs0's 4080**, beside
klams' two `text-embeddings-router` (TEI) processes. It serves HTTP
(`/healthz`, `/v1/*`) and MCP (`/mcp`) on `127.0.0.1:7780`, and
`tailscale serve` publishes that as `https://kubs0.encke-wahoo.ts.net:7780`.

| thing | where |
|---|---|
| releases | `~/.local/share/kodds/releases/<sha>/` (a `git archive` + its own `.venv`) |
| live / rollback | `~/.local/share/kodds/current`, `previous` (symlinks) |
| per-call request log | `~/.local/share/kodds/calls/<YYYY-MM>.jsonl` (mode 600; state, survives deploys) |
| route's pinned overlay | `~/.local/share/kodds/private/route.json` ([docs/route-overlay.md](../../../docs/route-overlay.md)) |
| unit | `~/.config/systemd/user/kodds.service`, installed from `deploy/kodds.service` |
| model | `~/models/gguf/Qwen3-14B-Q4_K_M.gguf` (not in any release) |

**The service never runs from `~/src/ai/kodds`.** `tasks/` resolves relative
to the source tree, so a release is a source tree, not a wheel. If the
service ran from the dev checkout, a sprint branch checked out there would go
live on the next restart.

## Run everything in the foreground

The model takes a while to load, and `just deploy` waits for it. Under karc a
backgrounded job dies when the turn ends (agent-skills #3126), and "I'll
check when it finishes" never happens. Use a long Bash timeout (10 min) for
`just deploy` and every poll. Never `&` or `run_in_background`.

## 1. Preflight

```sh
cd ~/src/ai/kodds
git status --short                      # must be empty; never stash
git rev-parse --abbrev-ref HEAD         # main, except for a deliberate pre-merge test
git pull --ff-only origin main
just check
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
```

Note the two TEI pids. They must be the same after the deploy. VRAM: kodds
needs ~10.5 GiB at n_ctx 8192, and the service refuses to start with less
than 11,000 MiB free. On a **re**deploy the old kodds is holding its own
share: `systemctl restart` frees it before the new one checks. If
something *other* than kodds and TEI is on the card, stop and ask. Never
free VRAM by stopping TEI.

## 2. Deploy

```sh
just deploy          # timeout 600000
```

It refuses a dirty tree and warns off-main (the warning is not permission).
It exports HEAD into a release dir and runs `uv sync --frozen --no-dev`
there. The CUDA build is declared in `pyproject.toml`
(`extra-build-variables`), so the uv cache's CUDA wheel is reused, and it
checks GPU offload, rebuilding once if it got a CPU build. Then it installs
the unit, flips `current` (the old one becomes `previous`), restarts,
ensures the `tailscale serve` entry exists, and waits for `/healthz`. It
fails unless healthz shows **this commit, loaded, with GPU offload**.

It prints each task's `calibrated`. If **route shows `calibrated=False`**,
read [docs/route-overlay.md](../../../docs/route-overlay.md): either the state
dir has no overlay (`just overlay-install`) or the overlay no longer matches
the fit (refit). Deploy never regenerates the overlay, on purpose. Report
it, don't "fix" it by running `just overlay`.

## 3. Verify live, over the tailnet

`just deploy` checked loopback. These go through `tailscale serve`, which is
what every other host uses:

```sh
U=https://kubs0.encke-wahoo.ts.net:7780
curl -fsS $U/healthz | python3 -m json.tool | head -40   # commit == merged main, gpu_offload true
curl -fsS $U/v1/classify -H 'content-type: application/json' \
  -d '{"task":"severity","inputs":{"finding":"kubsdb: root filesystem at 97%"}}'
curl -fsS $U/v1/classify -H 'content-type: application/json' \
  -d '{"task":"triage","inputs":{"message":"Your invoice is attached, click here to verify your password"}}'
curl -fsS $U/v1/classify -H 'content-type: application/json' \
  -d '{"task":"route","inputs":{"title":"Rotate the klams bearer token","content":"The token in klams.toml is old; mint a new one and update the MCP registrations."}}'
```

Check each result for `calibrated: true` and `model: Qwen3-14B-Q4_K_M.gguf`.
Each classify also returns a `request_id`. Confirm the last one landed in the
call log, and that healthz's `call_log.writable` is true:

```sh
tail -n 1 ~/.local/share/kodds/calls/$(date -u +%Y-%m).jsonl | python3 -m json.tool | grep request_id
stat -c '%a %n' ~/.local/share/kodds/calls ~/.local/share/kodds/calls/*.jsonl   # 700, 600
```

Verification calls are logged like any other (with no `caller`), which is
fine: the log is for joining ids a consumer kept, and nobody keeps these.
Take `inputs` names from `GET /v1/tasks` if the ones above ever drift.

MCP over the same URL (streamable HTTP, stateless, JSON responses):

```sh
curl -fsS $U/mcp -H 'content-type: application/json' \
  -H 'accept: application/json, text/event-stream' \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/list"}' | python3 -m json.tool | grep '"name"'
curl -fsS $U/mcp -H 'content-type: application/json' \
  -H 'accept: application/json, text/event-stream' \
  -d '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"classify","arguments":{"task":"severity","inputs":{"finding":"disk 97%"}}}}'
```

Expect `list_tasks`, `classify` and `score`, and a result with
`calibrated: true`. If kubs0's Claude has the server registered
(`claude mcp list | grep kodds`), one real `classify` call through it is
the strongest check. Fleet registration is agent-skills', not this skill's.

Last, re-run the two `nvidia-smi` queries: **the TEI pids must be
unchanged**, and total use should be ~13.4 GiB.

## Rollback

```sh
just rollback        # swaps current <-> previous and restarts
```

`previous` is one deploy deep, and `just deploy` prunes every other release.
To go back further, check out the commit and `just deploy` it. A failed
deploy does not roll back a merge: the code is fine, the rollout isn't.

If the unit won't start at all: `journalctl --user -u kodds -n 50`. The
refusals are explicit (`kodds: refusing to start: …`): a CPU build, too
little free VRAM, a missing model file, or a non-loopback `KODDS_HOST`.
After five failed starts in ten minutes systemd stops retrying:
`systemctl --user reset-failed kodds` once the cause is fixed.

## 4. Record it

Append a `## Deployed` section to this sprint's record
(`sprints/<NNN>-<slug>.md`): the commit, when (read the clock with
`TZ=America/Los_Angeles date`), healthz's calibrated flags per task, the
tailnet classify and MCP results, VRAM before/after, and the TEI pids
before/after. The branch is gone by now, so commit to `main` and push.

A **first** deploy, or one that changes the unit or the serve entry, is a
machine change: run the `record-machine-change` skill. The unit and the
`tailscale serve` entry are declared in k-homelab's kubs0 manifest. Keeping
those in step is k-homelab's sprint, not a step here.

## What this skill does not do

- **Regenerate route's overlay.** It is pinned (docs/route-overlay.md).
- **Register the MCP server on other hosts.** That is agent-skills (korg:3213).
- **Touch TEI or klams.** If VRAM is short, stop and ask.

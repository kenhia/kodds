# 003 — service + MCP, deployed resident on kubs0

korg: proposal 3209 (slice 1 of program 3214, "kodds as a fleet service") —
#3205 HTTP service, #3206 MCP surface, #3207 deploy, #3215 overlay doc.
Run as an overseen karc leg on kubs0.

## Goal

Turn 002's calibrated tasks into a live service: one process on kubs0 with
Qwen3-14B resident, HTTP and MCP on the same model, deployed from merged
main by a `deploy-kodds` skill that sprint-ship's Phase 7 runs.

## Premise check (sprint start)

All held. `TASKS_DIR` resolves to `parents[2]/tasks` (so a release is a
source tree). The CUDA build existed only as a README line. Port 7780 was
free (klams 7777/7779, khlenv 7770; tailscale serve had 4870, 4871, 7777,
7779 and 8027). The GPU showed 2,883 MiB used, which is just TEI's two
routers (pids 875202, 875203). Linger is on for ken. The dev venv's
llama-cpp-python had GPU offload. Ken's uncommitted roadmap edit (the Laya
bake-off diversion) rides along, per the proposal's comment 3049.

## Service (#3205) and MCP (#3206)

`kodds.service`: one Starlette app. The MCP SDK's `MCPServer` builds it
(`streamable_http_app`), and the HTTP routes are added with `custom_route`.
So `/healthz`, `/v1/tasks`, `/v1/classify`, `/v1/score` and `/mcp` share
one process, one lifespan and one `Service`, which means one model.

- **mcp is 2.x** (2.2.0). `FastMCP` is now `mcp.server.mcpserver.MCPServer`,
  the in-process test client is `mcp.Client(server)`, and pydantic models
  take snake_case field names. `ty` caught `ToolAnnotations(readOnlyHint=…)`
  being silently discarded, and a test now asserts the annotations arrive.
- **Serialisation**: `Service._run` holds one `threading.Lock` around every
  inference. Handlers run inference in a worker thread
  (`anyio.to_thread`), so `/healthz` keeps answering while route holds the
  lock. Results carry `latency_ms` (inference) and `queued_ms` (lock wait).
- **Errors**: 404 for an unknown task, 422 for bad inputs or choices, 400
  for a body that isn't a JSON object. Over MCP they are tool errors.
- **`/healthz`** per task: `calibrated`, `overlay` (file present),
  `prompt_sha256`, `fitted_prompt_sha256` and `prompt_matches_fit`, so a
  missing or drifted route overlay shows up rather than going unnoticed.
  Plus model, model path, GPU offload, free/total VRAM (nvidia-smi) and
  the commit (the release's `COMMIT` file, else `git rev-parse`).
- **MCP transport**: stateless with JSON responses, so a restart (every
  deploy) never strands a client on a session id. DNS-rebinding protection
  stays on. The SDK enables it for a loopback bind and would reject the
  ts.net `Host` that tailscale serve passes through with 421, so the
  allowlist is loopback plus `kubs0.encke-wahoo.ts.net` (configurable:
  `KODDS_PUBLIC_HOSTS`).
- **Refusals at start** (`main`): no GPU offload (a CPU build), less than
  `KODDS_MIN_FREE_MIB` (11,000) VRAM free, a missing model, or a
  non-loopback `KODDS_HOST`.
- Tool descriptions say raw probabilities are overconfident (T≈5) and to
  threshold only on calibrated results.
- `load_task`/`load_tasks` take `private_dir`, so the overlay can live in
  a state dir outside the release.

Tests (model-free, 001's bigram fake): calibrated vs raw passthrough,
404/422/400, score is always raw, tasks/healthz calibration reporting and a
drifted prompt hash, lock serialisation (six concurrent classifies, peak
concurrency 1), MCP over HTTP on the loopback and tailnet `Host`s (421 on a
foreign one), and an in-process MCP round trip (tools/list, classify, the
annotations, an unknown task as a tool error).

## Deploy (#3207)

- **CUDA declared in the repo** (the repair the item asked for):
  `[tool.uv.extra-build-variables] llama-cpp-python = { CMAKE_ARGS =
  "-DGGML_CUDA=on" }`. That gives the build a new uv cache key, so the dev
  venv rebuilt once, which proved the setting. The first attempt failed:
  **CMake could not find `nvcc`**, because a non-interactive shell (a karc
  leg, systemd) doesn't get `~/.bashrc`'s `/usr/local/cuda/bin`. The
  justfile now exports it on `PATH`. The runtime libraries resolve through
  ld.so.conf, so the unit needs no environment for them. Rebuild: 2m46s.
  `just setup` = sync + assert GPU offload.
- `deploy/kodds.service`: a user unit with `Restart=on-failure` (20 s), and
  systemd stops retrying after five failed starts in ten minutes (so an
  explicit refusal is not retried forever). It runs
  `~/.local/share/kodds/current/.venv/bin/kodds-serve`, never the dev
  checkout.
- `just deploy`: kubs0 only, refuses a dirty tree, warns off-main. It runs
  `git archive` HEAD into `releases/<sha>`, builds the venv in place
  (`uv sync --frozen --no-dev`; the venv records absolute paths, and
  `.complete` marks a finished build), and asserts GPU offload, rebuilding
  llama-cpp-python once if needed. Then it flips `current` (the old target
  becomes `previous`), installs the unit, restarts, adds the
  `tailscale serve --https=7780` entry if it's missing, and waits for
  `/healthz` to show this commit loaded with GPU offload. It prunes every
  release but current and previous. `just rollback` swaps the two.
- Route overlay: read from `~/.local/share/kodds/private/`, and deploy
  never generates it. `just overlay` regenerates the dev tree's copy (a
  refit step). `just overlay-install` installs it into the state dir only if
  route would be calibrated with it against the committed fit.
- `.claude/skills/deploy-kodds/SKILL.md` (preflight with TEI pids, deploy,
  tailnet verification over HTTP and MCP, rollback, record) and
  `.sprint-deploy`.

## Overlay doc (#3215)

`docs/route-overlay.md`: what the overlay is and where it lives, why it's
pinned (the prompt hash), how to tell it has drifted (`/healthz` fields, or
a korg project added or retired), the eight-step refresh, including
stopping the service first because two 14Bs don't fit beside TEI, and what
happens if you skip it. Linked from the README and the deploy skill. The
klams pointer is written after the doc merges.

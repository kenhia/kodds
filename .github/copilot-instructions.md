<!-- kproject:begin — managed by kprojects; do not edit inside this block -->

## kproject conventions

This project uses the kproject minimal harness
(<https://github.com/kenhia/kprojects>). Keep context small; prefer doing
over ceremony.

### Layout

- `sprints/` — the project's evolution, one record per PR-sized unit of
  work (a "sprint")
  - `planning/` — planning docs; at minimum `roadmap.md` (the general plan)
  - `review/` — more formal reviews as the project matures
  - sprint records: `###-<short-name>.md` for small projects, or a
    `###-<short-name>/` directory of files for larger/more formal ones
  - a sprint record is one informal narrative: goal, decisions, what
    shipped, follow-ups — written during the sprint, not after
  - projects that deploy end the record with a `## Deployed` section:
    what shipped, where, when, and what was verified live — appended
    after the deploy, not predicted before it
- `docs/` — project documentation, architecture, usage
- `.scratch/` — git-ignored scratch space for user or agent ephemera;
  use it instead of /tmp
- `justfile` — dev recipes; default recipe is `@just --list`; `just check`
  runs the CI gates; `just deploy` (or variants) if the project deploys
- `.env` — git-ignored; tokens and environment vars

### Workflow

- One sprint ≈ one PR. Sprint proposals and work items are managed in
  `korg`; durable cross-project knowledge goes in `klams`.
- Mark each work item resolved as its work completes — don't batch the
  resolutions into sprint-ship. A proposal's progress should be readable
  while the sprint is running, which is the only time it is useful.
- If the korg or klams MCP tools are unavailable in your session, say so
  up front — don't silently work around missing infrastructure.
- A few projects share contract surfaces with siblings and have a
  **guiding plan** constraining how those change; most have none, and one
  grep is the whole cost of finding out. Grep the `index.md` routing
  table in `kai:~/src/tools/cross-project-planning` — a local path on
  kai, read through kaed from any other host (`root: "kai:src"`, path
  `tools/cross-project-planning/…`); don't clone a second copy. Not
  listed → nothing applies. Listed → read the mapped plan folder before
  planning sessions and before changing a contract surface it names, and
  amend the plan in the same ship when what you build diverges from it.
- TDD preferred: write the failing test first when practical.

### Tooling preferences

- Python managed by `uv`; lint/format with `ruff`; typecheck with `ty`
  (astral toolchain)
- License is MIT unless specifically directed otherwise

<!-- kproject:end -->

## Project

kodds is a local LLM **classifier**: given a prompt and a set of choices, it
returns a probability for each choice, read from the model's next-token
logits (the "Jev" idea — see
<https://www.nobodywho.ai/posts/jev-in-25-lines/>, a parody that proves the
point). One forward pass per choice set, no generation, no API calls.

**Status:** live service (sprint 003, 2026-09-25). Qwen3-14B Q4_K_M is
resident on kubs0, serving calibrated tasks (route, severity, triage) over
HTTP and MCP at `https://kubs0.encke-wahoo.ts.net:7780`. It is deployed by
`just deploy` / the `deploy-kodds` skill, never from this checkout. A sweep
or GPU test needs `systemctl --user stop kodds` first: two 14Bs don't fit
beside TEI.

- **Stack:** Python via `uv`; `llama-cpp-python` built with CUDA; GGUF
  models from Hugging Face.
- **Host:** kubs0 (RTX 4080 SUPER, 16 GB). It must run **alongside** klams'
  two `text-embeddings-router` processes (~2.9 GB) — check `nvidia-smi`
  before loading anything big, and never evict them. kai's 5090 is the RA's
  (kvllm); use it only with Ken's say-so.
- **Model candidates:** Qwen3-4B Q8_0, Qwen3-8B Q6_K / Q8_0, Qwen3-14B
  Q4_K_M. Choose on calibration (Brier/ECE) as well as accuracy — the
  probabilities are the product.
- **Build/test:** `just check` (ruff format + lint, ty, pytest). Tests that
  need a GPU or downloaded weights must be marked and skipped by default,
  so `just check` stays fast and runnable without a model.

Read first: `src/kodds/`, `sprints/planning/roadmap.md`, the latest
`sprints/###-*.md`.

Gotchas:

- A choice usually spans several tokens. Score it as the summed log-prob of
  all of its tokens given the prompt, not just the first token.
- Qwen3 defaults to thinking mode; disable it (`/no_think` or the chat
  template's `enable_thinking=False`) or the next token is `<think>`.
- Model weights are big: keep them out of the repo (`.scratch/` or a models
  dir outside the tree).

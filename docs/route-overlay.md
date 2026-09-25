# route's private overlay: pinned, and how to refresh it

**Short version.** The service's route calibration only applies to the exact
overlay it was fitted on. Deploy never regenerates the overlay. If route
reports `calibrated: false`, the overlay is missing or has drifted, and the
fix is the refit below. Regenerating the overlay on its own is not a fix.

## What the overlay is

`route` picks which korg project owns a work item. Its prompt lists all 46
projects. Each entry has the project's one-line contract, a notes excerpt
(≤200 chars) and three real work-item titles. The contracts are public and
live in `tasks/route.toml`. The excerpts and titles are private korg
content (hosts, paths, the tailnet domain), and kenhia/kodds is public. So
they ship as a git-ignored **overlay**, `route.json`, whose descriptions
replace the TOML's when a task loads (sprint 002, "Routing prompt").

| where | path |
|---|---|
| dev tree (generated, used for sweeps and refits) | `tasks/private/route.json` |
| serving host, kubs0 (what the service reads) | `~/.local/share/kodds/private/route.json` |

The service reads the second path through `KODDS_PRIVATE_DIR`
(`deploy/kodds.service`). It lives outside the release directories, so a
deploy does not touch it. The previous one is kept beside it as
`route.json.prev`.

## Why it is pinned (Ken, 2026-09-25)

A calibration records `prompt_sha256`: a hash of the prompt as the model
sees it, less the inputs (`Task.prompt_hash()`). The overlay is part of that
prompt, and regenerating it changes the hash. korg notes change, a different
example title gets sampled, and a project is added or retired. When the hash
no longer matches the fit, `classify` drops the calibration and returns raw
probabilities with `calibrated: false`. It never borrows parameters fitted
for a different prompt.

So a regenerated overlay without a refit turns a calibrated route into an
uncalibrated one, silently. Raw route probabilities are overconfident. Its
ECE was 0.34 raw vs 0.07 calibrated on 002's test split. That is why deploy
**never** generates the overlay, and why `just overlay-install` refuses to
install one the committed calibration doesn't match.

## How to tell it has drifted

- `GET /healthz` (or `just deploy`'s output) lists every task with
  `calibrated`, `overlay` (file present), `prompt_sha256`,
  `fitted_prompt_sha256` and `prompt_matches_fit`.
  - `overlay: false`: there is no overlay in the state dir. Run
    `just overlay-install` from a dev tree whose overlay still matches.
  - `overlay: true, prompt_matches_fit: false`: the overlay or `route.toml`
    changed after the fit. Refit.
- Adding or retiring a korg project changes route's choice list, so its
  calibration stops applying (`calibration_for` also checks the choices).
  The service keeps serving the old list until someone runs the refit.
  That is the normal reason to refit.

## Refresh procedure (a refit)

Run on kubs0 in a dev checkout (`~/src/ai/kodds`), on a sprint branch,
because it produces a commit.

1. **Stop the service**: `systemctl --user stop kodds`. A sweep loads its
   own copy of the 14B (~10.5 GiB), and two of them do not fit on the 16 GB
   card beside klams' TEI embedders. The sweep refuses to start unless TEI is
   on the GPU, and never evicts it.
2. **If the project set changed**: `just evalset`, after
   `uv run python evals/build_evalset.py projects` has refreshed `route.toml`'s
   `[choices]`. Needs korg (REST on kubsdb:5674). The splits themselves stay
   fixed. A new project has no cal items and keeps the median bias.
3. `just overlay`: regenerates `tasks/private/route.json` from korg. Needs
   korg.
4. `just sweep route cal content-examples-notes` (GPU, roughly 7 minutes for
   300 items at ~1.4 s), then the same for `test`. It writes the log-probs
   under `evals/results/002/`.
5. `just calibrate --write Qwen3-14B-Q4_K_M.gguf route=content-examples-notes
   severity=default triage=default` (no GPU). It refuses if the swept
   variant doesn't hash the same as the task as it will be served, so it
   cannot write a fit for the wrong prompt.
6. Commit `tasks/route.calibration.json` (and `route.toml` /
   `evals/results/002/` if they changed), and ship through the usual sprint.
7. `just overlay-install`: copies `tasks/private/route.json` into the state
   dir, but only after checking that route is calibrated with it against
   the committed fit.
8. `just deploy` from merged main (or `systemctl --user start kodds` if
   nothing else changed). Check that `/healthz` shows route
   `calibrated: true`.

Budget about 20 minutes: two sweeps and a restart. The GPU is only needed
in step 4, and korg only in steps 2 and 3.

## If you skip it

Route keeps serving, just uncalibrated: `calibrated: false` and
`probs == raw`. The ranking is still usable: raw top-1 was 61% on test, against
64.7% calibrated (the per-choice bias moves some picks). But the
probabilities are overconfident. A consumer that thresholds
on them should check `calibrated` and not act on raw numbers. Severity and
triage don't use an overlay and are unaffected.

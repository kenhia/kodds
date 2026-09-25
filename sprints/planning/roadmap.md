# Roadmap

> The general plan for this project. Keep it current; detail lives in the
> sprint records.

## Done

- **001 — scoring core + model bake-off.** `score(prompt, choices)` over
  llama-cpp-python, a 154-item eval set, and a four-model bake-off on kubs0.
  Default model: **Qwen3-14B Q4_K_M** (11.6 GiB peak beside TEI); fallback
  8B Q6_K. Raw probabilities are overconfident (T≈10). See
  `sprints/001-scoring-core-and-bake-off.md`.
- **002 — calibration.** Named **tasks** (`tasks/*.toml`: template, choices,
  per-model calibration pinned to a prompt hash), fixed cal/test/train
  splits (route: 1,611 real korg items, ids only in the public repo), and a
  settled routing prompt: WI content plus each project's contract, a notes
  excerpt and three example titles, shipped as a private overlay. Route over
  46 projects: 61% → **64.7%** test accuracy, Brier 0.71 → **0.53**, ECE
  0.34 → **0.07** (prompt + per-choice bias + T). Severity/triage keep a
  pooled T≈5. See `sprints/002-calibration.md`.

## Next

- **003 — service + MCP.** HTTP API (task or prompt + choices →
  probabilities) as a systemd user service on kubs0, tailnet-served, with
  the model kept resident; an MCP surface over it so agents call it
  directly. Calibrated tasks from 002 are what makes the API worth calling.
  The serving host must generate route's private overlay
  (`evals/build_evalset.py overlay`, needs korg), or route serves raw
  probabilities; surface `calibrated` in the API rather than hiding it.
- **004 — first consumer** (tentative). Wire one real caller: kmon finding
  severity is the strongest candidate (91% / ECE 0.08 in 001), korg
  routing if 002 lifts it. A live consumer is also the only source of
  real labels for severity. Touches the consumer's contract, so it is
  that project's decision too.
- **005 — fine-tuning** (tentative, gated). A routing LoRA trained on korg
  history, only if 002's prompt work and calibration leave routing short
  of what its consumer needs. Trained on kai's 5090 against the default
  14B, which stays the default. See [fine-tuning.md](fine-tuning.md) for
  the gate, method and data.

## Later / Ideas

- Batch a call's choices into one eval (parallel sequences). The prompt's
  KV is already reused, so this is latency only — but for route it is now
  most of the latency: 46 per-choice evals put p50 at 1.4 s (severity and
  triage, 3 choices, ~95 ms). Do it when a consumer's measured load asks
  for it.
- Mail/notification triage as a consumer (98% on synthetic items).
- Log-prob caching for repeated prompts, if a consumer repeats them.

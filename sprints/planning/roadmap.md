# Roadmap

> The general plan for this project. Keep it current; detail lives in the
> sprint records.

## Done

- **001 — scoring core + model bake-off.** `score(prompt, choices)` over
  llama-cpp-python, a 154-item eval set, and a four-model bake-off on kubs0.
  Default model: **Qwen3-14B Q4_K_M** (11.6 GiB peak beside TEI); fallback
  8B Q6_K. Raw probabilities are overconfident (T≈10). See
  `sprints/001-scoring-core-and-bake-off.md`.

## Next

- **Calibration.** Per-task temperature scaling fitted on the eval set —
  001 measured a single T recovering ~⅓ of the Brier score.
- **002 — service.** HTTP API (prompt + choices → probabilities) as a
  systemd user service on kubs0, tailnet-served; model kept resident.
- Batch a call's choices into one eval (as parallel sequences) — 001
  already reuses the prompt's KV across choices, one eval per choice.

## Later / Ideas

- First consumers: korg WI → project routing, kmon finding severity,
  mail/notification triage.
- Richer routing input (WI content, fuller project contracts, few-shot) —
  title-only routing measured 47% at best in 001.
- MCP surface so agents can call it directly.

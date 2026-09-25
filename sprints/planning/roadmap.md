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

- **003 — service + MCP.** One process on kubs0 with the 14B resident
  (10.5 GiB beside TEI) serves the tasks over HTTP (`/v1/classify`,
  `/v1/score`, `/healthz`) and MCP (`/mcp`) at
  `https://kubs0.encke-wahoo.ts.net:7780`. Every result carries
  `calibrated`. Deployed from merged main by the `deploy-kodds` skill
  (release dirs, `current`/`previous`). Route's overlay is pinned in a
  state dir (`docs/route-overlay.md`). Warm latency: 93 ms
  severity/triage, 1.45 s route. See `sprints/003-service-and-mcp.md`.

## Next

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
- Per-task KV state for the resident service. The prefix cache keeps only
  the previous prompt's prefix, so route called right after severity or
  triage re-prefills its ~7k-token project list: 5.3 s vs 1.45 s warm
  (003's measurement). Save and restore a llama.cpp state per task when a
  consumer's call mix interleaves tasks enough to matter.
- Mail/notification triage as a consumer (98% on synthetic items).
- Log-prob caching for repeated prompts, if a consumer repeats them.
- **Diversion (low priority): Laya bake-off.** Not a planned direction for
  kodds, just a comparison that might be interesting.
  [Laya](https://brainfunctioncollapse.com/laya/about)
  ([repo](https://github.com/NandhaKishorM/laya),
  [weights](https://huggingface.co/convaiinnovations/laya), Apache-2.0) is a
  421M ModernBERT-large encoder with a decision head. It scores each option
  at its own `[MASK]` slot and returns probabilities, with no generation.
  Its authors claim ~35 ms per question on a T4, and ECE 0.081 after a
  per-question-type temperature refit. They also report it ships
  over-confident and is near chance zero-shot (0.36) until fine-tuned
  (0.766). It fits easily on kubs0: ~1–2 GB beside TEI, and it can also run
  on CPU. It doesn't suit route as it stands. Its context is 512 tokens
  (1,024 for the multilingual checkpoint), and route's prompt is ~7.3k
  tokens over 46 options. If picked up: add a Laya backend and run
  severity/triage on the existing cal/test splits, then maybe one
  fine-tune on route's train split against a shortlist. Compare accuracy,
  Brier/ECE and latency with the 14B.

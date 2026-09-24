# Roadmap

> The general plan for this project. Keep it current; detail lives in the
> sprint records.

## Now

- **001 — scoring core + model bake-off.** `score(prompt, choices)` over
  llama-cpp-python (multi-token choices, thinking off), a labelled eval set
  of ~150 homelab-shaped items, and a bake-off of Qwen3-4B Q8_0 / 8B Q6_K /
  8B Q8_0 / 14B Q4_K_M on kubs0 alongside the klams embedders: peak VRAM,
  latency, accuracy, calibration. Ends in a decision doc naming the default.

## Next

- **002 — service.** HTTP API (prompt + choices → probabilities) as a
  systemd user service on kubs0, tailnet-served; model kept resident.
- Batching / KV-prefix reuse so N choices share one prompt evaluation.

## Later / Ideas

- First consumers: korg WI → project routing, kmon finding severity,
  mail/notification triage.
- Calibration step (temperature scaling per task) if raw probabilities are
  off.
- MCP surface so agents can call it directly.

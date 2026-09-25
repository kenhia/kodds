# kodds

A local LLM classifier for the homelab. Give it a prompt and a set of
choices; it returns a probability for each choice, read straight from the
model's next-token logits — no generation, no API calls, data never leaves
the box. Inspired by the "Jev" idea (and the parody
[Jev in 25 lines](https://www.nobodywho.ai/posts/jev-in-25-lines/)).

Runs on kubs0 with a quantised Qwen3 GGUF via `llama-cpp-python`, alongside
the klams embedding servers. Not general model serving — that is `kvllm`.

How the default model was chosen — four Qwen3 builds compared on accuracy,
calibration and VRAM: [the bake-off infographic](https://raw.githack.com/kenhia/kodds/main/docs/bakeoff-001.html)
(source: [`docs/bakeoff-001.html`](docs/bakeoff-001.html)).

## Usage

```python
from kodds.llama import load

scorer = load("~/models/gguf/Qwen3-14B-Q4_K_M.gguf")
scorer.score("Is this email phishing? ...", ["legitimate", "spam", "phishing"])
# {'legitimate': 0.01, 'spam': 0.03, 'phishing': 0.96}
```

Each choice is scored as the summed log-prob of all its tokens plus the
end-of-turn token, so multi-token choices and choices sharing a prefix are
compared fairly. Raw probabilities are overconfident — see sprint 001.

**Tasks** are the calibrated unit: a prompt template, a fixed choice list and
per-model calibration, defined in [`tasks/`](tasks/).

```python
from kodds.tasks import load_tasks

tasks = load_tasks()  # route, severity, triage
result = tasks["severity"].classify(scorer, finding="kubsdb: root fs at 97%")
result.probs  # calibrated for this model, if it has a fit
result.calibrated  # False → raw probabilities (no fit for this model/prompt)
```

A calibration applies only to the model file, choice list and exact prompt
it was fitted on; otherwise `classify` returns raw probabilities and says
so. Routing's enriched project descriptions come from private korg data and
live in a git-ignored overlay (`evals/build_evalset.py overlay`); without
the overlay, `route` runs uncalibrated. See sprint 002 and
[`evals/README.md`](evals/README.md).

`llama-cpp-python` must be built with CUDA or it runs on the CPU:
`CMAKE_ARGS="-DGGML_CUDA=on" uv sync --reinstall-package llama-cpp-python`.
`just models` downloads the candidate GGUFs to `~/models/gguf`; `just
bakeoff` re-runs the model comparison.

## Development

This repo uses the [kprojects](https://github.com/kenhia/kprojects) harness:
`just` lists recipes, `just check` runs the gates (ruff, ty, pytest) with
no model needed; `just test-gpu` runs the GPU tests. The
plan lives in `sprints/planning/roadmap.md`.

## License

MIT — see [LICENSE](LICENSE).

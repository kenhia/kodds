# kodds

A local LLM classifier for the homelab. Give it a prompt and a set of
choices; it returns a probability for each choice, read straight from the
model's next-token logits — no generation, no API calls, data never leaves
the box. Inspired by the "Jev" idea (and the parody
[Jev in 25 lines](https://www.nobodywho.ai/posts/jev-in-25-lines/)).

Runs on kubs0 with a quantised Qwen3 GGUF via `llama-cpp-python`, alongside
the klams embedding servers. Not general model serving — that is `kvllm`.

## Development

This repo uses the [kprojects](https://github.com/kenhia/kprojects) harness:
`just` lists recipes, `just check` runs the gates (ruff, ty, pytest). The
plan lives in `sprints/planning/roadmap.md`.

## License

MIT — see [LICENSE](LICENSE).

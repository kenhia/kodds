# Windows: `just` runs recipes through `sh`, which Windows does not ship — put
# Git for Windows' `usr\bin` on PATH (it holds `sh.exe`) or run from Git Bash.
# (Upstream's own requirement: "sh must be available in the PATH".)

models_dir := env('KODDS_MODELS', home_directory() / 'models' / 'gguf')

# List available recipes
default:
    @just --list

# Run CI gates (lint, typecheck, tests)
check:
    uv run ruff format --check .
    uv run ruff check .
    uv run ty check
    uv run pytest

# Apply formatting and safe autofixes
fmt:
    uv run ruff format .
    uv run ruff check --fix .

# GPU tests against a real GGUF (skipped by `check`)
test-gpu model=(models_dir / "Qwen3-14B-Q4_K_M.gguf"):
    KODDS_MODEL={{model}} uv run pytest -m gpu

# Download the bake-off candidates into the models dir (outside the repo)
models:
    mkdir -p {{models_dir}}
    for f in Qwen3-4B/Qwen3-4B-Q8_0 Qwen3-8B/Qwen3-8B-Q6_K Qwen3-8B/Qwen3-8B-Q8_0 Qwen3-14B/Qwen3-14B-Q4_K_M; do \
        curl -fSL -C - -o {{models_dir}}/${f##*/}.gguf "https://huggingface.co/Qwen/${f%%/*}-GGUF/resolve/main/${f##*/}.gguf"; \
    done

# Run the eval set against every downloaded candidate, one process each
bakeoff:
    for m in {{models_dir}}/Qwen3-*.gguf; do uv run python evals/bakeoff.py "$m"; done
    uv run python evals/report.py

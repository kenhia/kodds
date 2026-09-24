# Windows: `just` runs recipes through `sh`, which Windows does not ship — put
# Git for Windows' `usr\bin` on PATH (it holds `sh.exe`) or run from Git Bash.
# (Upstream's own requirement: "sh must be available in the PATH".)

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

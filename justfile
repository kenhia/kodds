# Windows: `just` runs recipes through `sh`, which Windows does not ship — put
# Git for Windows' `usr\bin` on PATH (it holds `sh.exe`) or run from Git Bash.
# (Upstream's own requirement: "sh must be available in the PATH".)

models_dir := env('KODDS_MODELS', home_directory() / 'models' / 'gguf')

# llama-cpp-python is built with CUDA (pyproject's extra-build-variables), and
# CMake needs nvcc to do it. Non-interactive shells (karc legs, systemd) don't
# get ~/.bashrc's PATH, so put the toolkit on it here.
export PATH := "/usr/local/cuda/bin:" + env('PATH')

# List available recipes
default:
    @just --list

# Run CI gates (lint, typecheck, tests)
check:
    uv run ruff format --check .
    uv run ruff check .
    uv run ty check
    uv run pytest

# Sync the venv (llama-cpp-python built with CUDA) and prove GPU offload works
setup:
    uv sync
    uv run python -c "import llama_cpp, sys; sys.exit(0 if llama_cpp.llama_supports_gpu_offload() else 'llama-cpp-python is a CPU build')"

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

# Rebuild the eval splits (route's text is fetched from korg into .scratch/)
evalset:
    uv run python evals/build_evalset.py

# Score a task's split under prompt variants (GPU; e.g. `just sweep route cal content`)
sweep task split +variants:
    uv run python evals/sweep.py {{models_dir}}/Qwen3-14B-Q4_K_M.gguf {{task}} {{split}} {{variants}}

# Compare calibration methods on the swept log-probs (no GPU); `--write <model>` saves the winners
calibrate *args:
    uv run python evals/calibrate.py {{args}}

state_dir := home_directory() / '.local' / 'share' / 'kodds'
default_model := 'Qwen3-14B-Q4_K_M.gguf'

# Regenerate route's private overlay in tasks/private/ from korg (a REFIT step — see docs/route-overlay.md)
overlay:
    @echo "Regenerating tasks/private/route.json. This changes route's prompt hash:"
    @echo "the committed calibration stops applying until you refit (docs/route-overlay.md)."
    uv run python evals/build_evalset.py overlay

# Install tasks/private/route.json as the service's pinned overlay, only if the committed calibration matches it
overlay-install model=default_model:
    #!/usr/bin/env bash
    set -euo pipefail
    SRC=tasks/private/route.json
    [ -f "$SRC" ] || { echo "no $SRC — run 'just overlay' (needs korg) first" >&2; exit 1; }
    STAGE=$(mktemp -d); trap 'rm -rf "$STAGE"' EXIT
    cp "$SRC" "$STAGE/route.json"
    uv run python -c "
    import sys
    from kodds.tasks import load_tasks
    route = load_tasks(private_dir='$STAGE')['route']
    if route.calibration_for('{{model}}') is None:
        sys.exit('refusing: route would be UNcalibrated with this overlay (prompt hash '
                 + route.prompt_hash()[:12] + ' has no fit for {{model}}). See docs/route-overlay.md.')
    print('route calibrated with this overlay, prompt hash', route.prompt_hash()[:12])
    "
    mkdir -p {{state_dir}}/private
    [ -f {{state_dir}}/private/route.json ] && cp -f {{state_dir}}/private/route.json {{state_dir}}/private/route.json.prev
    install -m 600 "$SRC" {{state_dir}}/private/route.json
    echo "installed {{state_dir}}/private/route.json; restart the service to pick it up (systemctl --user restart kodds)"

# Deploy HEAD to kubs0: export a release, sync it (CUDA), flip `current`, restart, wait for the model to load
deploy:
    #!/usr/bin/env bash
    set -euo pipefail
    [ "$(hostname -s)" = "kubs0" ] || { echo "kodds is deployed on kubs0; this is $(hostname -s)." >&2; exit 1; }
    # Dirty bytes correspond to no commit, so the release would be unreproducible.
    [ -z "$(git status --short)" ] || { echo "working tree is dirty — commit first." >&2; git status --short >&2; exit 1; }
    BRANCH=$(git rev-parse --abbrev-ref HEAD)
    [ "$BRANCH" = "main" ] || echo "WARNING: deploying from '$BRANCH', not main. What ships should be what landed."
    SHA=$(git rev-parse HEAD)
    STATE={{state_dir}}
    REL=$STATE/releases/$SHA
    mkdir -p "$STATE/releases" "$STATE/private"

    # The venv records absolute paths, so it is built in place; `.complete`
    # marks a release that finished building.
    if [ ! -f "$REL/.complete" ]; then
        rm -rf "$REL"; mkdir -p "$REL"
        git archive "$SHA" | tar -x -C "$REL"
        echo "$SHA" > "$REL/COMMIT"
        (cd "$REL" && uv sync --frozen --no-dev)
        gpu() { "$REL/.venv/bin/python" -c "import llama_cpp, sys; sys.exit(0 if llama_cpp.llama_supports_gpu_offload() else 1)" 2>/dev/null; }
        if ! gpu; then
            echo "release got a CPU llama-cpp-python; rebuilding with CUDA"
            (cd "$REL" && uv sync --frozen --no-dev --reinstall-package llama-cpp-python)
            gpu || { echo "llama-cpp-python still has no GPU offload; not deploying." >&2; exit 1; }
        fi
        touch "$REL/.complete"
    fi
    echo "release $REL (GPU offload ok)"

    if [ -f "$STATE/private/route.json" ]; then
        echo "route overlay: $STATE/private/route.json (pinned; never regenerated by deploy)"
    else
        echo "WARNING: no route overlay in $STATE/private — route will serve UNcalibrated. See docs/route-overlay.md ('just overlay-install')."
    fi

    OLD=$(readlink "$STATE/current" 2>/dev/null || true)
    if [ -n "$OLD" ] && [ "$OLD" != "$REL" ]; then
        ln -sfn "$OLD" "$STATE/previous"
    fi
    ln -sfn "$REL" "$STATE/current"

    mkdir -p ~/.config/systemd/user
    install -m 644 deploy/kodds.service ~/.config/systemd/user/kodds.service
    systemctl --user daemon-reload
    systemctl --user enable kodds.service >/dev/null 2>&1
    systemctl --user reset-failed kodds.service 2>/dev/null || true
    systemctl --user restart kodds.service

    tailscale serve status 2>/dev/null | grep -q ':7780' \
        || tailscale serve --bg --https=7780 http://localhost:7780

    echo "waiting for the model to load..."
    for _ in $(seq 1 120); do
        if H=$(curl -fsS --max-time 5 http://127.0.0.1:7780/healthz 2>/dev/null); then break; fi
        systemctl --user is-failed --quiet kodds.service && {
            echo "kodds failed to start:" >&2; journalctl --user -u kodds -n 30 --no-pager >&2; exit 1; }
        H=""; sleep 2
    done
    [ -n "$H" ] || { echo "no /healthz after 240s" >&2; journalctl --user -u kodds -n 30 --no-pager >&2; exit 1; }
    echo "$H" | python3 -c "
    import json, sys
    h = json.load(sys.stdin)
    print('model', h['model'], '| gpu_offload', h['gpu_offload'], '| commit', h['commit'], '| vram', h['vram_mib'])
    for n, t in h['tasks'].items():
        print(f\"  {n}: calibrated={t['calibrated']} overlay={t['overlay']}\")
    log = h.get('call_log') or {}
    print('call log', log.get('path'), '| writable', log.get('writable'))
    if not log.get('writable'):
        print('WARNING: the call log is not writable; classify still serves, but request ids will not join to anything', file=sys.stderr)
    ok = h['loaded'] and h['gpu_offload'] and h['commit'] == '$SHA'
    sys.exit(0 if ok else 'healthz does not show this release loaded on the GPU')
    "

    # Keep only what is live and the rollback target.
    for d in "$STATE"/releases/*; do
        [ "$d" = "$REL" ] || [ "$d" = "$(readlink "$STATE/previous" 2>/dev/null)" ] || rm -rf "$d"
    done
    echo "deployed $SHA"

# Point `current` back at the previous release and restart
rollback:
    #!/usr/bin/env bash
    set -euo pipefail
    STATE={{state_dir}}
    PREV=$(readlink "$STATE/previous") || { echo "no previous release" >&2; exit 1; }
    CUR=$(readlink "$STATE/current")
    ln -sfn "$PREV" "$STATE/current"; ln -sfn "$CUR" "$STATE/previous"
    systemctl --user restart kodds.service
    echo "current -> $PREV (was $CUR); watch: curl -s http://127.0.0.1:7780/healthz"

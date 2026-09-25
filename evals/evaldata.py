"""Where the eval splits live, and reading them back.

severity/triage are synthetic and committed under ``evals/evalset/``; route is
real korg content, so it is fetched into git-ignored ``.scratch/evalset/`` by
``build_evalset.py`` and only its split assignment is committed.
"""

import json
from pathlib import Path

HERE = Path(__file__).parent
SPLITS = HERE / "splits"
PUBLIC = HERE / "evalset"
PRIVATE = HERE.parent / ".scratch" / "evalset"


def load(task: str, split: str) -> list[dict]:
    for directory in (PUBLIC, PRIVATE):
        path = directory / f"{task}.{split}.jsonl"
        if path.exists():
            return [json.loads(line) for line in path.read_text().splitlines()]
    raise FileNotFoundError(
        f"no {task}.{split}.jsonl — run `uv run python evals/build_evalset.py`"
    )

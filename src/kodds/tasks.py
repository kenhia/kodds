"""Named tasks: the unit kodds serves.

A task is a prompt template, a fixed list of choices and — per model file —
the calibration fitted for it. ``task.classify(scorer, **inputs)`` renders
the prompt, scores the choices and applies the loaded model's calibration.

A calibration is only valid for the model, prompt and choice list it was
fitted on, so each one records the choices and a hash of the rendered prompt.
When the loaded model has no fit, or the choices or the prompt have changed
since (routing's project list does), ``classify`` returns the raw
probabilities with ``calibrated=False`` rather than borrowing parameters
fitted for something else.

Definitions live in ``tasks/<name>.toml`` (hand-written) and fitted
parameters in ``tasks/<name>.calibration.json`` (written by
``evals/calibrate.py``), both small and diffable. A task whose choice
descriptions can't be public (route's are built from private korg content)
takes them from a git-ignored overlay, ``tasks/private/<name>.json``; on a
host without the overlay the prompt differs, and its hash says so.
"""

import hashlib
import json
import string
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from kodds import normalize
from kodds.scorer import Scorer

TASKS_DIR = Path(__file__).resolve().parents[2] / "tasks"

# The filler a content-free prompt uses in place of every input
# (contextual calibration: Zhao et al. 2021, "Calibrate Before Use").
CONTENT_FREE = "N/A"


@dataclass(frozen=True)
class Calibration:
    """``softmax((logprob + bias[choice]) / temperature)`` over the choices."""

    temperature: float = 1.0
    bias: Mapping[str, float] = field(default_factory=dict)
    choices: tuple[str, ...] = ()
    method: str = "none"
    provenance: Mapping[str, Any] = field(default_factory=dict)
    prompt_sha256: str | None = None
    """``Task.prompt_hash()`` at fit time; None pins nothing (hand-written)."""

    def apply(self, logprobs: Mapping[str, float]) -> dict[str, float]:
        return normalize(
            {
                c: (lp + self.bias.get(c, 0.0)) / self.temperature
                for c, lp in logprobs.items()
            }
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "temperature": self.temperature,
            "bias": dict(self.bias),
            "choices": list(self.choices),
            "provenance": dict(self.provenance),
            "prompt_sha256": self.prompt_sha256,
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> "Calibration":
        return cls(
            temperature=float(data.get("temperature", 1.0)),
            bias={c: float(b) for c, b in data.get("bias", {}).items()},
            choices=tuple(data.get("choices", ())),
            method=data.get("method", "none"),
            provenance=data.get("provenance", {}),
            prompt_sha256=data.get("prompt_sha256"),
        )


@dataclass(frozen=True)
class Result:
    probs: dict[str, float]
    """Calibrated when ``calibrated`` is true, otherwise the same as ``raw``."""
    raw: dict[str, float]
    logprobs: dict[str, float]
    calibrated: bool
    model: str | None


@dataclass(frozen=True)
class Task:
    name: str
    template: str
    """``str.format`` text: the task's inputs by name, plus ``{choice_list}``."""
    choices: tuple[str, ...]
    descriptions: Mapping[str, str] = field(default_factory=dict)
    system: str | None = None
    limits: Mapping[str, int] = field(default_factory=dict)
    """Per-input character cap; longer inputs are cut and marked with ``…``."""
    calibrations: Mapping[str, Calibration] = field(default_factory=dict)
    """Keyed by model file name, e.g. ``Qwen3-14B-Q4_K_M.gguf``."""

    @property
    def inputs(self) -> list[str]:
        names = [f for _, f, _, _ in string.Formatter().parse(self.template) if f]
        return [n for n in dict.fromkeys(names) if n != "choice_list"]

    def choice_list(self) -> str:
        return "\n".join(
            f"- {c}: {self.descriptions[c]}" if c in self.descriptions else f"- {c}"
            for c in self.choices
        )

    def render(self, **inputs: str) -> str:
        wanted = self.inputs
        missing = [n for n in wanted if n not in inputs]
        extra = [n for n in inputs if n not in wanted]
        if missing or extra:
            raise ValueError(
                f"task {self.name!r}: missing inputs {missing}, unknown inputs {extra}"
            )
        fields = {n: self._limit(n, inputs[n]) for n in wanted}
        return self.template.format(choice_list=self.choice_list(), **fields)

    def content_free(self) -> str:
        """The prompt with every input replaced by ``N/A``."""
        return self.render(**dict.fromkeys(self.inputs, CONTENT_FREE))

    def prompt_hash(self) -> str:
        """Identifies the prompt as the model sees it, less the inputs."""
        text = f"{self.system or ''}\0{self.content_free()}"
        return hashlib.sha256(text.encode()).hexdigest()

    def calibration_for(self, model: str | None) -> Calibration | None:
        cal = self.calibrations.get(model) if model else None
        if cal is None or cal.choices != self.choices:
            return None
        if cal.prompt_sha256 is not None and cal.prompt_sha256 != self.prompt_hash():
            return None
        return cal

    def classify(self, scorer: Scorer, **inputs: str) -> Result:
        logprobs = scorer.logprobs(self.render(**inputs), self.choices, self.system)
        raw = normalize(logprobs)
        cal = self.calibration_for(scorer.model)
        return Result(
            probs=cal.apply(logprobs) if cal else raw,
            raw=raw,
            logprobs=logprobs,
            calibrated=cal is not None,
            model=scorer.model,
        )

    def _limit(self, name: str, value: str) -> str:
        cap = self.limits.get(name)
        return value[:cap] + "…" if cap is not None and len(value) > cap else value


def load_task(
    path: str | Path, *, overlay: bool = True, private_dir: str | Path | None = None
) -> Task:
    """A task from its TOML, with fitted params from a sibling ``.calibration.json``.

    With ``overlay``, choice descriptions in ``<private_dir>/<name>.json``
    (default: ``private/`` beside the TOML), when present, replace the TOML's.
    The service points ``private_dir`` at a state dir outside the release.
    """
    path = Path(path)
    spec = tomllib.loads(path.read_text())
    choices = spec["choices"]
    if isinstance(choices, Mapping):
        names, descriptions = tuple(choices), dict(choices)
    else:
        names, descriptions = tuple(choices), {}
    private = overlay_path(path, private_dir)
    if overlay and private.exists():
        extra = json.loads(private.read_text())["descriptions"]
        unknown = set(extra) - set(names)
        if unknown:
            raise ValueError(f"{private}: descriptions for unknown choices {unknown}")
        descriptions |= extra
    cal_path = path.with_name(f"{path.stem}.calibration.json")
    fitted = json.loads(cal_path.read_text()) if cal_path.exists() else {}
    return Task(
        name=spec["name"],
        template=spec["template"],
        choices=names,
        descriptions=descriptions,
        system=spec.get("system"),
        limits=spec.get("limits", {}),
        calibrations={m: Calibration.from_json(c) for m, c in fitted.items()},
    )


def overlay_path(path: str | Path, private_dir: str | Path | None = None) -> Path:
    """Where the private overlay for the task defined at ``path`` lives."""
    path = Path(path)
    return Path(private_dir or path.parent / "private") / f"{path.stem}.json"


def load_tasks(
    directory: str | Path = TASKS_DIR,
    *,
    overlay: bool = True,
    private_dir: str | Path | None = None,
) -> dict[str, Task]:
    paths = sorted(Path(directory).glob("*.toml"))
    tasks = (load_task(p, overlay=overlay, private_dir=private_dir) for p in paths)
    return {t.name: t for t in tasks}


def save_calibration(
    task: Task, model: str, cal: Calibration, directory: str | Path = TASKS_DIR
) -> Path:
    """Record ``cal`` as ``task``'s fit for ``model``, keeping other models' fits."""
    path = Path(directory) / f"{task.name}.calibration.json"
    fitted = json.loads(path.read_text()) if path.exists() else {}
    fitted[model] = cal.to_json()
    path.write_text(json.dumps(fitted, indent=1, sort_keys=True) + "\n")
    return path

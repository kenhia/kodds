import json
import math
from dataclasses import replace

import pytest
from test_scorer import BigramBackend, plain

from kodds.scorer import Scorer
from kodds.tasks import Calibration, Task, load_task, load_tasks


def spam_model() -> BigramBackend:
    # After ":" the model likes "s" (spam) over "h" (ham), about e^2 : 1.
    return BigramBackend(
        {
            ":": {"s": 2.0, "h": 0.0},
            "s": {"p": 5.0},
            "p": {"a": 5.0},
            "a": {"m": 5.0},
            "m": {"$": 5.0},
            "h": {"a": 5.0},
        }
    )


def spam_task(**kw) -> Task:
    task = Task(
        name="spam", template="{message}\n{choice_list}\n:", choices=("spam", "ham")
    )
    return replace(task, **kw)


def test_render_fills_inputs_and_describes_choices():
    task = spam_task(descriptions={"spam": "junk", "ham": "real mail"})
    assert task.render(message="hi") == "hi\n- spam: junk\n- ham: real mail\n:"


def test_render_lists_bare_names_without_descriptions():
    assert spam_task().render(message="hi") == "hi\n- spam\n- ham\n:"


def test_render_truncates_limited_inputs():
    task = spam_task(template="{message}", limits={"message": 5})
    assert task.render(message="abcdefgh") == "abcde…"
    assert task.render(message="abc") == "abc"


def test_render_rejects_missing_and_unknown_inputs():
    with pytest.raises(ValueError, match="message"):
        spam_task().render()
    with pytest.raises(ValueError, match="extra"):
        spam_task().render(message="hi", extra="x")


def test_content_free_blanks_every_input():
    task = spam_task(template="{message} / {sender}")
    assert task.content_free() == "N/A / N/A"


def test_calibration_applies_bias_then_temperature():
    cal = Calibration(temperature=2.0, bias={"a": -1.0}, choices=("a", "b"))
    probs = cal.apply({"a": 1.0, "b": 0.0})
    # (1 - 1) / 2 = 0 vs 0 / 2 = 0: an even split.
    assert math.isclose(probs["a"], 0.5)
    assert math.isclose(sum(probs.values()), 1.0)


def test_identity_calibration_leaves_probabilities_alone():
    lp = {"a": math.log(0.8), "b": math.log(0.2)}
    probs = Calibration(choices=("a", "b")).apply(lp)
    assert math.isclose(probs["a"], 0.8)


def test_classify_applies_the_loaded_models_calibration():
    cal = Calibration(temperature=4.0, choices=("spam", "ham"))
    task = spam_task(template=":", calibrations={"m.gguf": cal})
    scorer = Scorer(spam_model(), template=plain, model="m.gguf")
    result = task.classify(scorer)
    assert result.calibrated
    assert result.model == "m.gguf"
    assert result.probs == cal.apply(result.logprobs)
    # Tempering pulls the confident raw answer towards even.
    assert 0.5 < result.probs["spam"] < result.raw["spam"]


def test_classify_falls_back_to_raw_for_an_uncalibrated_model():
    cal = Calibration(temperature=4.0, choices=("spam", "ham"))
    task = spam_task(template=":", calibrations={"other.gguf": cal})
    result = task.classify(Scorer(spam_model(), template=plain, model="m.gguf"))
    assert not result.calibrated
    assert result.probs == result.raw


def test_classify_falls_back_when_the_choice_list_changed():
    # Fitted when the task had a third choice: its params no longer apply.
    cal = Calibration(temperature=4.0, choices=("spam", "ham", "eggs"))
    task = spam_task(template=":", calibrations={"m.gguf": cal})
    result = task.classify(Scorer(spam_model(), template=plain, model="m.gguf"))
    assert not result.calibrated


def test_classify_falls_back_when_the_prompt_changed():
    task = spam_task(template=":")
    pinned = Calibration(
        temperature=4.0, choices=("spam", "ham"), prompt_sha256=task.prompt_hash()
    )
    scorer = Scorer(spam_model(), template=plain, model="m.gguf")
    assert replace(task, calibrations={"m.gguf": pinned}).classify(scorer).calibrated
    # Same choices, different wording: the fit no longer applies.
    reworded = replace(task, template=": ", calibrations={"m.gguf": pinned})
    assert not reworded.classify(scorer).calibrated


def test_prompt_hash_covers_system_and_descriptions():
    task = spam_task()
    assert task.prompt_hash() != replace(task, system="x").prompt_hash()
    described = replace(task, descriptions={"spam": "junk"})
    assert task.prompt_hash() != described.prompt_hash()


def test_classify_passes_the_system_prompt_through():
    seen = []

    def template(prompt, system=None):
        seen.append(system)
        return prompt

    task = spam_task(template=":", system="be terse")
    task.classify(Scorer(spam_model(), template=template))
    assert seen == ["be terse"]


def test_load_task_reads_toml_and_sibling_calibrations(tmp_path):
    (tmp_path / "spam.toml").write_text(
        'name = "spam"\n'
        'system = "You sort mail."\n'
        'template = """Message: {message}\n{choice_list}"""\n'
        "[limits]\nmessage = 100\n"
        '[choices]\nspam = "junk"\nham = "real mail"\n'
    )
    (tmp_path / "spam.calibration.json").write_text(
        json.dumps(
            {
                "m.gguf": {
                    "method": "temperature",
                    "temperature": 3.0,
                    "bias": {},
                    "choices": ["spam", "ham"],
                    "provenance": {"split": "cal", "n": 10},
                }
            }
        )
    )
    task = load_task(tmp_path / "spam.toml")
    assert task.choices == ("spam", "ham")
    assert task.system == "You sort mail."
    assert task.limits == {"message": 100}
    cal = task.calibration_for("m.gguf")
    assert cal is not None and cal.temperature == 3.0
    assert cal.provenance["split"] == "cal"
    assert load_tasks(tmp_path) == {"spam": task}


def test_private_overlay_replaces_descriptions(tmp_path):
    (tmp_path / "spam.toml").write_text(
        'name = "spam"\ntemplate = "{choice_list}"\n[choices]\nspam = "junk"\nham = "mail"\n'
    )
    (tmp_path / "private").mkdir()
    overlay = tmp_path / "private" / "spam.json"
    overlay.write_text(json.dumps({"descriptions": {"spam": "junk, e.g. 'WIN'"}}))
    task = load_task(tmp_path / "spam.toml")
    assert task.descriptions == {"spam": "junk, e.g. 'WIN'", "ham": "mail"}
    assert (
        load_task(tmp_path / "spam.toml", overlay=False).descriptions["spam"] == "junk"
    )
    overlay.write_text(json.dumps({"descriptions": {"eggs": "?"}}))
    with pytest.raises(ValueError, match="eggs"):
        load_task(tmp_path / "spam.toml")


def test_overlay_can_live_outside_the_task_dir(tmp_path):
    # The service keeps overlays in a state dir that outlives each release.
    (tmp_path / "spam.toml").write_text(
        'name = "spam"\ntemplate = "{choice_list}"\n[choices]\nspam = "junk"\n'
    )
    state = tmp_path / "state"
    state.mkdir()
    (state / "spam.json").write_text(json.dumps({"descriptions": {"spam": "WIN"}}))
    assert load_task(tmp_path / "spam.toml").descriptions["spam"] == "junk"
    tasks = load_tasks(tmp_path, private_dir=state)
    assert tasks["spam"].descriptions["spam"] == "WIN"


def test_repo_tasks_load_and_render():
    tasks = load_tasks()
    assert {"route", "severity", "triage"} <= set(tasks)
    for task in tasks.values():
        assert task.content_free()
        assert len(set(task.choices)) == len(task.choices)

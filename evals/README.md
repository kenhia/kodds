# evals

Labelled data and the scripts that measure kodds against it.

| path | what |
|---|---|
| `build_evalset.py` | the reviewable source of every split (see its docstring) |
| `splits/route.json` | route's fixed split: `[wi_number, project, split]` per korg work item |
| `evalset/{severity,triage}.{cal,test}.jsonl` | the synthetic tasks' rows, committed |
| `evaldata.py` | `load(task, split)` for the scripts below |
| `sweep.py` | scores one task's split under prompt variants; dumps log-probs |
| `calibrate.py` | compares calibration methods (fit on `cal`, report on `test`) and writes `tasks/*.calibration.json` |
| `bakeoff.py`, `report.py`, `evalset.jsonl`, `results/` | sprint 001's model bake-off and its frozen 154-item set |

Splits: `cal` fits calibration and chooses prompts; `test` only reports;
route's `train` is held in reserve for fine-tuning and supplies few-shot
examples. None is refitted once fixed.

## The routing corpus is not in this repo

Route is scored on real work items from Ken's korg (title + content), which
describe a private homelab. kodds is public, so only the split *assignment*
is committed; `build_evalset.py` fetches the text from korg into the
git-ignored `.scratch/evalset/` when it runs, which needs korg on the
tailnet.

The full corpus — the 1,611 items behind `splits/route.json`, as JSONL — is
available on request: open an issue on this repo.

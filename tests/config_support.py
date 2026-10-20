"""Helpers for tests that perturb a COPY of the eval config (never the real files).

make_pair writes temporary copies of the freeze file and the run config under tmp_path and recomputes the
run config's freeze pin, so every perturbation test passes the pin check unless it wants to fail it."""
from __future__ import annotations

import copy
import json
from pathlib import Path

from vera.eval_config import FREEZE_PATH, MODEL_SELECTION_PATH, RUN_PATH, file_sha256, load_eval_config, requirements_hash16


def real_freeze() -> dict:
    return json.loads(FREEZE_PATH.read_text())


def real_run() -> dict:
    return json.loads(RUN_PATH.read_text())


def make_pair(tmp_path: Path, freeze_edit=None, run_edit=None, *, repin: bool = True):
    """-> (freeze_path, run_path). Edits are callables mutating the parsed dict in place."""
    f, r = copy.deepcopy(real_freeze()), copy.deepcopy(real_run())
    if freeze_edit:
        freeze_edit(f)
    fp, rp = Path(tmp_path) / "freeze.json", Path(tmp_path) / "run.json"
    fb = json.dumps(f, indent=2, ensure_ascii=False).encode("utf-8")
    fp.write_bytes(fb)
    if repin:
        r["freeze_pin"]["file_sha256"] = file_sha256(fb)
        r["freeze_pin"]["requirements_hash16"] = requirements_hash16(f)
    if run_edit:
        run_edit(r)
    rp.write_text(json.dumps(r, indent=2, ensure_ascii=False))
    return fp, rp


def load_pair(tmp_path: Path, freeze_edit=None, run_edit=None, *, model_selection_path=MODEL_SELECTION_PATH,
              repin: bool = True):
    fp, rp = make_pair(tmp_path, freeze_edit, run_edit, repin=repin)
    return load_eval_config(fp, rp, model_selection_path=model_selection_path)


def add_models(models: dict):
    """run_edit adding model_families entries {model: family}."""
    def edit(r):
        for m, fam in models.items():
            r["model_families"][m] = {"family": fam, "source": "https://example.invalid/test-only"}
    return edit

"""scripts/trace_validate.py: confusion counts and label-to-check mapping (synthetic data only)."""
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location("trace_validate", Path(__file__).resolve().parents[1] / "scripts/trace_validate.py")
tv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tv)


def test_confusion_counts_and_rates():
    d = tv.confusion([(True, True), (True, False), (False, False), (False, True), (False, False)])
    assert (d["tp"], d["fn"], d["tn"], d["fp"]) == (1, 1, 2, 1)
    assert d["tpr"] == 0.5 and d["tnr"] == round(2 / 3, 4)


def test_undefined_rate_when_class_absent():
    assert tv.confusion([(False, False)])["tpr"] is None


def test_validate_maps_criteria_and_skips_non_applicable():
    results = {"checks_version": "v", "traces": [
        {"id": "q01", "variant": "baseline", "split": "dev", "checks": [
            {"name": "A1_citation_present", "applies": True, "passed": False},
            {"name": "A7_premise_corrected", "applies": False, "passed": None}]},
        {"id": "q30", "variant": "baseline", "split": "heldout", "checks": [
            {"name": "A1_citation_present", "applies": True, "passed": True}]}]}
    labels = {"labeller": "x", "items": [
        {"trace": "q01", "variant": "baseline", "labels": {"C1": {"label": "FAIL"}, "C7": {"label": "PASS"}}},
        {"trace": "q30", "variant": "baseline", "labels": {"C1": {"label": "PASS"}}}]}
    out = tv.validate(results, labels, {"labeller": "y", "items": []}, [])
    assert out["per_check"]["A1_citation_present"]["tp"] == 1 and out["per_check"]["A1_citation_present"]["n"] == 1
    assert out["skipped_label_cells"] == 2  # C7 not applicable; q30 is held-out

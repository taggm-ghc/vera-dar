"""K2.I0: answer contract foundation (plan-K2-contract.md r2): config texts, layers, L3 seam, isolation."""
import ast
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from vera import answer_contract as ac

ROOT = Path(__file__).resolve().parent.parent
FIXED = "VERA-DAR has not acquired or read the documents in this list. This is a discovery result, not an evidence-grounded answer."


def _cfg(tmp_path, **over):
    d = json.loads(ac.CONFIG_PATH.read_text())
    d.update(over)
    p = tmp_path / "c.json"
    p.write_text(json.dumps(d))
    return p


def test_not_read_line_verbatim_from_config():
    assert ac.text("not_read_line") == FIXED
    assert ac.not_read_line([]) == FIXED


def test_partial_variant_only_when_some_but_not_all_acquired():
    a, n = {"acquisition": ac.ACQ_ACQUIRED}, {"acquisition": ac.ACQ_NOT_ACQUIRED}
    assert ac.not_read_line([n, n]) == FIXED
    assert ac.not_read_line([a, a]) == FIXED  # all acquired is state A, which K2 does not produce
    partial = ac.not_read_line([a, n])
    assert partial == ac.text("not_read_line_partial") != FIXED
    assert "except those marked 'full document stored (not inspected for this answer)'" in partial


def test_layers_default_excludes_l1_and_always_has_l0(tmp_path):
    assert ac.layers() == ["L0"]  # R-028 Demo Day config (was ["L0", "L2"])
    assert ac.layers(_cfg(tmp_path, layers=["L2", "bogus", "L2"])) == ["L0", "L2"]
    assert ac.layers(tmp_path / "missing.json") == ["L0"]


def test_evidence_recall_switch_default_false_and_fail_closed(tmp_path):
    assert ac.evidence_recall_enabled() is False
    assert ac.evidence_recall_enabled(tmp_path / "missing.json") is False
    assert ac.evidence_recall_enabled(_cfg(tmp_path, memory={"evidence_recall_enabled": "true"})) is False  # strict bool


def test_write_on_l0_default_true_and_fail_closed(tmp_path):
    assert ac.write_on_l0() is True
    assert ac.write_on_l0(tmp_path / "missing.json") is False


def test_owner_wording_rulings_present():  # OD-K15b
    assert ac.text("discovery_note_submitted") == "Discovery notes are submitted, and those that pass wait for owner review."
    assert ac.text("evidence_findings_submitted") == "Findings are submitted, and those that pass wait for owner review."


def test_default_kind_is_non_answer():
    assert ac.DEFAULT_KIND == "unspecified" and "answer" not in ac.NON_ANSWER_KINDS


def test_acquisition_never_from_fetch_status_or_abstract_kinds():
    assert ac.acquisition_state({"fetch_status": "fetched"}) == ac.ACQ_NOT_ACQUIRED
    assert ac.acquisition_state({"stored_text_kind": "abstract_metadata", "admission_passed": True}) == ac.ACQ_NOT_ACQUIRED
    assert ac.acquisition_state({"stored_text_kind": "full_text"}) == ac.ACQ_NOT_ACQUIRED  # not admitted
    assert ac.acquisition_state({"stored_text_kind": "full_text", "admission_passed": True}) == ac.ACQ_ACQUIRED


def test_representation_text_per_state():
    assert ac.representation(ac.ACQ_NOT_ACQUIRED) == ("not in VERA-DAR's collection (index record or abstract; "
                                                      "full document not acquired)")
    assert ac.representation(ac.ACQ_IN_CORPUS) == "in VERA-DAR's collection: full document stored (not inspected for this answer)"
    assert ac.text("representation_not_acquired") == "index record or abstract; full document not acquired"  # fallback
    assert ac.representation(ac.ACQ_ACQUIRED) == "full document stored (not inspected for this answer)"


def test_no_sources_text_from_chain_config_never_empty(tmp_path):
    assert "has not answered from memory" in ac.no_sources_text()
    assert ac.no_sources_text(chain_path=tmp_path / "missing.json") == ac.text("no_sources_fallback")
    assert ac.text("no_sources_fallback")


# ---- L3 seam (V2/V9): "answer" unreachable while the stub returns [] ----
CASES = [[], [{}], [{"stored_text_kind": "full_text", "admission_passed": True, "acquisition": ac.ACQ_ACQUIRED}],
         [{"n": 1, "title": "t", "snippet": "x" * 1500}], None]


@pytest.mark.parametrize("cands", CASES)
def test_answer_kind_unreachable_with_stub(cands):
    assert ac.examined_full_documents(cands) == []
    assert ac.result_kind_for(cands) == "reading_list"


def test_stub_not_flippable_by_config_or_env(monkeypatch, tmp_path):
    for k in ("VERA_EXAMINED_FULL_DOCUMENTS", "EXAMINED_FULL_DOCUMENTS", "VERA_L3"):
        monkeypatch.setenv(k, "1")
    cfg = _cfg(tmp_path, examined_full_documents=["x"], layers=["L0", "L3"])
    assert ac.examined_full_documents(CASES[2]) == []
    assert "L3" in ac.layers(cfg) and ac.result_kind_for(CASES[2]) == "reading_list"  # a layer name alone enables nothing
    tree = ast.parse((ROOT / "vera" / "answer_contract.py").read_text())
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "examined_full_documents")
    body = ast.unparse(fn.body[1:])  # code only, not the docstring
    assert "environ" not in body and "getenv" not in body and "load_config" not in body and "open(" not in body


def test_l3_never_receives_notes_source_guard():  # D8c: the contract module has no memory/recall path at all
    src = (ROOT / "vera" / "answer_contract.py").read_text()
    assert "recalled_discovery" not in src and "memory_store" not in src


# ---- D5 isolation: L0 works when the L1 modules are absent or raise on import ----
def test_contract_never_imports_l1_modules():
    tree = ast.parse((ROOT / "vera" / "answer_contract.py").read_text())
    names = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            names += [a.name for a in n.names]
        elif isinstance(n, ast.ImportFrom):
            names += [(n.module or "") + "." + a.name for a in n.names]
    assert not [x for x in names if "discovery" in x.replace("answer_contract", "")]


def test_contract_imports_with_l1_modules_poisoned():
    code = (f"import sys\nsys.path.insert(0, {str(ROOT)!r})\nsys.modules['vera.discovery']=None\nsys.modules['vera.discovery_leak_check']=None\n"
            "import vera.answer_contract as a\nassert a.layers()[0]=='L0'\nprint('ok')")
    r = subprocess.run([sys.executable, "-I", "-B", "-c", code], cwd=ROOT, capture_output=True, text=True,
                       env={"PATH": ""})
    assert r.stdout.strip() == "ok", r.stderr


def test_config_has_no_forbidden_public_terms():  # texts shown to visitors name no back-end component
    for k, v in json.loads(ac.CONFIG_PATH.read_text())["texts"].items():
        assert not re.search(r"gate|1,?500|PLACEHOLDER|owner key|D6", v), k


@pytest.mark.parametrize("kind,fb,n,want", [
    ("answer", None, 0, "answer"),
    ("answer", "insufficient", 3, "answer"),
    ("reading_list", "insufficient", 2, "insufficient_evidence"),
    ("reading_list", "full_document_corpus_unusable", 2, "corpus_unusable"),
    ("reading_list", "some_future_token", 2, "insufficient_evidence"),
    ("reading_list", "checker_unavailable", 2, "insufficient_evidence"),
    ("reading_list", "claim_check_insufficient", 0, "insufficient_evidence"),
    ("reading_list", None, 2, "reading_list"),
    ("reading_list", None, 0, "no_sources"),
    ("decline", None, 0, "decline"),
    ("unspecified", None, 0, "insufficient_evidence"),
])
def test_outcome_for(kind, fb, n, want):
    assert ac.outcome_for(kind, fb, n) == want
    assert want != "partial_answer"

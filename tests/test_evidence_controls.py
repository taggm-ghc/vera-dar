"""PROC-PIN-1107 T1/T2/T3: process corrections as tests. SYNTHETIC rows only: no DB, no network, no env files.

T1  every evidence loader that builds a FullDocCorpus from stored rows must refuse a coordinated tamper (rows and
    their recorded hashes rewritten together). E1 (base loader) does. E2 (acquired loader) does not, and config/controls.json
    records that as an exception. The E2 test branches on that exception, so it is strict in both directions: while E2 is
    listed, the tampered document must be served; once the exception is removed, it must be refused. Setup failures are
    plain asserts and fail loudly (they are not an expected failure).
T2  config/controls.json (in-repo subset of the controls register) is complete and consistent with the loaders.
T3  a constraint marked temporary (config `_constraint`, or a control exception) carries a review date. Past that
    date the test STILL PASSES (owner rule: cutoffs never block fixes or corrections)
    but it emits a loud REVIEW-DUE warning and printed line, and scripts/release_gate.py prints the same as a NOTE.
    Re-decide the constraint (fix it, or renew review_by with an owner decision); do not delete the marker."""
import copy
import datetime
import ast
import hashlib
import inspect
import json
import re
import warnings
from pathlib import Path

import pytest

from tests import acq_support as A
from tests import fulldoc_bundle_synth as synth
from tests.test_fulldoc_db import _db, _script
from tests.test_fulldoc_doc_pins import _coordinated_tamper
from vera import fulldoc_corpus as fc, fulldoc_db

REPO = Path(__file__).resolve().parent.parent
CONTROLS = REPO / "config" / "controls.json"


def _controls() -> dict:
    return json.loads(CONTROLS.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------------------------------------------- T1
def test_base_loader_rejects_coordinated_tamper(tmp_path):
    fc.reset_for_tests()
    cfg = synth.synth_cfg()
    rows, report = _script().build_plan(synth.write_bundle(tmp_path / "b"), cfg, target=700)
    sources, spans = _db(rows)
    pinned = copy.deepcopy(cfg)
    pinned["pins"] = {**report["pins"], "documents": {
        s["provenance"]["document_id"]: {"text_sha256": s["provenance"]["text_sha256"],
                                        "licence_id": s["provenance"]["item"]["licence"], "licence_decision": "allow"}
        for s in sources}}
    _coordinated_tamper(sources, spans)
    c = fulldoc_db.load_from_rows(sources, spans, pinned)
    assert c.status == "corpus_unusable" and not c.acquired and c._texts == {}


def _e2_is_open_exception() -> bool:
    pin = next(c for c in _controls()["controls"] if c["id"] == "CTL-PIN")
    return any(e.get("path") == "E2" for e in pin.get("exceptions", []))


def test_acquired_loader_tamper_matches_register():
    from tests import test_fulldoc_e2e as e2e
    from vera import auto_acquire
    web, _ = A.epi_setup()
    store = A.Store()
    auto_acquire.run_job(A.job(e2e.Q_EPI), A.cfg(), A.deps([A.cand("10.9999/acq.epi4")], web, store=store),
                         clock=A.clock_at(), env=A.ENV_ON)
    # setup (plain asserts, fail loudly): a plan was stored and the untampered row is served
    assert store.plans, "setup: no acquired plan was stored"
    src, spans = A.rows_from_plans(store.plans)
    clean = fulldoc_db.load_acquired_rows(copy.deepcopy(src), copy.deepcopy(spans), fc.load_config(), A.cfg())
    assert clean.acquired, "setup: the untampered acquired row is not served"
    # coordinated tamper: one span character plus both recorded hashes
    first = next(x for x in sorted(spans, key=lambda x: x["start_index"]) if re.search("[a-z]", x["text"]))
    i = re.search("[a-z]", first["text"]).start()
    first["text"] = first["text"][:i] + ("z" if first["text"][i] != "z" else "y") + first["text"][i + 1:]
    text = "".join(x["text"] for x in sorted((x for x in spans if x["source_id"] == src[0]["source_id"]),
                                             key=lambda x: x["start_index"]))
    h = hashlib.sha256(text.encode("utf-8")).hexdigest()
    prov = json.loads(src[0]["provenance"])
    prov["text_sha256"] = h
    if isinstance(prov.get("meta"), dict):
        prov["meta"]["text_sha256"] = h
    src[0]["provenance"] = json.dumps(prov)
    c = fulldoc_db.load_acquired_rows(src, spans, fc.load_config(), A.cfg())
    if _e2_is_open_exception():
        assert c.acquired, ("config/controls.json lists CTL-PIN E2 as an open exception but the tampered acquired document "
                            "is now refused: E2 is fixed, remove the exception (and its _constraint marker)")
    else:
        assert not c.acquired, "CTL-PIN E2 is no longer an exception, so a coordinated tamper must be refused"


# ---------------------------------------------------------------------------------------------------------------- T2
def test_controls_register_coverage():
    reg = _controls()
    paths = reg["paths"]
    assert reg["controls"]
    for ctl in reg["controls"]:
        for key in ("id", "control", "must_cover", "covered", "owner", "test"):
            assert ctl.get(key), f"{ctl.get('id')}: missing {key}"
        exc = {e["path"]: e for e in ctl.get("exceptions", [])}
        for p in ctl["must_cover"]:
            assert p in paths, f"{ctl['id']}: unknown path {p}"
            if p not in ctl["covered"]:
                e = exc.get(p)
                assert e and e.get("review_by") and "accepted_by" in e, f"{ctl['id']}: {p} is neither covered nor a dated exception"
        assert not (set(ctl["covered"]) & set(exc)), f"{ctl['id']}: a path is both covered and an exception"
        tfile, _, tname = ctl["test"].partition("::")
        src = (REPO / tfile).read_text(encoding="utf-8")
        assert re.search(rf"def {re.escape(tname)}\b", src), f"{ctl['id']}: test {ctl['test']} not found"


def test_every_row_loader_is_registered_and_pins_stripping_has_an_exception():
    """Enumerate the loaders directly: any public fulldoc_db function that builds a corpus from stored rows must be a
    registered evidence path, and each place that drops the pins must be a CTL-PIN exception."""
    reg = _controls()
    registered = {v.split(":", 1)[1] for v in reg["paths"].values() if v.startswith("vera/fulldoc_db.py:")}
    loaders = {n for n, f in inspect.getmembers(fulldoc_db, inspect.isfunction)
               if not n.startswith("_") and f.__module__ == fulldoc_db.__name__ and re.fullmatch(r"load_\w*rows", n)}
    assert loaders == registered, f"unregistered or stale row loaders: {sorted(loaders ^ registered)}"
    pin = next(c for c in reg["controls"] if c["id"] == "CTL-PIN")
    allowed = {e["site"] for e in pin.get("exceptions", []) if e.get("site")}
    assert _pin_stripping_sites() == allowed, \
        f"pins-stripping sites differ from CTL-PIN exceptions: found {sorted(_pin_stripping_sites())}, listed {sorted(allowed)}"


def _is_pins(n) -> bool:
    return isinstance(n, ast.Constant) and n.value == "pins"


def _pin_stripping_sites(root: Path | None = None) -> set:
    """AST scan of vera/ for code that drops or blanks the pins: a dict comprehension filtering on the key "pins",
    a dict display {**cfg, "pins": ...}, .pop("pins"), `del x["pins"]`, and `x["pins"] = ...`. Returns "file:function"."""
    root = root or REPO
    hits = set()
    for f in sorted((root / "vera").rglob("*.py")):
        tree = ast.parse(f.read_text(encoding="utf-8"))
        rel = f.relative_to(root).as_posix()

        def walk(node, fn):
            for ch in ast.iter_child_nodes(node):
                name = ch.name if isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef)) else fn
                bad = (
                    (isinstance(ch, ast.DictComp) and any(_is_pins(c) for g in ch.generators for t in g.ifs
                                                          for c in ast.walk(t)))
                    or (isinstance(ch, ast.Dict) and None in ch.keys and any(_is_pins(k) for k in ch.keys))
                    or (isinstance(ch, ast.Call) and isinstance(ch.func, ast.Attribute) and ch.func.attr in ("pop", "setdefault")
                        and ch.args and _is_pins(ch.args[0]))
                    or (isinstance(ch, (ast.Delete, ast.Assign))
                        and any(isinstance(t, ast.Subscript) and _is_pins(t.slice)
                                for t in (ch.targets if isinstance(ch, (ast.Delete, ast.Assign)) else [])))
                )
                if bad:
                    hits.add(f"{rel}:{name or '<module>'}")
                walk(ch, name)
        walk(tree, None)
    return hits


def test_pin_stripping_scan_detects_each_spelling(tmp_path):
    (tmp_path / "vera").mkdir()
    cases = {
        "a": "def f(cfg):\n    return {k: v for k, v in cfg.items() if k != 'pins'}\n",
        "b": "def f(cfg):\n    return {**cfg, 'pins': None}\n",
        "c": "def f(cfg):\n    cfg.pop('pins')\n",
        "d": "def f(cfg):\n    del cfg['pins']\n",
        "e": "def f(cfg):\n    cfg['pins'] = {}\n",
    }
    for k, src in cases.items():
        (tmp_path / "vera" / f"{k}.py").write_text(src, encoding="utf-8")
    assert _pin_stripping_sites(tmp_path) == {f"vera/{k}.py:f" for k in cases}


# ---------------------------------------------------------------------------------------------------------------- T3
def _temporary_markers():
    for f in sorted((REPO / "config").glob("*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        stack = [(f.name, d)]
        while stack:
            where, node = stack.pop()
            if isinstance(node, dict):
                c = node.get("_constraint")
                if isinstance(c, dict) and c.get("kind") == "temporary":
                    yield f"{where}", c
                stack += [(f"{where}.{k}", v) for k, v in node.items() if isinstance(v, dict) and k != "_constraint"]
    for ctl in _controls()["controls"]:
        for e in ctl.get("exceptions", []):
            yield f"controls.json:{ctl['id']}:{e['path']}", e


def test_temporary_markers_are_complete():
    found = list(_temporary_markers())
    assert any(w.startswith("fulldoc_answer.json") for w, _ in found) and any(w.startswith("auto_acquire.json") for w, _ in found)
    for where, c in found:
        datetime.date.fromisoformat(c["review_by"])
        assert c.get("owner") and c.get("exit"), f"{where}: temporary marker needs owner and exit"


def _overdue(today):
    return [f"{w}: review_by {c['review_by']} passed; owner {c['owner']}; exit: {c['exit']}"
            for w, c in _temporary_markers() if datetime.date.fromisoformat(c["review_by"]) < today]


def test_temporary_constraints_review_due_is_loud_but_not_blocking():
    late = _overdue(datetime.date.today())
    for m in late:
        print(f"REVIEW-DUE {m}")
        warnings.warn(f"REVIEW-DUE {m}", UserWarning, stacklevel=1)
    # the detector itself works: far in the future every marker is overdue (checked without failing the suite)
    assert len(_overdue(datetime.date(2100, 1, 1))) == len(list(_temporary_markers()))


def test_release_gate_review_due_note_detects_overdue(tmp_path, monkeypatch):
    import importlib.util
    spec = importlib.util.spec_from_file_location("rg_probe", REPO / "scripts" / "release_gate.py")
    rg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rg)
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "x.json").write_text(json.dumps({"_constraint": {"kind": "temporary", "review_by": "2000-01-01",
                                                                             "owner": "o", "exit": "e"}}), encoding="utf-8")
    assert len(rg._review_due_notes(tmp_path)) == 1
    assert rg._review_due_notes(REPO) == [] or all("review_by" in n for n in rg._review_due_notes(REPO))


def test_release_gate_one_malformed_config_does_not_hide_other_notes(tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("rg_probe2", REPO / "scripts" / "release_gate.py")
    rg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rg)
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "controls.json").write_text(json.dumps({"controls": ["x"]}), encoding="utf-8")
    (tmp_path / "config" / "broken.json").write_text("{not json", encoding="utf-8")
    (tmp_path / "config" / "z.json").write_text(json.dumps({"_constraint": {"kind": "temporary", "review_by": "2000-01-01",
                                                                            "owner": "o", "exit": "e"}}), encoding="utf-8")
    notes = rg._review_due_notes(tmp_path)
    assert sum(n.startswith("REVIEW-DUE z.json") for n in notes) == 1
    assert sum(n.startswith("MALFORMED ") for n in notes) == 2

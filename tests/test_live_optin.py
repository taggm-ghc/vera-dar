"""Owner decision #7: default pytest makes no live calls; live tests need --run-live-stages AND API_BASE_URL."""
from tests.conftest import live_skip_reason


def test_flag_defaults_to_false(pytestconfig):
    assert pytestconfig.getoption("--run-live-stages") is False


def test_live_marker_registered(pytestconfig):
    assert all(any(m.startswith(n + ":") for m in pytestconfig.getini("markers")) for n in ("live", "live_api"))


def test_skip_unless_both_flag_and_env():
    assert "--run-live-stages" in live_skip_reason(False, {"API_BASE_URL": "x"})
    assert live_skip_reason(True, {}) is None  # provider-only live test needs only the flag
    assert "API_BASE_URL" in live_skip_reason(True, {}, needs_api=True)
    assert "API_BASE_URL" in live_skip_reason(True, {"API_BASE_URL": "  "}, needs_api=True)
    assert live_skip_reason(True, {"API_BASE_URL": "x"}, needs_api=True) is None


def test_skip_reason_never_echoes_url():
    assert "secret.example" not in live_skip_reason(False, {"API_BASE_URL": "https://secret.example"}, needs_api=True)


def test_live_tests_skipped_by_default(pytestconfig):
    if pytestconfig.getoption("--run-live-stages"):
        return  # explicit opt-in run; the guard applies to the default run only
    import subprocess, sys
    p = subprocess.run([sys.executable, "-B", "-m", "pytest", "-p", "no:cacheprovider", "tests/test_m4_m5_integration.py", "-q", "-rs"],
                       capture_output=True, text=True, cwd=str(pytestconfig.rootpath), timeout=120,
                       env={k: v for k, v in __import__("os").environ.items() if k != "API_BASE_URL"})
    assert "skipped" in p.stdout and p.returncode == 0 and " passed" not in p.stdout.splitlines()[-1] and " failed" not in p.stdout.splitlines()[-1], p.stdout[-500:]


def test_deprecated_alias_enables_live_with_one_warning(monkeypatch):
    import pytest
    from tests import conftest as c

    class Cfg:
        def getoption(self, name):
            return False

    class Item:
        keywords = {"live": True}
        marks = []

        def add_marker(self, m):
            self.marks.append(m)

    monkeypatch.setenv("VERA_INTEGRATION", "1")
    monkeypatch.setattr(c, "alias_active", lambda env, today=None: True)
    item = Item()
    with pytest.warns(DeprecationWarning, match="--run-live-stages") as rec:
        c.pytest_collection_modifyitems(Cfg(), [item, Item()])
    assert len(rec) == 1 and item.marks == []


def _root():
    import pathlib
    return pathlib.Path(__file__).resolve().parent.parent


def test_live_catalog_check_is_not_a_pytest_test():
    import ast
    from tests import test_sql_contract
    assert not hasattr(test_sql_contract, "test_sql_columns_exist_in_live_catalog")
    assert (_root() / "scripts" / "live_catalog_check.py").is_file()
    tree = ast.parse((_root() / "conftest.py").read_text())
    ignored = [ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
               and any(getattr(t, "id", "") == "collect_ignore" for t in n.targets)]
    assert ignored and "scripts" in ignored[0]


def _gate():
    import importlib.util
    spec = importlib.util.spec_from_file_location("release_gate", _root() / "scripts" / "release_gate.py")
    g = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(g)
    return g


def _evidence_root(tmp_path, script_text="print('x')\n"):
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "live_catalog_check.py").write_text(script_text)
    import hashlib
    return hashlib.sha256((tmp_path / "scripts" / "live_catalog_check.py").read_bytes()).hexdigest()


def test_gate_catalog_evidence_missing_fails(tmp_path):
    g = _gate()
    _evidence_root(tmp_path)
    for p in (None, tmp_path / "absent.txt"):
        o, why = g.catalog_evidence_outcome(p, tmp_path)
        assert o == "MISSING" and "--catalog-evidence" in why and "never connects" in why
    assert g.check_live_catalog_evidence(None, tmp_path) != []


def test_gate_catalog_evidence_stale_hash_fails(tmp_path):
    g = _gate()
    _evidence_root(tmp_path)
    ev = tmp_path / "ev.txt"
    ev.write_text("live_catalog_check script_sha256=" + "0" * 64 + "\nRESULT: PASS\n")
    assert g.catalog_evidence_outcome(ev, tmp_path)[0] == "STALE"


import pytest  # noqa: E402


@pytest.mark.parametrize("result", ["RESULT: FAIL", "RESULT: CANNOT_VERIFY", "RESULT: MATCH", "RESULT: ABORTED", "RESULT: MISMATCH 2",
                                    "RESULT: PASS\nRESULT: FAIL", ""])
def test_gate_catalog_evidence_mismatch_or_abort_fails(tmp_path, result):
    g = _gate()
    digest = _evidence_root(tmp_path)
    ev = tmp_path / "ev.txt"
    ev.write_text(f"live_catalog_check script_sha256={digest}\n{result}\n")
    o, why = g.catalog_evidence_outcome(ev, tmp_path)
    assert o == "FAILED" and "not a match" in why
    assert g.check_live_catalog_evidence(ev, tmp_path) == [why]


def test_gate_catalog_evidence_match_passes_only_when_bound(tmp_path):
    # D-B5-1: a matching body is no longer enough; the evidence must be bound to the candidate (name + sidecar).
    # Real script and checker bytes are copied (not imported); evidence and SHA are synthetic.
    import hashlib
    import shutil
    g = _gate()
    (tmp_path / "scripts").mkdir()
    for rel in (g.CATALOG_SCRIPT, g.CATALOG_CHECKER):
        shutil.copyfile(_root() / rel, tmp_path / rel)
    digest = hashlib.sha256((tmp_path / g.CATALOG_SCRIPT).read_bytes()).hexdigest()
    cand = "e" * 40
    body = f"live_catalog_check script_sha256={digest}\nchecked 5 (table, column) pairs; 0 missing\nRESULT: PASS\n"
    bare = tmp_path / "ev.txt"
    bare.write_text(body)
    assert g.catalog_evidence_outcome(bare, tmp_path, cand)[0] == "STALE"   # bare name: not bound
    assert g.catalog_evidence_outcome(bare, tmp_path)[0] == "STALE"         # no candidate: fail closed
    ev = tmp_path / f"catalog-evidence-{cand[:12]}.txt"
    ev.write_text(body)
    sc = [f"candidate_sha={cand}", f"head_after={cand}", f"script_sha256={digest}", f"evidence_file={ev.name}",
          f"evidence_sha256={hashlib.sha256(ev.read_bytes()).hexdigest()}", "exit_code=0", "worktree_clean=1",
          "run_start_utc=2026-10-17T16:51:13Z", "run_end_utc=2026-10-17T16:51:14Z"]
    (tmp_path / (ev.name + ".sidecar")).write_text("vera-catalog-evidence-sidecar v1\n" + "\n".join(sc) + "\n")
    assert g.catalog_evidence_outcome(ev, tmp_path, cand)[0] == "PASSED"
    assert g.check_live_catalog_evidence(ev, tmp_path, cand) == []
    ev.write_text(f"script_sha256={digest}\nchecked 5 (table, column) pairs; 0 missing\nRESULT: PASS\n")
    assert g.catalog_evidence_outcome(ev, tmp_path, cand)[0] == "STALE"     # bare header form no longer accepted


def test_gate_never_runs_live_pytest_or_connects():
    src = (_root() / "scripts" / "release_gate.py").read_text()
    assert "--run-live-stages" not in src and "CATALOG_NODE" not in src
    import ast
    for node in ast.walk(ast.parse(src)):
        names = [a.name for a in node.names] if isinstance(node, ast.Import) else \
            ([node.module or ""] + [a.name for a in node.names]) if isinstance(node, ast.ImportFrom) else []
        for n in names:
            assert not any(n.startswith(b) for b in ("psycopg2", "sqlalchemy", "live_catalog_check")), n

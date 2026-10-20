"""DAR test 14 (OD-K17bw s9D): existing API compatibility, checked as TEXT via ast.
main.py is never imported (test-clearance ruling). Stdlib + pytest only."""
import ast
import sys
from pathlib import Path

import pytest


def _root() -> Path:
    for p in Path(__file__).resolve().parents:
        if (p / "main.py").is_file() and (p / "vera").is_dir():
            return p
    pytest.fail("repo root not found")


ROOT = _root()
MAIN = ROOT / "main.py"
GROUNDED = ROOT / "vera" / "grounded_ask.py"

# Frozen at 9c3dd56: main.py:126 (class AskResponse), fields at main.py:127-140.
FROZEN_FIELDS = {
    "answer", "model", "tokens_used", "cost_usd", "sources", "traced_sources", "recalled",
    "memory_written", "memory_submitted", "claim_check", "result_kind", "candidates",
    "outcome",  # additive, FDC-1020 2026-10-20
    "contract", "recalled_discovery", "discovery_note_submitted",
    # D-055 Option C (owner-directed, additive): serialized only when full-document evidence exists, so the web-only
    # key set stays HEAD | NEW (tests/test_ask_response_contract.py:80-81 unchanged)
    "evidence_status", "examined_full_documents", "admitted_evidence", "citations",
}


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _ask_response_fields() -> list:
    for node in ast.walk(_tree(MAIN)):
        if isinstance(node, ast.ClassDef) and node.name == "AskResponse":
            return [s.target.id for s in node.body
                    if isinstance(s, ast.AnnAssign) and isinstance(s.target, ast.Name)]
    pytest.fail("class AskResponse not found in main.py")


def _imported_modules(path: Path) -> set:
    mods = set()
    for n in ast.walk(_tree(path)):
        if isinstance(n, ast.Import):
            for a in n.names:
                mods.add(a.name)
                mods.add(a.name.split(".")[0])
        elif isinstance(n, ast.ImportFrom):
            base = ("." * n.level) + (n.module or "")
            mods.add(base)
            for a in n.names:
                mods.add(f"{base}.{a.name}")
    return mods


def test_ask_response_fields_exactly_frozen():
    fields = _ask_response_fields()
    assert len(fields) == len(set(fields))
    assert set(fields) == FROZEN_FIELDS, (
        f"added={set(fields) - FROZEN_FIELDS} removed={FROZEN_FIELDS - set(fields)}")


def test_no_domain_field_in_ask_response():
    assert [f for f in _ask_response_fields() if "domain" in f.lower()] == []


def test_ask_route_still_declared_with_ask_response():
    found = False
    for n in ast.walk(_tree(MAIN)):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for d in n.decorator_list:
                if (isinstance(d, ast.Call) and isinstance(d.func, ast.Attribute) and d.func.attr == "post"
                        and d.args and isinstance(d.args[0], ast.Constant) and d.args[0].value == "/ask"):
                    kw = {k.arg: k.value for k in d.keywords}
                    assert isinstance(kw.get("response_model"), ast.Name)
                    assert kw["response_model"].id == "AskResponse"
                    found = True
    assert found, "POST /ask route with response_model=AskResponse not found"


def _top_level_dar_imports(tree) -> list:
    return [n.lineno for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))
            and any(x in ((n.module or "") if isinstance(n, ast.ImportFrom) else " ".join(a.name for a in n.names))
                    for x in ("domain_routing", "domain_registry", "dar_", "sqlalchemy"))]


@pytest.mark.parametrize("banned", ["vera.domain_routing", "vera.domain_registry", "vera.dar_corpus", "vera.dar_store"])
def test_main_does_not_import_dar_directly(banned):
    # DARINT owner decision: DAR is integrated, but only through vera.dar_runtime, lazily, inside a function.
    hits = [m for m in _imported_modules(MAIN) if m == banned or m.startswith(banned + ".")]
    assert hits == [], f"main.py imports {hits}; DAR only via vera.dar_runtime"


def test_main_current_has_no_dar_import_at_module_top():
    assert _top_level_dar_imports(_tree(MAIN)) == []
    assert not [n for n in _tree(MAIN).body if isinstance(n, ast.ImportFrom) and (n.module or "") == "vera.dar_runtime"]


def _function_scoped_imports(tree) -> list:
    """(node, enclosing function) for every import inside a function body of main.py."""
    out = []
    for fn in ast.walk(tree):
        if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for n in ast.walk(fn):
                if isinstance(n, (ast.Import, ast.ImportFrom)):
                    out.append((n, fn))
    return out


def test_applied_main_hunk_imports_dar_only_via_runtime_inside_function():
    # (c) is applied to main.py: the same checks the staged patch used to get, now on the applied hunk (ast, text only).
    tree = _tree(MAIN)
    found = []
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom):
            mod = n.module or ""
            if any(x in mod for x in ("dar", "domain_", "sqlalchemy")):
                found.append(n)
        elif isinstance(n, ast.Import):
            if any(any(x in a.name for x in ("dar", "domain_", "sqlalchemy")) for a in n.names):
                found.append(n)
    assert found, "main.py should import vera.dar_runtime (applied hunk not found)"
    for n in found:  # exactly `from vera.dar_runtime import get_corpus`
        assert isinstance(n, ast.ImportFrom) and n.module == "vera.dar_runtime" and n.level == 0, ast.dump(n)
        assert [a.name for a in n.names] == ["get_corpus"] and all(a.asname is None for a in n.names), ast.dump(n)
    inside = {id(n) for n, _fn in _function_scoped_imports(tree)}
    assert all(id(n) in inside for n in found), "import must be inside a function (indented), not module level"
    assert all(n.col_offset > 0 for n in found), "import must be indented"
    assert not [n for n in tree.body if id(n) in {id(f) for f in found}]


def test_flag_off_get_corpus_none_without_imports_or_env(monkeypatch):
    from vera import dar_runtime
    for m in ("vera.dar_store", "vera.dar_corpus", "sqlalchemy"):
        monkeypatch.delitem(sys.modules, m, raising=False)

    class NoEnv:
        def get(self, *a, **k):
            raise AssertionError("env read")
    monkeypatch.setattr(dar_runtime.domain_registry, "load_config",
                        lambda *a, **k: {"enabled": False, "corpus": {"stored_branch_enabled": False}})
    dar_runtime.reset_for_tests()
    assert dar_runtime.get_corpus(env=NoEnv()) is None
    assert not any(m in sys.modules for m in ("vera.dar_store", "vera.dar_corpus", "sqlalchemy"))


def _dr_hook_try(tree):
    """Innermost Try whose body contains `_dr.flag_enabled()` (same notion as test_dar_hook_static)."""
    best = None
    for n in ast.walk(tree):
        if isinstance(n, ast.Try):
            for b in n.body:
                for c in ast.walk(b):
                    if (isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute) and c.func.attr == "flag_enabled"
                            and isinstance(c.func.value, ast.Name) and c.func.value.id == "_dr"):
                        if best is None or n.lineno >= best.lineno:
                            best = n
    return best


def test_grounded_ask_references_domain_routing_only_inside_flag_guarded_hook():
    # Updated for the T2b commit. Valid with or without the hook: no reference passes; a reference
    # is allowed only inside the try body that calls `_dr.flag_enabled()`.
    tree = _tree(GROUNDED)
    refs, aliases = [], set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names:
                if "domain_routing" in a.name:
                    refs.append(n)
                    aliases.add(a.asname or a.name.split(".")[0])
        elif isinstance(n, ast.ImportFrom):
            if "domain_routing" in (n.module or "") or any(a.name == "domain_routing" for a in n.names):
                refs.append(n)
                for a in n.names:
                    aliases.add(a.asname or a.name)
    aliases.add("_dr") if refs else None
    for n in ast.walk(tree):
        if isinstance(n, ast.Name) and n.id in aliases:
            refs.append(n)
        elif isinstance(n, ast.Constant) and isinstance(n.value, str) and "domain_routing" in n.value:
            refs.append(n)
    if not refs:
        return
    hook = _dr_hook_try(tree)
    assert hook is not None, "domain_routing referenced but no `_dr.flag_enabled()` guarded hook try found"
    inside = {id(x) for b in hook.body for x in ast.walk(b)}
    outside = [r.lineno for r in refs if id(r) not in inside]
    assert outside == [], f"domain_routing referenced outside the flag-guarded hook at lines {outside}"

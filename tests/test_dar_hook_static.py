"""DAR T2b hook: static AST guard on vera/grounded_ask.py (Phase E). Text only; nothing imported.
Stdlib + pytest only. SKIPS while the hook is absent (T2b not yet applied)."""
import ast
from pathlib import Path

import pytest


def _root() -> Path:
    for p in Path(__file__).resolve().parents:
        if (p / "main.py").is_file() and (p / "vera").is_dir():
            return p
    pytest.fail("repo root not found")


GROUNDED = _root() / "vera" / "grounded_ask.py"
CFG_FNS = {"get_cfg", "get_registry"}
FLAG = "flag_enabled"
BRANCHES = "_domain_branches"


def _call_name(c: ast.Call):
    f = c.func
    if isinstance(f, ast.Attribute):
        return f.attr, (f.value.id if isinstance(f.value, ast.Name) else None)
    if isinstance(f, ast.Name):
        return f.id, None
    return None, None


def _dr_imports(tree):
    """Every import of domain_routing, any form. Returns (import nodes, module aliases, bare flag-function names).
    Forms: `from vera import domain_routing [as X]`, `import vera.domain_routing [as X]`,
    `from vera.domain_routing import flag_enabled [as Y]` (aliases resolved dynamically, nothing hard-coded)."""
    nodes, mods, flags = [], set(), set()
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom):
            if n.module == "vera":
                for a in n.names:
                    if a.name == "domain_routing":
                        nodes.append(n)
                        mods.add(a.asname or a.name)
            elif n.module in ("vera.domain_routing", "domain_routing") or (n.level and n.module == "domain_routing"):
                nodes.append(n)
                for a in n.names:
                    if a.name == FLAG:
                        flags.add(a.asname or a.name)
            elif n.module is None and n.level:
                for a in n.names:
                    if a.name == "domain_routing":
                        nodes.append(n)
                        mods.add(a.asname or a.name)
        elif isinstance(n, ast.Import):
            for a in n.names:
                if a.name == "vera.domain_routing":
                    nodes.append(n)
                    # `import vera.domain_routing` binds `vera`; the call is vera.domain_routing.flag_enabled()
                    mods.add(a.asname or "vera.domain_routing")
    return nodes, mods, flags


def _is_flag(n, ctx) -> bool:
    if not isinstance(n, ast.Call):
        return False
    _, mods, flags = ctx
    f = n.func
    if isinstance(f, ast.Name):
        return f.id in flags
    if isinstance(f, ast.Attribute) and f.attr == FLAG:
        return _dotted(f.value) in mods
    return False


def _dotted(node):
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def _names(node, ctx=None):
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name) and (ctx is None or isinstance(n.ctx, ctx))}


@pytest.fixture(scope="module")
def tree():
    src = GROUNDED.read_text(encoding="utf-8")
    t = ast.parse(src, filename=str(GROUNDED))
    ctx = _dr_imports(t)
    if any(_is_flag(n, ctx) for n in ast.walk(t)):
        return t
    if "domain_routing" in src or ctx[0]:
        pytest.fail("grounded_ask.py references domain_routing but the flag_enabled() hook shape is not recognised")
    pytest.skip("T2b hook absent: domain_routing is not referenced in vera/grounded_ask.py (T2b not yet applied)")


@pytest.fixture(scope="module")
def ctx(tree):
    return _dr_imports(tree)


@pytest.fixture(scope="module")
def hook(tree, ctx):
    """The innermost Try whose body (not handlers) contains the flag call."""
    best = None
    for n in ast.walk(tree):
        if isinstance(n, ast.Try) and any(_is_flag(c, ctx) for b in n.body for c in ast.walk(b)):
            if best is None or n.lineno >= best.lineno:
                best = n
    assert best is not None, "domain_routing flag_enabled() call is not inside any try block"
    return best


def _inside(outer, node) -> bool:
    return any(node is x for b in outer.body for x in ast.walk(b))


def test_a_hook_has_own_try_except_exception_no_reraise(hook):
    exc = [h for h in hook.handlers
           if (isinstance(h.type, ast.Name) and h.type.id == "Exception")
           or (isinstance(h.type, ast.Tuple) and any(isinstance(e, ast.Name) and e.id == "Exception" for e in h.type.elts))]
    assert exc, "hook try must have an `except Exception` handler"
    for h in hook.handlers:
        assert not any(isinstance(n, ast.Raise) for n in ast.walk(h)), "hook except handler must not re-raise"
    assert not hook.finalbody or not any(isinstance(n, ast.Raise) for b in hook.finalbody for n in ast.walk(b))
    # the try is dedicated to the hook: it is not the pre-existing search try (which calls search_question/search)
    called = {_call_name(c)[0] for b in hook.body for c in ast.walk(b) if isinstance(c, ast.Call)}
    assert "search_question" not in called, "hook must not share the search try/except"


def test_b_flag_checked_before_any_get_cfg_or_get_registry(tree, hook, ctx):
    flag_lines = [n.lineno for n in ast.walk(tree) if _is_flag(n, ctx)]
    cfg_lines = [n.lineno for n in ast.walk(tree) if isinstance(n, ast.Call) and _call_name(n)[0] in CFG_FNS]
    assert cfg_lines, "hook never calls get_cfg/get_registry"
    assert min(flag_lines) < min(cfg_lines), "flag_enabled() must precede every get_cfg/get_registry call"
    # every cfg/registry call is guarded: inside the hook try, and under an If that tests the flag
    guarded = [n for b in hook.body if isinstance(b, ast.If) and any(_is_flag(c, ctx) for c in ast.walk(b.test))
               for n in ast.walk(b)]
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and _call_name(n)[0] in CFG_FNS:
            assert any(n is g for g in guarded), f"line {n.lineno}: get_cfg/get_registry not under the flag_enabled() guard"


def test_c_results_replaced_only_inside_hook_try(tree, hook, ctx):
    fn = next(f for f in ast.walk(tree) if isinstance(f, (ast.FunctionDef, ast.AsyncFunctionDef))
              and any(f is x or any(n is hook for n in ast.walk(f)) for x in [f]) and any(n is hook for n in ast.walk(f)))
    inside_assigns, hook_names = [], set()
    for b in hook.body:
        for n in ast.walk(b):
            if isinstance(n, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                tg = n.targets if isinstance(n, ast.Assign) else [n.target]
                for t in tg:
                    nm = _names(t, ast.Store)
                    hook_names |= nm - {"results"}
                    if "results" in nm:
                        inside_assigns.append(n)
    assert inside_assigns, "routed result must replace `results` inside the hook try"
    for n in ast.walk(fn):
        if isinstance(n, (ast.Assign, ast.AnnAssign, ast.AugAssign)) and not _inside(hook, n):
            tg = n.targets if isinstance(n, ast.Assign) else [n.target]
            if any("results" in _names(t, ast.Store) for t in tg) and n.value is not None:
                used = _names(n.value) | {x.value.id for x in ast.walk(n.value)
                                           if isinstance(x, ast.Attribute) and isinstance(x.value, ast.Name)}
                assert not (used & (hook_names | {m.split(".")[0] for m in ctx[1]} | ctx[2] | {BRANCHES})), (
                    f"line {n.lineno}: `results` replaced from hook-derived values outside the hook try")


def test_d_domain_branches_never_reaches_response_or_admission(tree, hook):
    for n in ast.walk(tree):
        refs = isinstance(n, ast.Name) and n.id == BRANCHES
        if not refs:
            continue
        if isinstance(n.ctx, ast.Store):
            continue
    parents = {}
    for p in ast.walk(tree):
        for c in ast.iter_child_nodes(p):
            parents[c] = p
    for n in ast.walk(tree):
        if not (isinstance(n, ast.Name) and n.id == BRANCHES):
            continue
        in_hook = _inside(hook, n) or any(n is x for h in hook.handlers for b in h.body for x in ast.walk(b))
        if isinstance(n.ctx, ast.Store):
            continue  # plain local binding only (attribute/subscript targets are checked below)
        assert in_hook, f"line {n.lineno}: {BRANCHES} read outside the hook try"
        # walk up: only allowed consumer is a logging call; never an Assign value, Return, Dict, keyword, subscript/attr target
        cur = n
        while cur in parents and not isinstance(parents[cur], (ast.Try, ast.FunctionDef, ast.AsyncFunctionDef, ast.If, ast.For)):
            par = parents[cur]
            if isinstance(par, ast.Assign):
                assert cur is not par.value or all(isinstance(t, ast.Name) and t.id == BRANCHES for t in par.targets), (
                    f"line {n.lineno}: {BRANCHES} assigned to something other than itself")
            assert not isinstance(par, (ast.Return, ast.Yield, ast.Dict, ast.keyword, ast.AugAssign, ast.AnnAssign)), (
                f"line {n.lineno}: {BRANCHES} flows into a return/dict/keyword/assignment")
            if isinstance(par, ast.Call):
                owner = _call_name(par)[1] or _call_name(par)[0]
                assert owner in {"logger", "log", "logging", "len", "bool"}, (
                    f"line {n.lineno}: {BRANCHES} passed to non-logging call {owner}")
            cur = par
    for n in ast.walk(tree):
        if isinstance(n, (ast.Attribute, ast.Subscript)) and isinstance(n.ctx, ast.Store):
            assert BRANCHES not in _names(n), f"line {n.lineno}: {BRANCHES} used in an attribute/subscript store"


def test_e_exactly_one_domain_routing_import_inside_retrieve_all(tree, ctx):
    nodes = ctx[0]
    assert len(nodes) == 1, f"expected exactly one domain_routing import, found {len(nodes)}"
    fn = next((f for f in ast.walk(tree) if isinstance(f, ast.FunctionDef) and f.name == "retrieve_all"), None)
    assert fn is not None, "retrieve_all not found"
    assert any(nodes[0] is x for x in ast.walk(fn)), "domain_routing import is not inside retrieve_all"

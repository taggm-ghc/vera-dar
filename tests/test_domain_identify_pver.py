"""ACQ-REBUILD-PERF-1021: DomainIdentifier.classify memoizes per-term domain factors and profile vectors keyed on
self._pver. Every public mutation path must bump _pver, and the memoized result must equal an unmemoized
computation. Plus a structural guard so a NEW mutation path without a bump fails. Synthetic text only."""
import ast
import copy
import inspect
from pathlib import Path

import pytest

from tests.test_dar_identify import BIO, DEV, SEC, fresh, make_doc, seed_topics
from vera import domain_identify as di
from vera.domain_identify import DomainIdentifier, _tf, tokenize
from vera.domain_registry import InMemoryDomainRegistry

PROBES = [("chunk", make_doc(BIO, 31, 30)), ("chunk", make_doc(SEC, 5, 30)), ("chunk", make_doc(DEV, 7, 30)),
          ("question", "protein enzyme genome"), ("question", "exploit injection firewall"),
          ("chunk", make_doc(BIO, 2, 12, extra=list(SEC[:6])))]


def reference(ident, text, kind="chunk", exclude=()):
    """classify() re-derived with NO memo (ctx=None in _dcos): same filters, floors and ratios."""
    ic = ident.icfg
    floor, cap = ((ic["question_min_sim"], ic["max_question_domains"]) if kind == "question"
                  else (ic["chunk_min_sim"], ic["max_domains_per_chunk"]))
    tf = _tf(tokenize(text, ic))
    if not tf:
        return []
    ident._sync_profiles()
    sc = sorted(((d, ident._dcos(tf, p)) for d, p in ident._active().items() if d not in exclude),
                key=lambda x: (-x[1], x[0]))
    out = [{"domain_id": d, "score": s} for d, s in sc if s >= floor and s > 0]
    ratio = float(ic["chunk_secondary_ratio" if kind == "chunk" else "question_secondary_ratio"])
    if out and ratio > 0:
        out = [out[0]] + [x for x in out[1:] if x["score"] >= ratio * out[0]["score"]]
    return out[: int(cap)]


def warm(ident):
    return [ident.classify(t, k) for k, t in PROBES]


def check(ident):
    for k, t in PROBES:
        assert ident.classify(t, k) == reference(ident, t, k), (k, t[:30])
    assert ident.classify(PROBES[0][1], "chunk", exclude=(1,)) == reference(ident, PROBES[0][1], "chunk", (1,))


def test_seeded_memo_equals_unmemoized_and_is_reused():
    reg, ident, _ = fresh()
    seed_topics(ident)
    warm(ident)
    memo = ident._dmemo
    assert memo is not None and memo["ver"] == ident._pver
    check(ident)
    assert ident._dmemo is memo  # no mutation: the same memo object is reused


def test_add_document_to_existing_domain_bumps_and_matches():
    reg, ident, _ = fresh()
    seed_topics(ident)
    warm(ident)
    v = ident._pver
    ident.add_document("bio_new", make_doc(BIO, 11), "t")  # joins a profile via _add_to_profile
    assert ident._pver > v
    check(ident)


def test_promotion_of_pending_documents_bumps_and_matches():
    reg, ident, _ = fresh()
    ident.add_document("sec0", make_doc(SEC, 0), "t")
    warm(ident)
    v = ident._pver
    for i, key in enumerate(("sec1", "dev0", "dev1", "bio0", "bio1")):
        words = SEC if key.startswith("sec") else DEV if key.startswith("dev") else BIO
        ident.add_document(key, make_doc(words, i), "t")
        warm(ident)
    assert ident._pver > v and len(reg.list_domains()) >= 2
    check(ident)


def test_registry_sync_of_a_new_domain_bumps_and_matches():
    reg, ident, _ = fresh()
    seed_topics(ident)
    warm(ident)
    v = ident._pver
    reg.register("Protein Genome Enzyme", "", origin="config")  # picked up by _sync_profiles on the next call
    ident.classify("protein enzyme", "question")
    assert ident._pver > v
    check(ident)


def test_restore_document_bumps_and_matches():
    reg, ident, _ = fresh()
    ids = seed_topics(ident)
    warm(ident)
    v = ident._pver
    ident.restore_document("old", make_doc(BIO, 20), [ids["bio"]])
    assert ident._pver > v
    check(ident)


def test_restore_into_a_fresh_identifier_matches_the_live_one():
    reg, ident, _ = fresh()
    ids = seed_topics(ident)
    reg2 = InMemoryDomainRegistry()
    for d in reg.list_domains():
        reg2.register(d.name, d.description, origin=d.origin)
    rebuilt = DomainIdentifier(reg2, copy.deepcopy(ident.icfg))
    warm(rebuilt)  # memo warmed on the EMPTY state
    v = rebuilt._pver
    for label, words in (("sec", SEC), ("dev", DEV), ("bio", BIO)):
        for i in range(2):
            rebuilt.restore_document(f"{label}{i}", make_doc(words, i), [ids[label]])
    assert rebuilt._pver > v
    check(rebuilt)
    assert [rebuilt.classify(t, k) for k, t in PROBES] == [reference(rebuilt, t, k) for k, t in PROBES]


def test_merge_alias_bumps_and_matches():
    reg, ident, _ = fresh()
    ids = seed_topics(ident)
    warm(ident)
    v = ident._pver
    assert ident.merge_alias(ids["dev"], ids["sec"]) is True
    assert ident._pver > v and ident.aliases == {ids["dev"]: ids["sec"]}
    check(ident)
    assert all(x["domain_id"] != ids["dev"] for _k, t in PROBES for x in ident.classify(t))  # alias is inactive
    v = ident._pver
    assert ident.merge_alias(ids["dev"], ids["sec"]) is False  # no-op merge: no stale state either way
    check(ident)
    assert ident._pver >= v


def test_memo_is_rebuilt_only_after_a_version_change():
    reg, ident, _ = fresh()
    seed_topics(ident)
    warm(ident)
    m1 = ident._dmemo
    ident.add_document("bio_more", make_doc(BIO, 13), "t")
    ident.classify(PROBES[0][1])
    assert ident._dmemo is not m1 and ident._dmemo["ver"] == ident._pver


# ------------------------------------------------------------------------------------------------ structural guard
GUARDED = ("_profiles", "_aliases")
MUTATORS = {"setdefault", "update", "pop", "popitem", "clear", "__setitem__", "__delitem__"}


def _is_self_attr(n, names):
    return isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == "self" and n.attr in names


def _root_guarded(n):
    """True when `n` is self._profiles / self._aliases or a subscript chain on one of them."""
    while isinstance(n, ast.Subscript):
        n = n.value
    return _is_self_attr(n, GUARDED)


def _mutates(fn):
    for n in ast.walk(fn):
        targets = []
        if isinstance(n, ast.Assign):
            targets = n.targets
        elif isinstance(n, (ast.AugAssign, ast.AnnAssign)):
            targets = [n.target]
        elif isinstance(n, ast.Delete):
            targets = n.targets
        for t in targets:
            for x in ast.walk(t):
                if (isinstance(x, ast.Subscript) and _root_guarded(x)) or (
                        _is_self_attr(x, GUARDED) and isinstance(x.ctx, (ast.Store, ast.Del))):
                    return True
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in MUTATORS
                and _root_guarded(n.func.value)):
            return True
    return False


def _bumps(fn):
    return any(isinstance(n, ast.AugAssign) and _is_self_attr(n.target, {"_pver"}) and isinstance(n.op, ast.Add)
               for n in ast.walk(fn))


def _self_calls(fn):
    return {n.func.attr for n in ast.walk(fn) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
            and isinstance(n.func.value, ast.Name) and n.func.value.id == "self"}


def unguarded_mutators(source: str, cls="DomainIdentifier") -> list:
    tree = ast.parse(source)
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
    fns = {f.name: f for f in node.body if isinstance(f, ast.FunctionDef)}
    ok = {n for n, f in fns.items() if _bumps(f)}
    changed = True
    while changed:  # a method that calls a bumping method also bumps
        changed = False
        for n, f in fns.items():
            if n not in ok and _self_calls(f) & ok:
                ok.add(n)
                changed = True
    return sorted(n for n, f in fns.items() if n != "__init__" and _mutates(f) and n not in ok)


def test_every_profile_or_alias_mutator_bumps_pver():
    src = Path(inspect.getsourcefile(di)).read_text(encoding="utf-8")
    assert unguarded_mutators(src) == []
    tree = ast.parse(src)  # the guard must see the known mutators, or it is vacuous
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "DomainIdentifier")
    seen = {f.name for f in cls.body if isinstance(f, ast.FunctionDef) and _mutates(f)}
    assert {"merge_alias", "_sync_profiles", "_add_to_profile", "_promote"} <= seen


@pytest.mark.parametrize("body", [
    "self._profiles[d] = {}", "self._profiles[d][t] = 1.0", "self._aliases.setdefault(a, b)",
    "self._profiles.update(x)", "self._aliases.pop(a)", "del self._aliases[a]", "self._profiles = {}",
    "self._profiles.clear()", "self._aliases[a] += 1",
])
def test_guard_detects_a_new_mutation_path_without_a_bump(body):
    bad = f"class DomainIdentifier:\n    def __init__(self):\n        self._profiles = {{}}\n        self._aliases = {{}}\n" \
          f"    def new_path(self, d, t, a, b, x):\n        {body}\n"
    assert unguarded_mutators(bad) == ["new_path"]
    good = bad + "        self._pver += 1\n"
    assert unguarded_mutators(good) == []
    via = bad + "    def caller(self):\n        self._bump()\n    def _bump(self):\n        self._pver += 1\n"
    assert unguarded_mutators(via) == ["new_path"]  # calling a bumper from ANOTHER method does not cover new_path

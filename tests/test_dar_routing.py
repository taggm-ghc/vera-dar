"""VERA-DAR T2 tests (directive 9D: 7, 8, 9, 10, 11, 15; owner section 4 a-j; vet-B RC-1, RC-3, RC-4).
DB-free and network-free: the search callable is a fake. Does NOT import vera.grounded_ask (it transitively imports
openai); that file is only parsed with ast."""
import ast
import importlib
import math
import re
from dataclasses import dataclass
from pathlib import Path

from vera import domain_routing as dr

SRC = Path(dr.__file__)
REGISTRY_SRC = SRC.parent / "domain_registry.py"
GROUNDED_ASK_SRC = SRC.parent / "grounded_ask.py"
CFG = {"enabled": True, "rrf_k": 60, "max_domain_branches": 5, "domains": [
    {"name": "Alpha", "terms": ["alphaterm"]}, {"name": "Beta", "terms": []}]}


@dataclass(frozen=True)
class D:
    domain_id: int
    name: str


class Reg:
    def __init__(self, *ds):
        self.ds = list(ds)

    def list_domains(self):
        return list(self.ds)


REG = Reg(D(1, "Alpha"), D(2, "Beta"), D(3, "Gamma"))


def item(u, **kw):
    return {"url": f"https://x.org/{u}", "title": u, "snippet": "s", **kw}


def fake_search(table, calls=None):
    """Domain queries are '<question> <domain name>', so match on the SUFFIX only (the question may contain every
    domain name)."""
    def f(q):
        if calls is not None:
            calls.append(q)
        for needle, res in table.items():
            if q.endswith(" " + needle):
                return list(res)
        return []
    return f


def urls(rs):
    return [r["url"].rsplit("/", 1)[1] for r in rs]


def strip(rs):
    return [{k: v for k, v in r.items() if k != "_domain_branches"} for r in rs]


# ---- 9D test 7 / owner (b): multiple domains, domain-specific branches included ----------------------------------
def test_7_multiple_domains_all_searched():
    calls = []
    base = [item("b1"), item("b2")]
    s = fake_search({"Alpha": [item("a1")], "Beta": [item("c1")]}, calls)
    out, tr = dr.route_search("about alphaterm and Beta", base, s, REG, CFG, num_results=5)
    assert tr["routing"] == "on" and tr["selected"] == [1, 2] and tr["searched"] == [1, 2]
    assert len(calls) == 2 and set(urls(out)) == {"b1", "b2", "a1", "c1"}
    assert tr["branch_counts"] == {"unrestricted": 2, "domain:1": 1, "domain:2": 1}


def test_b_enabled_routing_includes_domain_specific_branches():
    calls = []
    out, tr = dr.route_search("Alpha", [item("b1")], fake_search({"Alpha": [item("a1")]}, calls), REG, CFG,
                              num_results=5)
    assert calls == ["Alpha Alpha"] and "a1" in urls(out) and "domain:1" in tr["branch_counts"]
    q = dr.domain_query("Q", D(9, "Zed"), {"query_template": "{domain}: {question}"})
    assert q == "Zed: Q"


# ---- 9D test 8 / owner (e): dedup ----------------------------------------------------------------------------------
def test_8_dedup_overlaps_by_doi_or_url():
    base = [item("o", doi="10.1/X")]
    s = fake_search({"Alpha": [{"url": "https://other.org/z", "doi": "10.1/x"}, item("a1")],
                     "Beta": [item("a1/")]})
    out, tr = dr.route_search("Alpha Beta", base, s, REG, CFG, num_results=5)
    assert len(out) == 2 and tr["dedup_count"] == 2
    # the kept dict is the best-ranked domain copy (Beta rank 1 beats Alpha rank 2), whose url has a trailing slash
    a1 = next(r for r in out if r["url"].rstrip("/").endswith("/a1"))
    assert a1["_domain_branches"] == ["domain:1", "domain:2"]
    assert out[0]["_domain_branches"] == ["unrestricted", "domain:1"]


def test_8b_repeat_within_one_branch_scores_once():
    base = [item("b1")]
    once, _ = dr.route_search("Alpha", base, fake_search({"Alpha": [item("a1"), item("a2")]}), REG, CFG, num_results=5)
    twice, _ = dr.route_search("Alpha", base, fake_search({"Alpha": [item("a1"), item("a1"), item("a2")]}), REG, CFG,
                               num_results=5)
    assert urls(twice) == urls(once)  # a1 is not boosted by its own repeat, and a2 keeps its place


def test_e_dedup_keeps_unrestricted_record():
    base = [item("shared", title="BASE", extra="base")]
    s = fake_search({"Alpha": [item("only-a"), item("shared", title="DOMAIN", extra="domain")]})
    out, _ = dr.route_search("Alpha", base, s, REG, CFG, num_results=5)  # the domain copy ranks lower; try higher too
    sh = next(r for r in out if r["url"].endswith("shared"))
    assert sh["title"] == "BASE" and sh["extra"] == "base"
    s2 = fake_search({"Alpha": [item("shared", title="DOMAIN", extra="domain")]})
    out2, _ = dr.route_search("Alpha", [item("b0"), item("shared", title="BASE")], s2, REG, CFG, num_results=5)
    assert next(r for r in out2 if r["url"].endswith("shared"))["title"] == "BASE"  # domain rank 1 vs base rank 2


# ---- 9D test 9 / owner (c)(d): fallback and reserved slots ----------------------------------------------------------
def test_9_fallback_runs_and_includes_unclassified():
    base = [item("unclassified-1"), item("unclassified-2")]
    s = fake_search({"Alpha": [item("a1"), item("a2")]})
    out, tr = dr.route_search("Alpha", base, s, REG, CFG, num_results=2)
    assert {"unclassified-1", "unclassified-2"} <= set(urls(out))  # never silently dropped
    assert tr["fallback_contribution"] == 2 and tr["fallback_only"] == 2


def test_rc1_reserved_fallback_slots_survive_cut():
    base = [item(f"b{i}") for i in range(1, 6)]
    s = fake_search({"Alpha": [item(f"a{i}") for i in range(1, 6)], "Beta": [item(f"c{i}") for i in range(1, 6)]})
    n = 5
    out, tr = dr.route_search("Alpha Beta", base, s, REG, CFG, num_results=n)
    top = urls(out)[:n]
    need = math.ceil(n / 2)
    assert sum(u.startswith("b") for u in top) >= need and top[:need] == ["b1", "b2", "b3"]
    assert tr["reserved"] == need
    out2, tr2 = dr.route_search("Alpha Beta", base, s, REG, {**CFG, "min_fallback_slots": 1}, num_results=n)
    assert urls(out2)[0] == "b1" and tr2["reserved"] == 1


def test_c_at_least_two_slots_reserved_when_allowed():
    s = fake_search({"Alpha": [item(f"a{i}") for i in range(1, 8)]})
    for n in (3, 4, 5, 10):
        base = [item(f"b{i}") for i in range(1, 12)]
        out, tr = dr.route_search("Alpha", base, s, REG, CFG, num_results=n)
        assert tr["reserved"] >= 2 and urls(out)[:2] == ["b1", "b2"], n
        assert sum(u.startswith("b") for u in urls(out)[:n]) >= 2
    out, tr = dr.route_search("Alpha", [item("b1")], s, REG, CFG, num_results=5)  # not enough distinct fallback items
    assert tr["reserved"] == 1
    out, tr = dr.route_search("Alpha", [item("b1"), item("b2"), item("b3")], s, REG, {**CFG, "min_fallback_slots": 2},
                              num_results=2)  # num_results=2 leaves one slot for domain items
    assert tr["reserved"] == 1


def test_d_domain_results_cannot_displace_every_unrestricted_result():
    strong = fake_search({"Alpha": [item(f"a{i}") for i in range(1, 11)], "Beta": [item(f"c{i}") for i in range(1, 11)]})
    for n in (1, 2, 3, 5):
        base = [item(f"b{i}") for i in range(1, 4)]
        out, _ = dr.route_search("Alpha Beta", base, strong, REG, CFG, num_results=n)
        assert any(u.startswith("b") for u in urls(out)[:n]), n
        assert urls(out)[0] == "b1"


# ---- 9D test 10 / owner (h): provenance preserved, admission and contract untouched ----------------------------------
def test_10_provenance_preserved_and_no_cross_domain_filtering():
    base = [item("shared", licence="CC-BY", source_type="journal", extra="base")]
    s = fake_search({"Alpha": [item("shared", extra="domain", licence="other"), item("only-a")]})
    snapshot = [dict(b) for b in base]
    out, _ = dr.route_search("Alpha", base, s, REG, CFG, num_results=5)
    sh = next(r for r in out if r["url"].endswith("shared"))
    assert sh["extra"] == "base" and sh["licence"] == "CC-BY"  # fallback dict kept, no field merging (RC-3)
    assert {k: v for k, v in sh.items() if k != "_domain_branches"} == snapshot[0]
    assert sh["_domain_branches"] == ["unrestricted", "domain:1"]
    assert base == snapshot  # input dicts never mutated
    assert "only-a" in urls(out)  # cross-domain / domain-only item kept
    assert not set(dr.__dict__) & {"admit", "admission", "gate", "evidence_gate"}


def test_10b_module_has_no_admission_calls_and_no_sql():
    tree = ast.parse(SRC.read_text())
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | {
        n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    assert not [x for x in names if "admi" in x.lower() or "evidence_gate" in x.lower()]
    text = SRC.read_text().lower()
    for bad in ("select ", "insert into", "update ", "delete from", "vera_vjay", "{s}."):
        assert bad not in text


def test_h_items_keep_baseline_keys_plus_only_internal_key():
    base = [item("b1", licence="CC-BY", doi="10.1/a"), item("b2")]
    dom = [item("a1", source_type="journal", extra=1), item("b2")]
    out, _ = dr.route_search("Alpha", base, fake_search({"Alpha": dom}), REG, CFG, num_results=5)
    sources = {r["url"]: r for r in base + dom}
    for r in out:
        src = sources[r["url"]]
        assert set(r) - {"_domain_branches"} == set(src), r  # no admission or contract key added or removed
        assert strip([r])[0] == src
        assert isinstance(r["_domain_branches"], list)


def test_gate_modules_do_not_import_routing_and_routing_imports_no_gate():
    gate_re = re.compile(r"admi|gate|contract|licen[cs]e|evidence|claim")

    def imports_of(path):
        out = set()
        for n in ast.walk(ast.parse(path.read_text())):
            if isinstance(n, ast.Import):
                out |= {a.name for a in n.names}
            elif isinstance(n, ast.ImportFrom):
                out.add(n.module or "")
                out |= {f"{n.module}.{a.name}" for a in n.names}
        return out

    for path in (SRC, REGISTRY_SRC):  # direction 1: routing / registry import no admission or gate module
        assert not [m for m in imports_of(path) if gate_re.search(m.lower())], path.name
    gates = [p for p in SRC.parent.rglob("*.py") if gate_re.search(p.stem.lower())]
    assert gates  # the scan found the gate modules, so the check below is not vacuous
    for p in gates:  # direction 2: no admission or gate module imports domain_routing / domain_registry
        assert not [m for m in imports_of(p) if "domain_routing" in m or "domain_registry" in m], p.name


# ---- 9D test 11 / owner (a): flag OFF and single-domain passthrough ------------------------------------------------
def test_11_off_and_single_domain_identical_to_baseline():
    base = [item("b1"), item("b2")]
    s = fake_search({"Alpha": [item("a1")]})
    for cfg in ({}, {**CFG, "enabled": False}, {**CFG, "enabled": "true"}, {**CFG, "enabled": 1}):
        out, tr = dr.route_search("Alpha", base, s, REG, cfg)
        assert out is base and tr["routing"] == "off"
    out, tr = dr.route_search("nothing relevant", base, s, REG, CFG)
    assert out is base and tr["routing"] == "no_domain"
    out, tr = dr.route_search("Alpha", base, fake_search({}), REG, CFG)  # domain branch adds nothing
    assert out is base
    same = fake_search({"Alpha": list(base)})  # single domain returning exactly the baseline list
    out, _ = dr.route_search("Alpha", base, same, REG, CFG, num_results=2)
    assert urls(out) == urls(base)
    assert strip(out) == base
    bad = {k: v for k, v in CFG.items() if k != "rrf_k"}  # rrf_k comes from config, never a default
    out, tr = dr.route_search("Alpha", base, s, REG, bad)
    assert out is base and tr["routing"] == "config_error"


def test_a_disabled_routing_returns_baseline_and_never_calls_search():
    calls = []
    base = [item("b1")]
    out, tr = dr.route_search("Alpha", base, fake_search({"Alpha": [item("a1")]}, calls), REG, {**CFG, "enabled": False})
    assert out is base and calls == [] and tr == {"routing": "off"}


def test_empty_registry_is_passthrough():
    calls = []
    base = [item("b1")]
    out, tr = dr.route_search("Alpha Beta", base, fake_search({"Alpha": [item("a1")]}, calls), Reg(), CFG)
    assert out is base and tr["routing"] == "no_domain" and calls == []


def test_max_domain_branches_caps_searched_but_not_selected():
    out, tr = dr.route_search("Alpha Beta", [item("b1")], fake_search({}), REG, {**CFG, "max_domain_branches": 1})
    assert tr["selected"] == [1, 2] and tr["searched"] == [1]


# ---- owner (f): determinism --------------------------------------------------------------------------------------------
def test_f_dedup_and_ranking_deterministic():
    def run():
        base = [item(f"b{i}") for i in range(1, 5)]
        s = fake_search({"Alpha": [item("a1"), item("b2"), item("a2"), item("a3")],
                         "Beta": [item("a3"), item("c1"), item("a1"), item("c2")]})
        return dr.route_search("Alpha Beta", base, s, REG, CFG, num_results=5)
    first = run()
    for _ in range(3):
        assert run() == first  # identical order, records and trace


# ---- owner (g): routing errors return the original results ----------------------------------------------------------------
def test_g_search_raises_returns_original_results():
    def boom(q):
        raise RuntimeError("x")
    base = [item("b1")]
    out, tr = dr.route_search("Alpha", base, boom, REG, CFG)
    assert out is base and tr["failed_branches"] == ["domain:1"]
    ok = fake_search({"Beta": [item("c1")]})
    def half(q):
        return boom(q) if q.endswith(" Alpha") else ok(q)
    out, tr = dr.route_search("Alpha Beta", base, half, REG, CFG, num_results=5)
    assert tr["failed_branches"] == ["domain:1"] and set(urls(out)) == {"b1", "c1"}  # fallback stays, other branch used


def test_g2_registry_or_unexpected_error_returns_original_results():
    class BadReg:
        def list_domains(self):
            raise ValueError("registry down")
    base = [item("b1")]
    out, tr = dr.route_search("Alpha", base, fake_search({}), BadReg(), CFG)
    assert out is base and tr["routing"] == "error" and tr["error_class"] == "ValueError"


def test_flag_and_cfg_files(tmp_path):
    p = tmp_path / "c.json"
    assert dr.flag_enabled(p) is False  # missing = OFF
    p.write_text("{not json")
    assert dr.flag_enabled(p) is False
    p.write_text('{"enabled": true}')
    assert dr.flag_enabled(p) is True


# ---- owner (j): trace ---------------------------------------------------------------------------------------------------
def test_j_trace_records_branches_counts_fallback_and_dedup():
    base = [item("b1"), item("b2")]
    s = fake_search({"Alpha": [item("b1"), item("a1")], "Beta": [item("a1"), item("c1")]})
    out, tr = dr.route_search("Alpha Beta", base, s, REG, CFG, num_results=5)
    assert tr["routing"] == "on" and tr["selected"] == [1, 2] and tr["searched"] == [1, 2]
    assert tr["branch_counts"] == {"unrestricted": 2, "domain:1": 2, "domain:2": 2}
    assert tr["fallback_contribution"] == 2 and tr["fallback_only"] == 1  # b2 came only from the fallback
    assert tr["dedup_count"] == 2 and tr["failed_branches"] == [] and tr["reserved"] == 2
    assert "limitation" in tr and len(out) == 4


# ---- owner (i), 9D test 15 and RC-4: injection, no search import, no engine / HTTP library ------------------------------
BANNED = ["sqlalchemy", "psycopg2", "psycopg", "asyncpg", "pg8000", "requests", "httpx", "urllib3", "urllib.request",
          "http.client", "socket", "ssl", "aiohttp", "fastapi", "starlette", "uvicorn", "openai", "groq", "anthropic",
          "dotenv", "subprocess", "multiprocessing", "vera.search_and_fetch", "vera.grounded_ask"]


def _is_banned(name):
    return any(name == b or name.startswith(b + ".") for b in BANNED)


def _imports(path):
    out = set()
    for n in ast.walk(ast.parse(path.read_text())):
        if isinstance(n, ast.Import):
            out |= {a.name for a in n.names}
        elif isinstance(n, ast.ImportFrom):
            out.add(n.module or "")
            out |= {f"{n.module}.{a.name}" for a in n.names}
    return out


def test_i_search_is_injected_never_imported():
    assert not [m for m in _imports(SRC) if "search_and_fetch" in m]
    seen = []
    dr.route_search("Alpha", [item("b1")], lambda q: seen.append(q) or [], REG, CFG)
    assert seen == ["Alpha Alpha"]  # the injected callable is what ran


def test_15_no_db_or_network_imports(monkeypatch):
    for path in (SRC, REGISTRY_SRC):  # B2's registry is imported by routing, so it is scanned too
        assert not [m for m in _imports(path) if _is_banned(m)], path.name
    mods = importlib.import_module("sys").modules
    for name in ("vera.domain_routing", "vera.domain_registry"):
        monkeypatch.delitem(mods, name, raising=False)
    before = set(mods)
    importlib.import_module("vera.domain_registry")
    importlib.import_module("vera.domain_routing")
    newly = sorted(set(mods) - before)
    assert not [m for m in newly if _is_banned(m)], newly


# ---- hook parity: make_search_fn must pass the same kwargs as the baseline call in grounded_ask.retrieve_all -----------
def _kwargs_of_search_question_call(tree):
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
             and n.func.id == "search_question"]
    assert len(calls) == 1, len(calls)
    return {k.arg: ast.unparse(k.value) for k in calls[0].keywords}


def test_make_search_fn_kwarg_parity_with_grounded_ask():
    ga = ast.parse(GROUNDED_ASK_SRC.read_text())
    retrieve_all = next(n for n in ast.walk(ga) if isinstance(n, ast.FunctionDef) and n.name == "retrieve_all")
    mine = ast.parse(SRC.read_text())
    msf = next(n for n in ast.walk(mine) if isinstance(n, ast.FunctionDef) and n.name == "make_search_fn")
    assert _kwargs_of_search_question_call(msf) == _kwargs_of_search_question_call(retrieve_all)


# ---- DARINT L4: content-derived routing, aliases, domain terms --------------------------------------------------------
@dataclass(frozen=True)
class DD:
    domain_id: int
    name: str
    origin: str = "derived"


class ContentReg:
    """Fake registry: identify_question stands in for identifier.classify(question, 'question') (content, not names)."""

    def __init__(self, domains, hits, terms=None, aliases=None):
        self.ds, self.hits, self.terms, self.aliases = list(domains), hits, terms, aliases or {}

    def list_domains(self):
        return list(self.ds)

    def identify_question(self, q):
        return list(self.hits)

    def domain_terms(self, domain_id, n):
        if self.terms is None:
            raise AttributeError
        return self.terms[domain_id][:n]


def test_derived_domain_whose_name_is_not_in_the_question_is_selected_by_content():
    reg = ContentReg([DD(1, "zeta quark flux"), DD(2, "eta moth larva")], hits=[2])
    ident = dr.identify_domains("how do caterpillars develop", reg, {"domains": []})
    assert [d.domain_id for d in ident["selected"]] == [2] and ident["method"] == "content"
    calls = []
    out, tr = dr.route_search("how do caterpillars develop", [item("b1")], fake_search({"eta moth larva": [item("m1")]}, calls),
                              reg, {"enabled": True, "rrf_k": 60, "max_domain_branches": 2}, num_results=5)
    assert tr["routing"] == "on" and tr["searched"] == [2] and "m1" in urls(out) and "b1" in urls(out)
    assert calls == ["how do caterpillars develop eta moth larva"]


def test_domain_query_uses_profile_terms_for_derived_and_name_otherwise():
    reg = ContentReg([DD(1, "n1"), DD(2, "n2", origin="config")], hits=[], terms={1: ["alpha", "beta", "gamma", "delta"], 2: ["x"]})
    cfg = {"identification": {"name_terms": 3}}
    assert dr.domain_query("Q", reg.ds[0], cfg, reg) == "Q alpha beta gamma"
    assert dr.domain_query("Q", reg.ds[1], cfg, reg) == "Q n2"  # not derived: name
    assert dr.domain_query("Q", reg.ds[0], cfg) == "Q n1"  # no registry: name
    bad = ContentReg([DD(1, "n1")], hits=[], terms=None)  # domain_terms raises -> name
    assert dr.domain_query("Q", bad.ds[0], cfg, bad) == "Q n1"
    assert dr.domain_query("Q", DD(3, "n3"), cfg, ContentReg([], [], terms={3: []})) == "Q n3"  # empty terms -> name


def test_alias_is_not_double_counted_and_canonical_replaces_it():
    reg = ContentReg([DD(1, "one"), DD(2, "two"), DD(3, "three")], hits=[2, 1, 3], aliases={2: 1})
    ident = dr.identify_domains("unrelated words", reg, {"domains": []})
    assert [d.domain_id for d in ident["selected"]] == [1, 3]  # alias 2 -> canonical 1 once


def test_content_branches_capped_and_fallback_always_present_never_excluded():
    reg = ContentReg([DD(i, f"dom{i}") for i in range(1, 6)], hits=[1, 2, 3, 4, 5])
    calls = []
    s = fake_search({f"dom{i}": [item(f"d{i}")] for i in range(1, 6)}, calls)
    base = [item("b1"), item("b2")]
    out, tr = dr.route_search("zzz", base, s, reg, {"enabled": True, "rrf_k": 60, "max_domain_branches": 2}, num_results=5)
    assert tr["searched"] == [1, 2] and len(calls) == 2 and tr["selected"] == [1, 2, 3, 4, 5]
    assert {"b1", "b2"} <= set(urls(out)) and "unrestricted" in tr["branch_counts"]
    # every domain branch empty or failing: exactly the baseline
    out2, tr2 = dr.route_search("zzz", base, fake_search({}), reg, {"enabled": True, "rrf_k": 60, "max_domain_branches": 2})
    assert out2 is base
    # nothing identified: baseline by identity
    out3, tr3 = dr.route_search("zzz", base, s, ContentReg(reg.ds, hits=[]), {"enabled": True, "rrf_k": 60})
    assert out3 is base and tr3["routing"] == "no_domain"

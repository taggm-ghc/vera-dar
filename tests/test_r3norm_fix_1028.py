"""R3NORM-FIX-1028: review fixes D1-D4 (admission legacy lookup, full-document rebuild accepts legacy rows,
canonical_id_for loops lookup_keys, query handling). Offline."""
import contextlib
import copy

import pytest

from tests import fulldoc_bundle_synth as synth
from vera import corpus_admission as ca, dar_store, fulldoc_db
from vera import url_key as uk
from vera.canonical_lookup import RoKeyLookup


# ---- D1: WoCorpusStore.admit checks legacy spellings before the INSERT --------------------------------------------
class _Eng:
    @contextlib.contextmanager
    def begin(self):
        yield object()


class _Lookup:
    def __init__(self, stored):
        self.stored, self.asked = set(stored), []

    def any_known(self, keys):
        self.asked.append(list(keys))
        return any(k in self.stored for k in keys)


def _store(stored):
    st = ca.WoCorpusStore(_Eng(), "Q", _Lookup(stored))
    st._run_id = 1
    sqls = []

    def ins(conn, sql, params):
        sqls.append(sql)
        return len(sqls)

    st._insert_new, st._one = ins, lambda conn, sql, params: 1
    return st, sqls


def _cand(url, doi=None):
    return {"url": url, "doi": doi, "canonical_url": ca.canonical_url({"url": url, "doi": doi}), "snippet": "abs",
            "gate_a_decision": "fetch", "provenance": {}}


STORED_AND_VARIANTS = [
    ("https://arxiv.org/abs/2101.00001", "https://arxiv.org/pdf/2101.00001.pdf"),
    ("https://arxiv.org/abs/2101.00001", "http://arxiv.org/abs/2101.00001"),
    ("https://example.com/a", "HTTPS://Example.COM/a"),
    ("https://example.com/a", "https://example.com/a#frag"),
    ("https://example.com/a", "https://example.com/a/"),
    ("https://example.com/a?x=1", "https://example.com/a?utm_source=n&x=1"),
    ("https://example.com/a?a=1&b=2", "https://example.com/a?b=2&a=1"),
    ("https://example.com/a", "https://example.com:443/a"),
]


@pytest.mark.parametrize("stored,variant", STORED_AND_VARIANTS)
def test_admit_known_for_variant_spellings_no_insert(stored, variant):
    st, sqls = _store({stored})
    assert st.admit(_cand(variant)) == "known"
    assert not any("canonical_sources" in s for s in sqls)  # the canonical_sources INSERT never ran


def test_admit_known_when_row_is_under_legacy_spelling():
    """The stored row is the OLD spelling (trailing-slash stripped only); the new key differs from it."""
    st, sqls = _store({"HTTPS://Example.com/a"})
    assert st.admit(_cand("HTTPS://Example.com/a/")) == "known"
    assert not any("canonical_sources" in s for s in sqls)


def test_admit_known_by_doi_legacy_key():
    st, sqls = _store({"https://doi.org/10.1/x"})
    assert st.admit(_cand("https://pub.example/p", doi="10.1/X")) == "known"
    assert not any("canonical_sources" in s for s in sqls)


def test_admit_inserts_when_nothing_matches():
    st, sqls = _store({"https://example.com/other"})
    assert st.admit(_cand("https://example.com/a")) == "admitted"
    assert any("canonical_sources" in s for s in sqls)
    assert st.key_lookup.asked and st.key_lookup.asked[0][0] == "https://example.com/a"  # new key first


def test_admit_without_lookup_fails_closed_no_canonical_insert():
    """OD-1028(8): no RO lookup -> no canonical_sources INSERT, distinct outcome."""
    st = ca.WoCorpusStore(_Eng(), "Q")
    sqls = []
    st._run_id, st._one = 1, (lambda c, s, p: 1)
    st._insert_new = lambda c, s, p: sqls.append(s) or 5
    assert st.key_lookup is None and st.admit(_cand("https://example.com/a")) == "skipped_no_ro_lookup"
    assert not any("canonical_sources" in s or "sources (" in s for s in sqls)


def test_store_from_env_wires_ro_lookup(monkeypatch, caplog):
    monkeypatch.setattr(ca, "_STORE", {})
    urls = []
    cfg = {"enabled": True, "db_url_env": "W", "question_text": "q"}
    st = ca.store_from_env(cfg, env={"W": "wo", "VERA_DB_URL_RO": "ro"}, engine_factory=lambda u, **kw: urls.append(u) or u)
    assert isinstance(st.key_lookup, RoKeyLookup) and urls == ["ro", "wo"]
    monkeypatch.setattr(ca, "_STORE", {})
    with caplog.at_level("WARNING", logger=ca.logger.name):
        assert ca.store_from_env(cfg, env={"W": "wo"}, engine_factory=lambda u, **kw: u).key_lookup is None
    warns = [r for r in caplog.records if "read-only lookup" in r.getMessage()]
    assert len(warns) == 1 and "wo" not in warns[0].getMessage().split()


def test_ro_key_lookup_sql_is_select_any():
    seen = {}

    class C:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def execute(self, stmt, params):
            seen["sql"], seen["p"] = str(stmt), params
            return type("R", (), {"first": lambda s: (1,)})()

    class E:
        def connect(self):
            return C()

    assert RoKeyLookup(E()).any_known(["a", "a", "b"]) is True
    assert seen["p"] == {"ks": ["a", "b"]} and "ANY(:ks)" in seen["sql"] and "vera_vjay.canonical_sources" in seen["sql"]
    assert RoKeyLookup(E()).any_known([]) is False


# ---- D3: canonical_id_for loops lookup_keys -----------------------------------------------------------------------
def test_canonical_id_for_tries_legacy_keys():
    rd = dar_store.DarReader.__new__(dar_store.DarReader)
    asked = []
    stored = {"https://example.com/": 9}

    def rows(sql, params):
        asked.append(params["u"])
        return [(stored[params["u"]],)] if params["u"] in stored else []

    rd._rows = rows
    assert rd.canonical_id_for("https://example.com") == 9  # old m2 key kept '/' for the root
    assert asked[0] == "https://example.com" and "https://example.com/" in asked
    assert rd.canonical_id_for("https://nowhere.example/z") is None
    assert rd.canonical_id_for("https://example.com/q", {"url": "HTTPS://Example.com/q/"}) is None


def test_canonical_id_for_src_legacy_spelling():
    rd = dar_store.DarReader.__new__(dar_store.DarReader)
    rd._rows = lambda sql, p: [(4,)] if p["u"] == "HTTPS://Example.com/q" else []
    assert rd.canonical_id_for("https://example.com/q", {"url": "HTTPS://Example.com/q/"}) == 4


# ---- D4: query handling -------------------------------------------------------------------------------------------
@pytest.mark.parametrize("a,b", [
    ("https://e.com/p?id=2&id=1", "https://e.com/p?id=1&id=2"),
    ("https://e.com/p?q=a+b", "https://e.com/p?q=a%20b"),
    ("https://e.com/p?q=a/b", "https://e.com/p?q=a%2Fb"),
    ("https://e.com/p?flag", "https://e.com/p?flag="),
])
def test_query_distinctions_kept(a, b):
    assert uk.normalize_url(a) != uk.normalize_url(b)


def test_query_sorted_by_key_only_stable_and_raw():
    assert uk.normalize_url("https://e.com/p?b=2&id=2&a=1&id=1") == "https://e.com/p?a=1&b=2&id=2&id=1"
    assert uk.normalize_url("https://e.com/p?q=a+b&x=%2f&UTM_x=1&fbclid=2") == "https://e.com/p?q=a+b&x=%2F"
    assert uk.normalize_url("https://e.com/p?%75tm_source=1&k=v") == "https://e.com/p?k=v"  # encoded tracking name
    assert uk.normalize_url("https://e.com/p?q=a%20b") == "https://e.com/p?q=a%20b"


# ---- D2: full-document rebuild accepts legacy-stored canonical_url (incl. pinned runs) -----------------------------
LANDING = {"syn_epi": "https://Example.org/Paper/1/", "syn_clim": "https://example.org/paper/2?utm_source=n&b=1",
           "syn_repro": "https://example.net/Pdf/3/?x=2&a=1"}


def _world(tmp_path, pins=None):
    """DB-shaped rows (as written by the ingest script); DOI documents whose landing URLs are unnormalised, so the stored
    key is the DOI form and the legacy spellings are the landing-URL ones."""
    from tests.test_fulldoc_write_atomic import _script
    root = synth.write_bundle(tmp_path / "b", items_over={d: {"landing_url": u} for d, u in LANDING.items()})
    cfg = synth.synth_cfg(pins)
    rows, report = _script().build_plan(root, cfg, target=700)
    sources, spans = [], []
    for sid, r in enumerate(rows, 1):
        src = r["source"]
        sources.append({"source_id": sid, "canonical_url": src["canonical_url"], "content": src["content"],
                        "content_hash": src["content_hash"], "provenance": copy.deepcopy(src["provenance"])})
        spans += [{"source_id": sid, "start_index": sp["start_index"], "end_index": sp["end_index"], "text": sp["text"]}
                  for sp in r["spans"]]
    return cfg, sources, spans


def _as_legacy(sources):
    """Re-key every row under the pre-R3NORM m2_runner spelling of its landing URL (a legacy key of the same source)."""
    for s_ in sources:
        item = s_["provenance"]["item"]
        src = {"doi": item.get("doi"), "url": item["landing_url"]}
        s_["canonical_url"] = uk.legacy_keys(src)[-1]
    return sources


def test_rebuild_new_key_rows_ok_and_legacy_differs(tmp_path):
    cfg, sources, spans = _world(tmp_path)
    assert fulldoc_db.load_from_rows(copy.deepcopy(sources), spans, cfg).status == "ok"
    old = [r["canonical_url"] for r in _as_legacy(copy.deepcopy(sources))]
    assert all(o != s_["canonical_url"] for o, s_ in zip(old, sources))  # the test really exercises a legacy spelling


def test_rebuild_accepts_legacy_canonical_url_unpinned(tmp_path):
    cfg, sources, spans = _world(tmp_path)
    c = fulldoc_db.load_from_rows(_as_legacy(sources), spans, cfg)
    assert c.status == "ok" and len(c.acquired) == 3, c.unusable


def test_rebuild_rejects_unrelated_canonical_url(tmp_path):
    cfg, sources, spans = _world(tmp_path)
    sources[0]["canonical_url"] = "https://elsewhere.example/not-this"
    c = fulldoc_db.load_from_rows(sources, spans, cfg)
    assert c.status == "ok" and {u["code"] for u in c.unusable} == {fulldoc_db.R_DBROW} and len(c.acquired) == 2


def test_rebuild_pinned_run_with_legacy_rows_stays_usable(tmp_path):
    cfg, sources, spans = _world(tmp_path)
    clean = fulldoc_db.load_from_rows(copy.deepcopy(sources), spans, cfg)
    docs = {a["document_id"]: {"text_sha256": a["text_sha256"], "licence_id": "CC-BY-4.0", "licence_decision": "allow"}
            for a in clean.acquired}
    lic = next(iter(clean.acquired))
    b = sources[0]["provenance"]["bundle"]
    pcfg = copy.deepcopy(cfg)
    pcfg["pins"] = {"manifest_sha256": b["manifest_sha256"], "sums_sha256": b["sums_sha256"], "documents": docs}
    # licence_id in the pin must equal the derived one: take it from the provenance instead of a literal
    from vera import fulltext_loader
    for s_ in sources:
        did = s_["provenance"]["document_id"]
        rec = fulltext_loader.prepare_from_parts(s_["provenance"]["item"], "".join(sp["text"] for sp in spans
                                                 if sp["source_id"] == s_["source_id"]), s_["provenance"]["meta"],
                                                 fulltext_loader.load_config(), now=s_["provenance"].get("retrieved_at"))
        docs[did]["licence_id"] = ((rec["provenance"].get("licence") or {}).get("content") or {}).get("id")
    ok = fulldoc_db.load_from_rows(copy.deepcopy(sources), spans, pcfg)
    assert ok.status == "ok" and len(ok.acquired) == 3, ok.unusable
    legacy = fulldoc_db.load_from_rows(_as_legacy(sources), spans, pcfg)
    assert legacy.status == "ok" and len(legacy.acquired) == 3, legacy.unusable
    sources[1]["canonical_url"] = "https://elsewhere.example/x"  # a foreign key still disables the whole pinned run
    bad = fulldoc_db.load_from_rows(sources, spans, pcfg)
    assert bad.status == "corpus_unusable" and not bad.acquired

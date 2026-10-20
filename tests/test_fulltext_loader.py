"""DARINT G1/G2: vera/fulltext_loader.py + scripts/ingest_fulltext.py with SYNTHETIC fixtures and FAKE engines.
No DB, no network, no env values (the DB end-to-end runs separately on an ephemeral cluster)."""
import hashlib
import json
from pathlib import Path

import pytest
from sqlalchemy.exc import IntegrityError

from scripts import ingest_fulltext as cli
from tests import fulltext_synth as fs
from vera import fulltext_loader as fl

ROOT = Path(__file__).resolve().parent.parent
CFG = fl.load_config()


def corpus(tmp_path, **kw):
    m = fs.write_corpus(tmp_path, **kw)
    assert cli.main(["convert", str(m), "--raw-dir", str(tmp_path / "raw")]) in (0, 2)
    return m


def items(m):
    return json.loads(Path(m).read_text())["sources"]


def code(e):
    return e.value.code


# ------------------------------------------------------------------------------------ filters
def test_prepare_ok_and_provenance(tmp_path):
    m = corpus(tmp_path)
    rec = fl.prepare(items(m)[0], m.parent, CFG, now="2026-10-18T00:00:00+00:00")
    p = rec["provenance"]
    assert p["stored_text_kind"] == "full_text" and rec["canonical_url"] == "https://doi.org/10.9999/syn.epi"
    assert p["licence_decision"] == "allow" and p["fulltext_format"] == "jats-xml"
    assert p["sections"] and p["citations"] and p["references"]
    assert rec["content_hash"] == hashlib.sha256(rec["text"].encode()).hexdigest()
    assert "SYNTHETIC-HINT" not in json.dumps(rec)  # domain_hint is never carried
    doc = fl.doc_for_dar(rec, 5, 7, CFG)
    assert "SYNTHETIC-HINT" not in json.dumps(doc) and doc["admission_passed"] is True
    assert doc["db_source_id"] == 5 and doc["db_canonical_source_id"] == 7 and doc["stored_text_kind"] == "full_text"


@pytest.mark.parametrize("over,want", [
    ({"doi": "(DOI to confirm)"}, "R-PLACEHOLDER"),
    ({"title": "(title to confirm at fetch)"}, "R-PLACEHOLDER"),
    ({"doi": "not-a-doi"}, "R-DOI"),
    ({"verifier_status": "UNVERIFIED"}, "R-VERIFIER"),
    ({"read_status": "SUMMARY"}, "R-READ"),
    ({"fulltext_format": "pdf"}, "R-FORMAT"),
    ({"fetched_at": None}, "R-FIELD"),
    ({"licence": "CC BY-ND 4.0", "licence_url": "https://creativecommons.org/licenses/by-nd/4.0/"}, "R-LIC-REJECT"),
    ({"licence": "Strictly for fee"}, "R-LIC-REJECT"),
    ({"fulltext_url": "https://arxiv.org/pdf/2101.00001"}, "R-LIC-REJECT"),
    ({"fulltext_url": "https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi?id=PMC1 (OA service gives the link)"}, "R-URL"),
    ({"landing_url": "ftp://example.invalid/x"}, "R-URL"),
])
def test_item_filters_refuse(tmp_path, over, want):
    m = corpus(tmp_path)
    with pytest.raises(fl.Refused) as e:
        fl.prepare({**items(m)[0], **over}, m.parent, CFG)
    assert code(e) == want


def test_two_placeholder_doi_items_both_refused(tmp_path):
    m = corpus(tmp_path)
    a, b = items(m)[:2]
    for it in ({**a, "doi": "(DOI to confirm)"}, {**b, "doi": "(DOI to confirm)"}):
        with pytest.raises(fl.Refused) as e:
            fl.prepare(it, m.parent, CFG)
        assert code(e) == "R-PLACEHOLDER"


def test_plain_nc_is_allowed(tmp_path):
    m = corpus(tmp_path, raw_over={"syn_epi": fs.jats("syn_epi", "epi", "10.9999/syn.epi",
                                                      licence_url="https://creativecommons.org/licenses/by-nc/4.0/")})
    it = {**items(m)[0], "licence": "CC BY-NC 4.0", "licence_url": "https://creativecommons.org/licenses/by-nc/4.0/"}
    assert fl.prepare(it, m.parent, CFG)["provenance"]["licence_decision"] == "allow"


def test_missing_licence_is_held():
    it = fs.item("x", "10.9999/x", licence=None, licence_url=None)
    with pytest.raises(fl.Refused) as e:
        fl.check_licence(it, {}, CFG)
    assert code(e) == "R-LIC-HOLD"


def test_embedded_licence_nd_rejects_and_mismatch_holds(tmp_path):
    nd = fs.jats("syn_epi", "epi", "10.9999/syn.epi", licence_url="https://creativecommons.org/licenses/by-nd/4.0/")
    m = corpus(tmp_path, raw_over={"syn_epi": nd})
    with pytest.raises(fl.Refused) as e:
        fl.prepare(items(m)[0], m.parent, CFG)
    assert code(e) == "R-LIC-REJECT"
    v3 = fs.jats("syn_clim", "clim", "10.9999/syn.clim", licence_url="https://creativecommons.org/licenses/by/3.0/")
    m2 = corpus(tmp_path / "b", raw_over={"syn_clim": v3})
    with pytest.raises(fl.Refused) as e:
        fl.prepare(items(m2)[1], m2.parent, CFG)
    assert code(e) == "R-LIC-MISMATCH"


def test_abstract_only_and_short_and_ratio_refused(tmp_path):
    m = corpus(tmp_path, raw_over={"syn_epi": fs.jats("syn_epi", "epi", "10.9999/syn.epi", with_body=False)})
    with pytest.raises(fl.Refused) as e:
        fl.prepare(items(m)[0], m.parent, CFG)
    assert code(e) == "R-ABSTRACT"
    good = items(m)[1]
    with pytest.raises(fl.Refused) as e:
        fl.prepare(good, m.parent, dict(CFG, min_fulltext_chars=10 ** 7))
    assert code(e) == "R-SHORT"
    with pytest.raises(fl.Refused) as e:
        fl.prepare(good, m.parent, dict(CFG, min_body_to_abstract_ratio=1000))
    assert code(e) == "R-ABSTRACT"


def test_edited_text_refused(tmp_path):
    m = corpus(tmp_path)
    it = items(m)[0]
    p = m.parent / it["text_file"]
    p.write_text(p.read_text() + " tampered")
    with pytest.raises(fl.Refused) as e:
        fl.prepare(it, m.parent, CFG)
    assert code(e) == "R-HASH"


def test_text_file_outside_manifest_dir_refused(tmp_path):
    m = corpus(tmp_path)
    with pytest.raises(fl.Refused) as e:
        fl.prepare({**items(m)[0], "text_file": "../../etc/passwd"}, m.parent, CFG)
    assert code(e) == "R-TEXT"


def test_non_document_kind_refused(tmp_path):
    m = corpus(tmp_path)
    with pytest.raises(fl.Refused) as e:
        fl.prepare(items(m)[0], m.parent, dict(CFG, stored_text_kind="abstract_metadata"))
    assert code(e) == "R-KIND"


def test_routing_override_is_in_memory_only():
    path = ROOT / "config" / "domain_routing.json"
    before = path.read_bytes()
    shipped = json.loads(before)
    out = fl.ingest_routing_config(cfg=CFG)
    assert out["enabled"] is True and path.read_bytes() == before
    assert out["corpus"]["admission_hook_enabled"] == shipped["corpus"]["admission_hook_enabled"]
    assert out["corpus"]["stored_branch_enabled"] == shipped["corpus"]["stored_branch_enabled"]
    assert fl.ingest_routing_config(cfg={"dar": {"enable_for_ingest": False}})["enabled"] == shipped["enabled"]


def test_no_update_delete_or_ddl_in_new_code():
    for f in ("vera/fulltext_loader.py", "scripts/ingest_fulltext.py", "vera/jats_text.py"):
        src = (ROOT / f).read_text().upper()
        for bad in ("UPDATE {S}", "DELETE FROM", "TRUNCATE ", "DROP TABLE", "DROP SCHEMA", "ALTER TABLE",
                    "CREATE TABLE", "GRANT "):
            assert bad not in src, (f, bad)


# ------------------------------------------------------------------------------------ fake engines
class _Res:
    def __init__(self, rows):
        self.rows = rows

    def first(self):
        return self.rows[0] if self.rows else None

    def fetchall(self):
        return list(self.rows)


class _Conn:
    def __init__(self, eng):
        self.eng = eng

    def __enter__(self):
        return self

    def __exit__(self, et, *a):
        self.eng.events.append("rollback" if et else "commit")
        return False

    def begin_nested(self):
        conn = self

        class SP:
            def commit(self):
                conn.eng.events.append("sp_commit")

            def rollback(self):
                conn.eng.events.append("sp_rollback")
        return SP()

    def execute(self, stmt, params=None):
        sql = " ".join(str(stmt).split())
        self.eng.calls.append((sql, dict(params or {})))
        out = self.eng.handler(sql, dict(params or {}))
        if isinstance(out, Exception):
            raise out
        return _Res(out)


class FakeEngine:
    def __init__(self, handler):
        self.calls, self.events, self.handler = [], [], handler

    def begin(self):
        return _Conn(self)

    connect = begin


def wo_handler(dup_candidate=False):
    def h(sql, p):
        if sql.startswith("INSERT INTO vera_vjay.questions"):
            return [(1,)]
        if sql.startswith("INSERT INTO vera_vjay.runs"):
            return [(2,)]
        if sql.startswith("INSERT INTO vera_vjay.candidates"):
            return IntegrityError("x", {}, Exception("dup")) if dup_candidate else [(10,)]
        if sql.startswith("INSERT INTO vera_vjay.canonical_sources"):
            return [(20,)]
        if sql.startswith("INSERT INTO vera_vjay.sources"):
            return [(30,)]
        if sql.startswith("SELECT run_id FROM vera_vjay.runs"):
            return [(p["r"],)] if p["r"] == 2 else []
        return []
    return h


def ro_handler(canonical=None, maxver=0, same=None, tables=True, snapshot_docs=(), other_url=None):
    def h(sql, p):
        if "WHERE content_hash = :h" in sql:   # global (cross-canonical-URL) lookup, uses sources_content_hash_idx
            return [other_url] if other_url else []
        if "to_regclass" in sql:
            return [("x",)] if tables else [(None,)]
        if "FROM vera_vjay.canonical_sources WHERE canonical_url" in sql:
            return [(canonical,)] if canonical else []
        if "max(version)" in sql:
            return [(maxver,)]
        if "content_hash = :h" in sql:
            return [same] if same else []
        return []
    return h


def _rec(tmp_path):
    m = corpus(tmp_path)
    return fl.prepare(items(m)[0], m.parent, CFG)


def test_store_new_source_one_transaction(tmp_path):
    rec = _rec(tmp_path)
    wo = FakeEngine(wo_handler())
    st = fl.FulltextStore(wo, CFG["question_text"], fl.RoLookup(FakeEngine(ro_handler())), CFG)
    out, ids = st.admit_fulltext(rec)
    assert out == "admitted" and ids == {"source_id": 30, "canonical_source_id": 20, "candidate_id": 10, "version": 1,
                                         "licence_observations": 2}
    sqls = [s for s, _ in wo.calls]
    assert [s.split()[2] for s in sqls] == ["vera_vjay.questions", "vera_vjay.runs", "vera_vjay.candidates",
                                            "vera_vjay.canonical_sources", "vera_vjay.sources",
                                            "vera_vjay.licence_observations", "vera_vjay.licence_observations"]
    assert all("RETURNING" in s for s in sqls[:5]) and not any(s.startswith("UPDATE") for s in sqls)
    assert not any("RETURNING" in s for s in sqls[5:])
    assert wo.events.count("commit") == 1  # all seven statements in one transaction
    cand = wo.calls[2][1]
    assert cand["fs"] == "fetched" and cand["d"] is None and "accepted divergence" in cand["ra"]
    src = wo.calls[4][1]
    assert json.loads(src["p"])["stored_text_kind"] == "full_text" and src["v"] == 1 and src["t"] == rec["text"]


def test_store_existing_canonical_gets_new_version(tmp_path):
    rec = _rec(tmp_path)
    wo = FakeEngine(wo_handler())
    st = fl.FulltextStore(wo, "q", fl.RoLookup(FakeEngine(ro_handler(canonical=77, maxver=1))), CFG, run_id=2)
    out, ids = st.admit_fulltext(rec)
    assert out == "admitted" and ids["canonical_source_id"] == 77 and ids["version"] == 2
    assert not any("canonical_sources" in s for s, _ in wo.calls)


def test_store_same_full_text_is_known_without_writes(tmp_path):
    rec = _rec(tmp_path)
    wo = FakeEngine(wo_handler())
    st = fl.FulltextStore(wo, "q", fl.RoLookup(FakeEngine(ro_handler(canonical=77, same=(31, "full_text")))), CFG)
    assert st.admit_fulltext(rec) == ("known", {"source_id": 31, "canonical_source_id": 77})
    assert wo.calls == []


def test_store_duplicate_candidate_rolls_back(tmp_path):
    rec = _rec(tmp_path)
    wo = FakeEngine(wo_handler(dup_candidate=True))
    st = fl.FulltextStore(wo, "q", fl.RoLookup(FakeEngine(ro_handler())), CFG, run_id=2)
    with pytest.raises(fl.IngestError, match="already exists in run"):
        st.admit_fulltext(rec)
    assert wo.events[-1] == "rollback" and not any("sources" in s.split()[2] for s, _ in wo.calls)


def test_new_run_rows_roll_back_with_a_failed_first_document(tmp_path):
    rec = _rec(tmp_path)
    wo = FakeEngine(wo_handler(dup_candidate=True))
    st = fl.FulltextStore(wo, "q", fl.RoLookup(FakeEngine(ro_handler())), CFG)
    with pytest.raises(fl.IngestError):
        st.admit_fulltext(rec)
    tables = [s.split()[2] for s, _ in wo.calls]
    assert tables[:3] == ["vera_vjay.questions", "vera_vjay.runs", "vera_vjay.candidates"]
    assert wo.events.count("rollback") == 1 and wo.events.count("commit") == 0  # one transaction for all of them
    assert st._run_id is None


def test_dar_persist_failure_rolls_back_the_whole_document(tmp_path):
    rec = _rec(tmp_path)
    wo = FakeEngine(wo_handler())
    st = fl.FulltextStore(wo, "q", fl.RoLookup(FakeEngine(ro_handler())), CFG)

    def failing_step(conn, ids):
        assert ids["source_id"] == 30          # runs after the sources row, on the same connection
        conn.execute(__import__("sqlalchemy").text("INSERT INTO vera_vjay.evidence_spans (x) VALUES (1)"))
        raise RuntimeError("permission denied for table research_domains")
    with pytest.raises(RuntimeError):
        st.admit_fulltext(rec, in_tx=failing_step)
    assert wo.events.count("rollback") == 1 and wo.events.count("commit") == 0 and st._run_id is None
    tables = [s.split()[2] for s, _ in wo.calls]
    assert tables == ["vera_vjay.questions", "vera_vjay.runs", "vera_vjay.candidates",
                      "vera_vjay.canonical_sources", "vera_vjay.sources", "vera_vjay.licence_observations",
                      "vera_vjay.licence_observations", "vera_vjay.evidence_spans"]
    out, ids = fl.FulltextStore(FakeEngine(wo_handler()), "q", fl.RoLookup(FakeEngine(ro_handler())), CFG
                                ).admit_fulltext(rec, in_tx=lambda c, i: "done")
    assert out == "admitted" and ids["in_tx"] == "done"


def test_licence_observation_rows_match_ask_path_semantics(tmp_path):
    rec = _rec(tmp_path)
    rows = fl.licence_observations_for(rec, CFG, 2, 10, now="2026-10-19T00:00:00+00:00")
    assert [r["licence_scope"] for r in rows] == ["metadata", "content"]
    assert [r["governing"] for r in rows] == [False, True]          # full text: content licence governs
    c = rows[1]
    assert c["spdx"] == "CC BY 4.0" and c["decision"] == "allow" and c["reason_code"] == "licence_allow"
    assert "manifest declared: https://creativecommons.org/licenses/by/4.0/" in c["detected_via"]
    assert "document: https://creativecommons.org/licenses/by/4.0/" in c["detected_via"]
    assert c["url"] == rec["provenance"]["url"] and c["doc_ref"] == "10.9999/syn.epi" and c["host"] == "doi.org"
    assert c["url_sha256"] == hashlib.sha256(rec["provenance"]["url"].encode()).hexdigest()
    assert c["run_id"] == "2" and c["candidate_id"] == "10" and c["native_decision"] is None
    assert rows[0]["spdx"] is None and rows[0]["obs_id"] != c["obs_id"]
    assert rec["provenance"]["licence"]["embedded"]["statement"] == "SYNTHETIC licence line."
    from vera.corpus_admission import LICENCE_OBS_INSERT
    import re as _re
    assert set(_re.findall(r":(\w+)", LICENCE_OBS_INSERT)) <= set(c)   # every bound column is supplied


def test_licence_observations_off_writes_none(tmp_path):
    rec = _rec(tmp_path)
    wo = FakeEngine(wo_handler())
    st = fl.FulltextStore(wo, "q", fl.RoLookup(FakeEngine(ro_handler())), dict(CFG, licence_observations_enabled=False))
    out, ids = st.admit_fulltext(rec)
    assert "licence_observations" not in ids and not any("licence_observations" in s for s, _ in wo.calls)


def test_cli_refuses_when_observation_table_absent(tmp_path, capsys):
    m = corpus(tmp_path)
    env = {CFG["db_url_env_wo"]: SENTINEL, CFG["db_url_env_ro"]: SENTINEL}

    def h(s, p):
        if "to_regclass" in s:
            return [(None,)] if p["t"].endswith("licence_observations") else [("x",)]
        if "has_table_privilege" in s:
            return [(True,)]
        return []
    fac = lambda url, **k: FakeEngine(h)  # noqa: E731
    assert cli.main(["ingest", str(m), "--write", "--new-run"], env=env, engine_factory=fac) == 4
    assert "lacks INSERT on licence_observations" in capsys.readouterr().out


def test_conn_writer_uses_the_given_transaction():
    eng = FakeEngine(lambda s, p: [(5,)])
    with eng.begin() as conn:
        w = fl.ConnWriter(conn)
        assert w.write_span(1, "t", 0, 1) == 5 and w.write_corpus_assoc(1, 2, "x") is True
    assert eng.events == ["sp_commit", "sp_commit", "commit"]   # savepoints inside ONE transaction


def test_cli_refuses_when_insert_grants_missing(tmp_path, capsys):
    m = corpus(tmp_path)
    env = {CFG["db_url_env_wo"]: SENTINEL, CFG["db_url_env_ro"]: SENTINEL}

    def h(s, p):
        if "to_regclass" in s:
            return [("x",)]
        if "has_table_privilege" in s:
            return [(not p["t"].endswith("research_domains"),)]
        return []
    fac = lambda url, **k: FakeEngine(h)  # noqa: E731
    assert cli.main(["ingest", str(m), "--write", "--new-run"], env=env, engine_factory=fac) == 4
    assert "lacks INSERT on research_domains" in capsys.readouterr().out


def test_convert_batch_survives_hostile_file(tmp_path, capsys):
    deep = b"<?xml version='1.0'?><article><body>" + b"<sec>" * 5000 + b"</sec>" * 5000 + b"</body></article>"
    m = fs.write_corpus(tmp_path, raw_over={"syn_epi": deep})
    assert cli.main(["convert", str(m), "--raw-dir", str(tmp_path / "raw")]) == 2
    out = capsys.readouterr().out
    assert "REFUSED syn_epi R-CONVERT" in out and "OK syn_clim" in out and "OK syn_repro" in out


def test_check_persist_flags_unresolved():
    with pytest.raises(fl.IngestError):
        fl.check_persist({"status": "persisted", "unresolved": 1}, "x")
    with pytest.raises(fl.IngestError):
        fl.check_persist({"status": "skipped", "reason": "no_db_ids"}, "x")
    assert fl.check_persist({"status": "persisted", "unresolved": 0, "promoted_skipped": 0}, "x")


class _Reader:
    def __init__(self, docs):
        self.docs = docs

    def load_snapshot(self, kinds):
        return {"domains": [{"name": "alpha beta", "name_norm": "alpha beta", "description": "", "origin": "derived"}],
                "documents": self.docs, "corpus_assoc": [], "chunk_assoc": []}

    def domain_id_by_norm(self, n):
        return 501


class _Lookup:
    def __init__(self, counts, spans):
        self.c, self.s = counts, spans

    def span_counts(self, ids):
        return self.c

    def spans(self, ids):
        return [x for x in self.s if x[1] in set(ids)]


def test_prime_and_resume(tmp_path):
    m = corpus(tmp_path)
    recs = [fl.prepare(it, m.parent, CFG) for it in items(m)[:2]]
    docs = [dict(fl.doc_for_dar(r, 100 + i, 200 + i, CFG)) for i, r in enumerate(recs)]
    for d in docs:
        d.pop("admission_passed")
    from vera.dar_corpus import DarCorpus
    dar = DarCorpus(fl.ingest_routing_config(cfg=CFG))
    lookup = _Lookup({100: 3}, [(9001, 100, 0, 10)])
    ctx, pending = fl.prime_from_store(dar, lookup, _Reader(docs), ["abstract"])
    assert [d["db_source_id"] for d in pending] == [101]          # no spans -> resumed, not restored
    assert dar.stats()["documents"] == 1
    assert ctx["canonical"] == {docs[0]["source_key"]: 200, docs[1]["source_key"]: 201}
    assert ctx["spans"] == {f"{docs[0]['source_key']}:0-10": 9001} and set(ctx["domains"].values()) == {501}
    calls = []

    def fake_persist(plan, writer, ctx=None):
        calls.append((plan["doc"]["db_source_id"], ctx is not None, isinstance(writer, fl.ConnWriter)))
        return {"status": "persisted", "unresolved": 0, "promoted_skipped": 0}
    eng = FakeEngine(lambda s, p: [])
    fl.resume_pending(dar, eng, None, pending, ctx, fake_persist)
    assert calls == [(101, True, True)] and dar.stats()["documents"] == 2 and eng.events == ["commit"]


# ------------------------------------------------------------------------------------ CLI
SENTINEL = "postgresql://SENTINEL-USER:SENTINEL-PW@sentinel-host.invalid/db"


def test_cli_dry_run_ok_and_refusals(tmp_path, capsys):
    m = corpus(tmp_path, items_over={"syn_repro": {"verifier_status": "PENDING"}})
    rep = tmp_path / "rep.json"
    assert cli.main(["ingest", str(m), "--report", str(rep)], env={}) == 2
    out = capsys.readouterr().out
    assert out.count("OK syn_") >= 2 and "REFUSED syn_repro R-VERIFIER" in out
    r = json.loads(rep.read_text())
    assert r["mode"] == "dry-run" and [i["outcome"] for i in r["items"]] == ["ok", "ok", "refused"]
    assert "SYNTHETIC sentence" not in rep.read_text()          # report carries no text


def test_cli_write_guards_never_print_env(tmp_path, capsys):
    m = corpus(tmp_path)
    env = {CFG["db_url_env_wo"]: SENTINEL, CFG["db_url_env_ro"]: SENTINEL}
    assert cli.main(["ingest", str(m), "--write"], env=env) == 64
    assert cli.main(["ingest", str(m), "--write", "--new-run"], env={}) == 4
    fac = lambda url, **k: FakeEngine(lambda s, p: [(None,)] if "to_regclass" in s else [])  # noqa: E731
    assert cli.main(["ingest", str(m), "--write", "--new-run"], env=env, engine_factory=fac) == 4
    assert cli.main(["ingest", str(m), "--write", "--run-id", "9"], env=env, engine_factory=fac) == 4
    out = capsys.readouterr().out
    assert "SENTINEL" not in out and "sentinel-host" not in out
    assert "migration 007 not applied" in out and "run 9 not found" in out


def test_cli_write_driver_error_text_withheld(tmp_path, capsys):
    m = corpus(tmp_path)
    env = {CFG["db_url_env_wo"]: SENTINEL, CFG["db_url_env_ro"]: SENTINEL}

    def boom(s, p):
        return RuntimeError("could not connect to sentinel-host.invalid")
    fac = lambda url, **k: FakeEngine(boom)  # noqa: E731
    assert cli.main(["ingest", str(m), "--write", "--new-run", "--sources-only"], env=env, engine_factory=fac) == 3
    assert "sentinel-host" not in capsys.readouterr().out


# ------------------------------------------------------------- D-057: duplicate full text under another canonical URL
def test_same_text_under_another_canonical_url_is_known_not_ingested_again(tmp_path):
    rec = _rec(tmp_path)
    wo = FakeEngine(wo_handler())
    # canonical URL is new (csid None) but the content hash already exists under canonical source 88 (source 41)
    lk = fl.RoLookup(FakeEngine(ro_handler(canonical=None, other_url=(41, 88, "full_text"))))
    out, ids = fl.FulltextStore(wo, "q", lk, CFG).admit_fulltext(rec)
    assert out == "known" and wo.calls == []                      # no run/candidate/canonical/source write at all
    assert ids["source_id"] == 41 and ids["canonical_source_id"] == 88
    assert ids["decision"] == "known_same_content_other_canonical_url" and ids["duplicate_of_canonical_source_id"] == 88


def test_same_text_other_url_with_wrong_stored_kind_is_refused(tmp_path):
    rec = _rec(tmp_path)
    lk = fl.RoLookup(FakeEngine(ro_handler(other_url=(41, 88, "abstract"))))
    with pytest.raises(fl.IngestError, match="already stored as kind"):
        fl.FulltextStore(FakeEngine(wo_handler()), "q", lk, CFG).admit_fulltext(rec)


def test_same_url_new_content_is_still_a_new_version(tmp_path):
    rec = _rec(tmp_path)
    wo = FakeEngine(wo_handler())
    lk = fl.RoLookup(FakeEngine(ro_handler(canonical=77, maxver=1, same=None, other_url=None)))
    out, ids = fl.FulltextStore(wo, "q", lk, CFG, run_id=2).admit_fulltext(rec)
    assert out == "admitted" and ids["version"] == 2 and ids["canonical_source_id"] == 77


def test_exact_rerun_is_a_noop(tmp_path):
    rec = _rec(tmp_path)
    wo = FakeEngine(wo_handler())
    lk = fl.RoLookup(FakeEngine(ro_handler(canonical=77, same=(31, "full_text"), other_url=(31, 77, "full_text"))))
    assert fl.FulltextStore(wo, "q", lk, CFG).admit_fulltext(rec) == ("known", {"source_id": 31, "canonical_source_id": 77})
    assert wo.calls == []


def test_content_hash_lookup_is_indexed_in_existing_migrations():
    sql = (Path(__file__).resolve().parents[1] / "db" / "migrations" / "002_vera_indices.sql").read_text()
    assert "CREATE INDEX sources_content_hash_idx ON vera_vjay.sources (content_hash)" in sql


def test_cli_report_records_duplicate_decision(tmp_path, capsys):
    m = corpus(tmp_path)
    env = {CFG["db_url_env_wo"]: SENTINEL + "w", CFG["db_url_env_ro"]: SENTINEL + "r"}
    wo = FakeEngine(lambda s, p: [(True,)] if "has_table_privilege" in s else [("x",)] if "to_regclass" in s else [])
    ro = FakeEngine(ro_handler(canonical=None, other_url=(41, 88, "full_text")))
    fac = lambda url, **k: wo if url.endswith("w") else ro  # noqa: E731
    rep = tmp_path / "rep.json"
    rc = cli.main(["ingest", str(m), "--write", "--new-run", "--sources-only", "--report", str(rep)], env=env,
                  engine_factory=fac)
    assert rc == 0
    items_ = json.loads(rep.read_text())["items"]
    assert items_ and all(i["outcome"] == "known" and i["decision"] == "known_same_content_other_canonical_url"
                          and i["duplicate_of_canonical_source_id"] == 88 for i in items_)
    assert "KNOWN" in capsys.readouterr().out
    assert not any(s.startswith("INSERT") for s, _ in wo.calls)

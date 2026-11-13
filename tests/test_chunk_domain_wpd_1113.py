"""WP-D-1113: tax_chunk_domain (migration 027 draft) and the job-side domain writer (vera/chunk_domain_writer.py).
Offline only: a recording connection for the writer and the job's FakeDB for the integration cases; the migration is checked
statically against the 020 tax_chunk_topic definition it mirrors. Nothing here connects to a database."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from tests.test_chunk_taxo_job_1109 import FakeDB, make, spans
from vera import chunk_domain_writer as dw
from vera import chunk_taxo_store as store

ROOT = Path(__file__).resolve().parent.parent
MIG = ROOT / "db" / "migrations" / "pending" / "027-DRAFT-chunk-domain.sql.draft"
RBK = ROOT / "db" / "migrations" / "rollback" / "027-DRAFT-chunk-domain-rollback.sql.draft"
M020 = ROOT / "db" / "migrations" / "pending" / "020-DRAFT-taxonomy.sql.draft"
BY_TEXT = {t: n for n, t in store.SQL.items()}


class Res:
    def __init__(self, rows=(), rowcount=0):
        self.rows, self.rowcount = list(rows), rowcount

    def first(self):
        return self.rows[0] if self.rows else None


class RecConn:
    """Executes allowlisted statements by recording (name, params); the probe answers `table`."""

    def __init__(self, table, retire_count=0):
        self.table, self.log, self.retire_count = table, [], retire_count

    def execute(self, stmt, params=None):
        name = BY_TEXT[str(stmt)]  # KeyError: a statement outside the allowlist
        self.log.append((name, params))
        if name == "probe_chunk_domain":
            return Res([(self.table,)])
        if name == "link_retire_chunk_domain":
            return Res(rowcount=self.retire_count)
        return Res()

    def names(self):
        return [n for n, _ in self.log]


def item(level="subtopic", sub="S0", top="T", dom="D", multi=None):
    d = dict(subtopic=sub, topic=top, domain=dom, level=level, distance=0.1, margin=0.5, reason=None)
    if multi:
        d["multi"] = multi
    return d


ABSTAINED = dict(subtopic=None, topic=None, domain=None, level=None, distance=0.9, margin=None, reason="NO_FIT")
IDS = {"domain": {"D": 71, "D2": 72}, "topic": {"T": 51, "T2": 52}, "subtopic": {"S0": 31}}


# ---------------------------------------------------------------------------------------------------------------- writer
def test_table_absent_is_a_noop_with_metric():
    c, st = RecConn(False), dw.new_state()
    dw.write_page(c, st, [1, 2], {1: item(), 2: item()}, IDS, "v1")
    dw.write_page(c, st, [3], {3: item()}, IDS, "v1")
    assert c.names() == ["probe_chunk_domain"]  # probed once per pass, nothing retired or inserted
    assert dw.metrics(st) == {"domain_table_absent": 1, "domain_rows_skipped_no_table": 3}  # T51: the would-be count


def test_table_present_writes_strict_domain_rows():
    c, st = RecConn(True), dw.new_state()
    a = {1: item("subtopic"), 2: item("topic", sub=None), 3: item("domain", sub=None, top=None)}
    dw.write_page(c, st, [1, 2, 3], a, IDS, "v1")
    assert c.names() == ["probe_chunk_domain", "link_retire_chunk_domain", "link_insert_chunk_domain"]
    assert c.log[1][1] == {"version": "v1", "ids": [1, 2, 3]}
    rows = c.log[2][1]
    assert rows == [{"a": s, "b": 71, "version": "v1", "score": None, "score_kind": None} for s in (1, 2, 3)]
    assert dw.metrics(st) == {"domain_table_absent": 0, "domain_rows_attempted": 3, "domain_chunks_without_domain": 0, "domain_rows_retired": 0}


def test_chunk_with_no_topic_link_gets_no_domain_row():
    c, st = RecConn(True), dw.new_state()
    dw.write_page(c, st, [1, 2], {1: ABSTAINED, 2: item()}, IDS, "v1")
    assert [r["a"] for r in c.log[-1][1]] == [2]
    c2, st2 = RecConn(True), dw.new_state()
    dw.write_page(c2, st2, [5], {5: ABSTAINED}, IDS, "v1")  # whole page abstained: the retire still runs, the insert has no rows
    assert c2.names() == ["probe_chunk_domain", "link_retire_chunk_domain"]  # an empty row list is never executed (run_sql)
    dw.write_page(c2, st2, [6, 7], {}, IDS, "v1")  # span without an association item at all
    assert c2.names() == ["probe_chunk_domain", "link_retire_chunk_domain", "link_retire_chunk_domain"]


def test_multi_topic_chunk_gets_one_domain_row_from_its_strict_node():
    # strict node in domain D; routing-only links to topics of another domain D2 never create a domain row
    multi = [("topic", "T2", 0.41, "cosine_r1"), ("topic", "T", 0.40, "cosine_r2")]
    c, st = RecConn(True), dw.new_state()
    dw.write_page(c, st, [1], {1: item("topic", sub=None, multi=multi)}, IDS, "v1")
    assert c.log[-1][1] == [{"a": 1, "b": 71, "version": "v1", "score": None, "score_kind": None}]
    assert dw.domain_key(item(multi=multi)) == "D"
    assert dw.domain_key({"level": None, "domain": "D"}) is None  # routing-only: no strict level
    assert dw.domain_key(None) is None and dw.domain_key("x") is None


def test_unknown_domain_key_is_counted_not_written():
    c, st = RecConn(True), dw.new_state()
    dw.write_page(c, st, [1, 2], {1: item(dom="GONE"), 2: item(dom="D2")}, IDS, "v1")
    assert [r["b"] for r in c.log[-1][1]] == [72]
    assert dw.metrics(st)["domain_chunks_without_domain"] == 1


def test_retire_runs_before_insert_and_counts_rows():
    c, st = RecConn(True, retire_count=4), dw.new_state()
    dw.write_page(c, st, [1], {1: item()}, IDS, "v2")
    assert c.names().index("link_retire_chunk_domain") < c.names().index("link_insert_chunk_domain")
    assert dw.metrics(st)["domain_rows_retired"] == 4


def test_optional_cosine_scores_are_validated():
    c, st = RecConn(True), dw.new_state()
    dw.write_page(c, st, [1, 2, 3], {s: item() for s in (1, 2, 3)}, IDS, "v1",
                  scores={1: (0.4321567, "cosine_lvl"), 2: (float("nan"), "cosine_lvl"), 3: (0.5, None)})
    rows = {r["a"]: r for r in c.log[-1][1]}
    assert (rows[1]["score"], rows[1]["score_kind"]) == (0.432157, "cosine_lvl")
    assert (rows[2]["score"], rows[2]["score_kind"]) == (None, None) and (rows[3]["score"], rows[3]["score_kind"]) == (None, None)


def test_metrics_empty_when_never_probed():
    assert dw.metrics(dw.new_state()) == {}


def test_store_statements_are_allowlisted_and_safe():
    for nm in ("probe_chunk_domain", "link_insert_chunk_domain", "link_retire_chunk_domain"):
        up = store.SQL[nm].upper()
        for bad in ("DELETE", "TRUNCATE", "DROP ", "ALTER ", "CREATE ", "GRANT", "REVOKE"):
            assert bad not in up, (nm, bad)
    probe_sql = store.SQL["probe_chunk_domain"]  # T53: column functions only (grants are column-level), never has_table_privilege
    assert "has_table_privilege" not in probe_sql and probe_sql.count("has_column_privilege") == 8 and "'UPDATE'" in probe_sql
    assert "to_regclass('vera_vjay.tax_chunk_domain')" in probe_sql and "domain_chunks" not in " ".join(store.SQL.values())  # T55
    assert "ON CONFLICT DO NOTHING" in store.SQL["link_insert_chunk_domain"]
    assert "vera_vjay.tax_chunk_domain (span_id, domain_id, assigned_via, classifier_name, classifier_version, score, score_kind)" \
        in store.SQL["link_insert_chunk_domain"]
    r = store.SQL["link_retire_chunk_domain"]
    assert "SET state = 'retired'" in r and "classifier_name = 'chunk_taxo_auto'" in r and "classifier_version <> :version" in r \
        and "span_id = ANY(:ids)" in r
    assert "chunk_domain" not in store.CHUNK_LINKS  # never read as a prior, never in the topic link loop
    with pytest.raises(store.SqlRefused):
        store.run_sql(RecConn(True), "link_select_chunk_domain")


# ------------------------------------------------------------------------------------------------------ job integration
def run_job(db, **kw):
    j = make(db, **kw)
    r = j.run()
    return j, r


def run_metrics(db):
    return json.loads(db.runs[-1]["metrics"])["run"]


def test_job_table_absent_writes_no_domain_and_records_metric():
    db = FakeDB(spans())
    j, r = run_job(db)
    assert r["complete"] and "link_insert_chunk_domain" not in db.names() and "link_retire_chunk_domain" not in db.names()
    assert db.names().count("probe_chunk_domain") == 1
    m = run_metrics(db)
    assert m["domain_table_absent"] == 1 and "domain_rows_attempted" not in m and m["domain_rows_skipped_no_table"] == 10
    assert len(db.active("chunk_subtopic")) + len(db.active("chunk_topic")) == 10  # topic and sub-topic links unchanged


def test_job_table_present_writes_one_domain_row_per_stored_chunk_in_page_order():
    db = FakeDB(spans())
    db.domain_table = True
    j, r = run_job(db)
    assert r["complete"]
    dom = db.active("chunk_domain")
    assert [a for a, _ in dom] == list(range(1, 11)) and len({b for _, b in dom}) == 1
    n = db.names()
    # same page, same transaction path: sub-topic and topic inserts, then the domain insert, then the next page or the votes
    assert n.index("link_insert_chunk_subtopic") < n.index("link_insert_chunk_domain") < n.index("link_insert_source_topic")
    m = run_metrics(db)
    assert m["domain_table_absent"] == 0 and m["domain_rows_attempted"] == 10 and m["domain_chunks_without_domain"] == 0
    for k, v in db.links.items():
        if k[0] == "chunk_domain":
            assert v["score"] is None and v["score_kind"] is None


def test_job_rerun_same_version_adds_no_duplicate_active_rows():
    db = FakeDB(spans())
    db.domain_table = True
    run_job(db)
    first = db.active("chunk_domain")
    run_job(db)
    assert db.active("chunk_domain") == first


def test_job_new_version_retires_old_domain_rows_and_rewrites():
    db = FakeDB(spans())
    db.domain_table = True
    run_job(db)
    v1 = {k: dict(v) for k, v in db.links.items() if k[0] == "chunk_domain"}
    db.spans = spans(12)  # a new span set: the StubCore version hashes it, so the next pass is a new version
    run_job(db)
    states = [v["state"] for k, v in db.links.items() if k[0] == "chunk_domain"]
    assert len(v1) == 10 and states.count("retired") == 10 and states.count("active") == 12
    assert [a for a, _ in db.active("chunk_domain")] == list(range(1, 13))


def test_job_abstained_chunks_get_no_domain_row():
    class AbstainOdd(type(make(FakeDB(spans())).core)):
        def build(self, chunks, vecs, params, prior=None):
            m = super().build(chunks, vecs, params, prior)
            for sid in list(m.assoc):
                if sid % 2:
                    m.assoc[sid] = dict(subtopic=None, topic=None, domain=None, level=None, distance=0.9, margin=None, reason="NO_FIT")
            return m
    db = FakeDB(spans())
    db.domain_table = True
    j = make(db)
    j.core = AbstainOdd()
    assert j.run()["complete"]
    assert [a for a, _ in db.active("chunk_domain")] == [2, 4, 6, 8, 10]
    assert len(db.active("chunk_subtopic")) + len(db.active("chunk_topic")) == 5  # same chunks as the topic links


@pytest.mark.parametrize("has,curator,expect_write", [
    ((True, True, True), False, True),     # 022 present: write mode
    ((True, False, True), False, False),   # 020 only: compute only, no taxonomy write at all
    ((True, False, True), True, True),     # 020 only with the curator switch: writes curator-origin links, so the domain too
])
def test_job_mode_combinations(has, curator, expect_write):
    db = FakeDB(spans(), has=has)
    db.domain_table = True
    kw = {"allow_curator_origin_on_020": True} if curator else {}
    j, r = run_job(db, **kw)
    assert r["complete"]
    assert bool(db.active("chunk_domain")) is expect_write
    if not expect_write:
        assert "probe_chunk_domain" not in db.names()


def test_job_inert_never_probes():
    db = FakeDB(spans(), has=(True, True, False))
    j, r = run_job(db)
    assert r["state"] == "inert" and "probe_chunk_domain" not in db.names()


# ------------------------------------------------------------------------------------------------- migration static checks
def _block(text, name):
    m = re.search(r"CREATE TABLE vera_vjay\.%s \((.*?)\n\);" % name, text, re.S)
    assert m, name
    return m.group(1)


def _norm(s):
    return re.sub(r"\s+", " ", s).strip()


def test_027_table_mirrors_020_tax_chunk_topic():
    m020, m027 = M020.read_text(), MIG.read_text()
    want = _block(m020, "tax_chunk_topic").replace("tax_chunk_topic_topic_fk", "tax_chunk_domain_domain_fk").replace("chunk_topic", "chunk_domain").replace("topic_id", "domain_id").replace("tax_topics", "tax_domains")
    assert _norm(_block(m027, "tax_chunk_domain")) == _norm(want)
    assert "CREATE UNIQUE INDEX tax_chunk_domain_active_uq ON vera_vjay.tax_chunk_domain (span_id, domain_id) WHERE state = 'active';" in m027
    assert "CREATE INDEX tax_chunk_domain_domain_idx ON vera_vjay.tax_chunk_domain (domain_id) WHERE state = 'active';" in m027
    assert "CREATE TRIGGER tax_chunk_domain_guard_trg BEFORE INSERT OR UPDATE ON vera_vjay.tax_chunk_domain FOR EACH ROW " \
           "EXECUTE FUNCTION vera_vjay.tax_link_guard_trgfn();" in m027
    # same grant pattern as 020's tax_chunk_topic
    assert "GRANT INSERT (span_id, domain_id, assigned_via, classifier_name, classifier_version, score, score_kind) ON vera_vjay.tax_chunk_domain TO vera_vjay_rw;" in m027
    assert "GRANT UPDATE (state) ON vera_vjay.tax_chunk_domain TO vera_vjay_rw;" in m027
    assert "GRANT SELECT ON vera_vjay.tax_chunk_domain TO vera_vjay_rw, vera_vjay_ro;" in m027
    assert len(re.findall(r"CONSTRAINT tax_chunk_domain_\w+", m027)) == 13


def test_027_is_additive_guarded_and_never_widens_grants():
    t = MIG.read_text()
    code = "\n".join(ln for ln in t.splitlines() if not ln.lstrip().startswith("--"))
    for bad in ("CASCADE", "DROP ", "ALTER ", "TRUNCATE TABLE", "TRUNCATE vera", "DELETE FROM", "CREATE ROLE", "ALTER ROLE", "GRANT DELETE", "TO PUBLIC", "TO vera_vjay_wo"):
        assert bad not in code, bad
    assert not re.search(r"^\s*GRANT\b[^;]*\b(DELETE|TRUNCATE|REFERENCES|TRIGGER|ALL)\b", code, re.M | re.I)
    assert code.count("CREATE TABLE") == 1 and "CREATE FUNCTION" not in code and "CREATE VIEW" not in code
    assert "already applied or name collision" in code and "vera_vjay.tax_link_guard_trgfn" in code
    assert code.strip().endswith("COMMIT;") and "BEGIN;" in code and "ON_ERROR_STOP" in t
    assert "\\if :{?app_rw_login}" in t
    assert not re.search(r"://\S*@|\.com\b", t, re.I)  # no URL with credentials, no host name


def test_027_rollback_is_guarded_and_drops_only_the_table():
    t = RBK.read_text()
    code = "\n".join(ln for ln in t.splitlines() if not ln.lstrip().startswith("--"))
    assert "CASCADE" not in code and code.count("DROP ") == 1 and "DROP TABLE vera_vjay.tax_chunk_domain;" in code
    assert "purge_rows" in code and "rollback refused" in code and "must run as the owner" in code
    assert code.index("rollback refused") < code.index("DROP TABLE")
    assert code.strip().endswith("COMMIT;")

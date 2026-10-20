"""VERA-DAR 007 domain registry migration: static text checks only. No DB, no network, no SQL execution."""
import fnmatch
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MIG_DIR = ROOT / "db" / "migrations"
# N-1: draft names deliberately do not match apply_migrations.py's "{prefix}_vera_*.sql" glob, for any argument.
MIG = MIG_DIR / "pending" / "007-DRAFT-domain-registry.sql.draft"
ROLLBACK = MIG_DIR / "rollback" / "007-DRAFT-domain-registry-rollback.sql.draft"
BANNED_TABLE_NAMES = ("research_domains", "domain_corpus", "domain_chunks")


def _matches_runner_glob(name):
    return fnmatch.fnmatch(name, "*_vera_*.sql") or "_vera_" in name


def test_drafts_exist_and_match_no_runner_glob_N1():
    assert MIG.is_file() and ROLLBACK.is_file()
    for p in (MIG, ROLLBACK):
        assert not _matches_runner_glob(p.name), p.name
        assert not p.name.endswith(".sql"), p.name
    # Nothing under db/migrations/ that matches the runner glob may mention the DAR tables.
    for p in MIG_DIR.rglob("*_vera_*.sql"):
        text = p.read_text(encoding="utf-8")
        for t in BANNED_TABLE_NAMES:
            assert t not in text, (p, t)


def _sql(path):
    return path.read_text(encoding="utf-8")


def _no_comments(text):
    # Strip SQL line comments so that prose such as "No UPDATE/DELETE" does not trip the keyword checks.
    return "\n".join(line.split("--", 1)[0] for line in text.splitlines())


def _norm(text):
    return re.sub(r"\s+", " ", _no_comments(text)).strip()


def test_files_are_in_pending_and_rollback_not_runner_visible():
    assert MIG.is_file() and ROLLBACK.is_file()
    assert MIG.parent.name == "pending" and MIG.parent.parent == MIG_DIR
    assert ROLLBACK.parent.name == "rollback" and ROLLBACK.parent.parent == MIG_DIR
    # The runner globs migrations/<prefix>_vera_*.sql only (db/apply_migrations.py:28); nothing 007 may sit there.
    assert list(MIG_DIR.glob("007_vera_*.sql")) == []


def test_forward_header_says_not_executed_and_forbids_runner():
    head = "\n".join(_sql(MIG).splitlines()[:8])
    assert "NOT EXECUTED" in head
    assert "Approval 1" in head
    assert "isolated DB" in head
    assert "apply_migrations.py" in head and "--dry-run" in head


def test_composite_pks_present():
    text = _norm(_sql(MIG))
    assert "PRIMARY KEY (domain_id, canonical_source_id)" in text
    assert "PRIMARY KEY (domain_id, span_id)" in text


def test_reverse_indexes_present():
    text = _norm(_sql(MIG))
    assert re.search(r"CREATE INDEX IF NOT EXISTS domain_corpus_canonical_source_idx ON vera_vjay\.domain_corpus \(canonical_source_id, domain_id\)", text)
    assert re.search(r"CREATE INDEX IF NOT EXISTS domain_chunks_span_idx ON vera_vjay\.domain_chunks \(span_id, domain_id\)", text)


def test_normalized_name_is_unique_column_not_expression_index():
    # RC-5: uniqueness lives on a name_norm column written by the Python registry, not on lower(btrim(name)).
    text = _norm(_sql(MIG))
    assert re.search(r"name_norm TEXT NOT NULL CONSTRAINT research_domains_name_norm_key UNIQUE", text)
    assert "lower(btrim(name))" not in text
    assert "CREATE UNIQUE INDEX" not in text


def test_name_norm_nonempty_check_M1():
    # M-1 (RC-5): empty normalized names are rejected by the schema.
    text = _norm(_sql(MIG))
    assert "CHECK (name_norm <> '')" in text


def test_domain_fields_match_registry_M2():
    # M-2: description and origin mirror the registry's Domain fields.
    text = _norm(_sql(MIG))
    assert "description TEXT NOT NULL DEFAULT ''" in text
    assert re.search(r"origin TEXT NOT NULL DEFAULT 'manual' CONSTRAINT research_domains_origin_chk CHECK \(origin IN \('config','manual','fixture','derived'\)\)", text)


def test_asserted_by_on_both_junctions_M3():
    # M-3: persisted associations keep their asserter.
    text = _norm(_sql(MIG))
    assert re.search(r"CREATE TABLE IF NOT EXISTS vera_vjay\.domain_corpus \(.*?asserted_by TEXT NOT NULL", text)
    assert re.search(r"CREATE TABLE IF NOT EXISTS vera_vjay\.domain_chunks \(.*?asserted_by TEXT NOT NULL", text)


def test_grant_roles_whitelisted_and_privileges_insert_select_only():
    # Only the three VERA-DAR roles may be granted; privileges are limited to INSERT and SELECT (no UPDATE, DELETE, ALL).
    allowed_roles = {"vera_vjay_wo", "vera_vjay_rw", "vera_vjay_ro"}
    text = _no_comments(_sql(MIG))
    grants = re.findall(r"GRANT ([^;]+?) TO (\w+)", text)
    assert grants
    for privs, role in grants:
        assert role in allowed_roles, role
        verbs = set(re.findall(r"[A-Z]+", re.sub(r"\([^)]*\)", "", privs.split(" ON ")[0]).upper()))  # column lists (R3) are not verbs
        assert verbs <= {"INSERT", "SELECT"}, verbs
        assert "ALL" not in verbs


def test_fk_targets_and_restrict():
    text = _norm(_sql(MIG))
    assert re.search(r"canonical_source_id BIGINT NOT NULL CONSTRAINT \w+ REFERENCES vera_vjay\.canonical_sources \(canonical_source_id\) ON DELETE RESTRICT", text)
    assert re.search(r"span_id BIGINT NOT NULL CONSTRAINT \w+ REFERENCES vera_vjay\.evidence_spans \(span_id\) ON DELETE RESTRICT", text)
    assert len(re.findall(r"REFERENCES vera_vjay\.research_domains \(domain_id\) ON DELETE RESTRICT", text)) == 2
    assert "ON DELETE CASCADE" not in text


def test_identity_bigint_keys():
    text = _norm(_sql(MIG))
    assert "domain_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY" in text


def test_no_destructive_statements_on_existing_tables_in_forward_file():
    # "ON DELETE RESTRICT" is an FK clause, not a DELETE statement; strip it before the keyword scan.
    text = _no_comments(_sql(MIG)).upper().replace("ON DELETE RESTRICT", "")
    for kw in ("DROP", "ALTER", "UPDATE", "DELETE", "TRUNCATE"):
        assert not re.search(r"\b" + kw + r"\b", text), kw
    assert "CREATE TRIGGER" not in text  # RC-6: no trigger


def test_only_new_tables_are_created():
    creates = re.findall(r"CREATE TABLE IF NOT EXISTS (vera_vjay\.\w+)", _no_comments(_sql(MIG)))
    assert creates == ["vera_vjay.research_domains", "vera_vjay.domain_corpus", "vera_vjay.domain_chunks"]


def test_memory_placeholder_exclusion_documented():
    # RC-6: header note on placeholder anchors with the exclusion predicate, documented not enforced.
    raw = _sql(MIG)
    assert "memory_store.py:244-248" in raw
    assert "start_index = 0" in raw
    assert "NOT (es.start_index = 0 AND es.text = :span_text)" in raw


def test_grants_are_insert_or_select_only():
    text = _no_comments(_sql(MIG))
    grants = re.findall(r"GRANT ([^;]+?) TO (\w+)", text)
    assert grants, "expected grant statements"
    for privs, _role in grants:
        assert "UPDATE" not in privs and "DELETE" not in privs and "TRUNCATE" not in privs


def test_rollback_reverses_forward_file():
    fwd_tables = re.findall(r"CREATE TABLE IF NOT EXISTS (vera_vjay\.\w+)", _no_comments(_sql(MIG)))
    rb = _no_comments(_sql(ROLLBACK))
    drops = re.findall(r"DROP TABLE IF EXISTS (vera_vjay\.\w+)", rb)
    # Every table created forward is dropped back, and nothing else is dropped.
    assert sorted(drops) == sorted(fwd_tables)
    # Junction tables before the parent (FKs are ON DELETE RESTRICT).
    assert drops.index("vera_vjay.research_domains") == len(drops) - 1
    assert drops.index("vera_vjay.domain_chunks") < drops.index("vera_vjay.research_domains")
    assert drops.index("vera_vjay.domain_corpus") < drops.index("vera_vjay.research_domains")
    # Rollback is destructive and marked as such.
    assert "DESTRUCTIVE" in _sql(ROLLBACK)[:300]
    # No ALTER/UPDATE of anything in the rollback.
    up = rb.upper()
    assert not re.search(r"\b(ALTER|UPDATE|DELETE|TRUNCATE)\b", up)


def test_rollback_drops_no_functions_or_triggers_because_none_are_created():
    assert "FUNCTION" not in _no_comments(_sql(MIG)).upper()
    assert "FUNCTION" not in _no_comments(_sql(ROLLBACK)).upper()


def test_fail_closed_preflight_before_any_ddl():
    raw = _no_comments(_sql(MIG))
    n = _norm(_sql(MIG))
    assert "RAISE EXCEPTION" in raw
    i_raise = raw.index("RAISE EXCEPTION")
    # the raise sits before any DDL and names both required roles and the order
    assert i_raise < raw.index("CREATE SCHEMA") < raw.index("CREATE TABLE")
    assert re.search(r"NOT EXISTS \(SELECT 1 FROM pg_roles WHERE rolname = 'vera_vjay_wo'\) OR NOT EXISTS \(SELECT 1 FROM pg_roles WHERE rolname = 'vera_vjay_ro'\)", n)
    msg = re.search(r"RAISE EXCEPTION '([^']*)'", n).group(1)
    assert "vera_vjay_wo" in msg and "vera_vjay_ro" in msg
    assert msg.index("create_vera_accounts.sql") < msg.index("001-005") < msg.rindex("007")
    # rw stays conditional with a NOTICE
    assert "RAISE NOTICE" in raw
    # wo/ro grants are unconditional (no IF EXISTS silently skipping them)
    assert not re.search(r"IF EXISTS \(SELECT 1 FROM pg_roles WHERE rolname = 'vera_vjay_(wo|ro)'\)", n)


def test_exact_grants_ro_select_wo_insert_plus_key_column():
    n = _norm(_sql(MIG))
    T = r"vera_vjay\.research_domains, vera_vjay\.domain_corpus, vera_vjay\.domain_chunks"
    assert re.search(r"GRANT INSERT ON " + T + r" TO vera_vjay_wo;", n)
    assert "GRANT SELECT (domain_id) ON vera_vjay.research_domains TO vera_vjay_wo;" in n
    assert re.search(r"GRANT SELECT ON " + T + r" TO vera_vjay_ro;", n)
    assert re.search(r"GRANT SELECT, INSERT ON " + T + r" TO vera_vjay_rw;", n)
    assert len(re.findall(r"GRANT ", n)) == 4


def test_header_documents_required_order():
    head = "\n".join(_sql(MIG).splitlines()[:14])
    assert "REQUIRED ORDER" in head and "create_vera_accounts.sql" in head and "001-005" in head
    assert "FAIL CLOSED" in head


def test_on_error_stop_and_fk_table_preflight():
    raw = _no_comments(_sql(MIG))
    assert raw.lstrip().startswith("\\set ON_ERROR_STOP on")
    i = raw.index("to_regclass('vera_vjay.canonical_sources') IS NULL OR to_regclass('vera_vjay.evidence_spans') IS NULL")
    assert i < raw.index("CREATE SCHEMA")
    assert raw.index("RAISE EXCEPTION", i) < raw.index("CREATE SCHEMA")
    assert "vera_vjay.canonical_sources" in raw[i:raw.index("CREATE SCHEMA")]

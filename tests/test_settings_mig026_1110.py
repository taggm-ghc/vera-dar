"""SETTINGS-W2-1110: static checks of draft migration 026 (settings table). Text only: no DB, no network, no SQL execution."""
import fnmatch
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FWD = ROOT / "db" / "migrations" / "pending" / "026_settings.sql"
RBK = ROOT / "db" / "migrations" / "rollback" / "026_settings_rollback.sql"
KIT = next((a / "work" / "SETTINGS-MIG026-1110" for a in ROOT.parents if (a / "work" / "SETTINGS-MIG026-1110").is_dir()), ROOT / "no-kit")
TABLES = ["set_keys", "set_settings", "set_settings_hist", "set_deny_lists", "set_deny", "set_deny_hist"]


def _code(p):
    return "\n".join(line.split("--", 1)[0] for line in p.read_text(encoding="utf-8").splitlines())


def test_files_exist_and_are_not_runner_visible():
    for p in (FWD, RBK):
        assert p.is_file()
        assert not fnmatch.fnmatch(p.name, "*_vera_*.sql") and "_vera_" not in p.name
    assert list((ROOT / "db" / "migrations").glob("026_vera_*.sql")) == []


def test_parses_structurally():
    for p in (FWD, RBK):
        t = _code(p)
        assert t.count("$$") % 2 == 0, p
        outside = re.sub(r"\$\$.*?\$\$", "", t, flags=re.S)
        outside = re.sub(r"'(?:[^']|'')*'", "''", outside)
        assert outside.count("(") == outside.count(")"), p
        assert len(re.findall(r"^BEGIN;", t, flags=re.M)) == 1 and len(re.findall(r"^COMMIT;", t, flags=re.M)) == 1
        assert "CASCADE" not in t.upper()


def test_object_counts():
    t = _code(FWD)
    assert re.findall(r"CREATE TABLE vera_vjay\.(\w+)", t) == TABLES
    assert re.findall(r"CREATE VIEW vera_vjay\.(\w+)", t) == ["set_deny_active"]
    assert "security_invoker = true" in t
    assert re.findall(r"CREATE (?:UNIQUE )?INDEX (\w+)", t) == ["set_settings_active_uq", "set_deny_list_exp_idx"]
    assert len(re.findall(r"CREATE FUNCTION", t)) == 6
    assert len(re.findall(r"CREATE TRIGGER", t)) == 18
    head = t[: t.index("CREATE FUNCTION")]
    assert len(re.findall(r"\bCONSTRAINT\s+\w+", head)) == 37


def test_one_active_version_and_append_only_history():
    t = _code(FWD)
    assert "set_settings_active_uq ON vera_vjay.set_settings (key) WHERE state = 'active'" in t
    assert "set_settings_key_fk FOREIGN KEY (key) REFERENCES vera_vjay.set_keys" in t
    for h in ("set_settings_hist", "set_deny_hist"):
        assert re.search(r"BEFORE UPDATE OR DELETE ON vera_vjay\.%s FOR EACH ROW EXECUTE FUNCTION vera_vjay\.set_block_fn" % h, t)
        assert re.search(r"BEFORE TRUNCATE ON vera_vjay\.%s" % h, t)
    assert "one-way cap" in t and "lower_only" in t and "raise_only" in t


def test_deny_tables_are_structurally_deny_only():
    t = _code(FWD)
    for tab in ("set_deny_lists", "set_deny", "set_deny_hist"):
        body = re.search(r"CREATE TABLE vera_vjay\.%s \((.*?)\n\);" % tab, t, flags=re.S).group(1)
        cols = re.findall(r"^\s{4}(\w+)\s+\w", body, flags=re.M)
        assert not [c for c in cols if re.match(r"(allow|permit|grant|effect)", c)], (tab, cols)
    d = re.search(r"CREATE TABLE vera_vjay\.set_deny \((.*?)\n\);", t, flags=re.S).group(1)
    for col in ("owner", "reason", "created_at", "expires_at"):
        assert re.search(r"^\s+%s\s+\S+.*NOT NULL" % col, d, flags=re.M), col
    assert "interval '30 days'" in d and "interval '90 days'" in d
    assert "kind = 'deny'" in t and "deny_[a-z0-9_]+$" in t


def test_grants_are_select_only_for_rw_and_ro():
    t = _code(FWD)
    grants = re.findall(r"GRANT\s+(.*?)\s+ON\s+(.*?)\s+TO\s+([^;]*);", re.sub(r"REVOKE.*?;", "", t, flags=re.S), flags=re.S)
    assert len(grants) == 1
    privs, on, to = grants[0]
    assert privs == "SELECT" and "vera_vjay_rw" in to and "vera_vjay_ro" in to and "wo" not in to
    for tab in TABLES + ["set_deny_active"]:
        assert "vera_vjay.%s" % tab in on
    assert "REVOKE ALL ON vera_vjay.set_keys" in t and "FROM PUBLIC, vera_vjay_wo, vera_vjay_rw, vera_vjay_ro" in t
    assert not re.search(r"GRANT\s+(INSERT|UPDATE|DELETE|ALL|TRUNCATE)", t, flags=re.I)


def test_rollback_refuses_rows_beyond_the_seed_and_has_no_cascade():
    t = _code(RBK)
    assert "rollback refused" in t and "purge_rows" in t and "registered_in_seed <> $1" in t
    assert re.findall(r"DROP TABLE vera_vjay\.(\w+);", t) == ["set_deny_hist", "set_deny", "set_deny_lists", "set_settings_hist", "set_settings", "set_keys"]
    assert len(re.findall(r"DROP FUNCTION", t)) == 6 and "DROP VIEW vera_vjay.set_deny_active" in t


def test_seed_is_generated_and_pinned():
    t = FWD.read_text(encoding="utf-8")
    m = re.search(r"-- BEGIN GENERATED SEED \(gen_seed_026\.py; manifest sha256 ([0-9a-f]{64}); registry file sha256 ([0-9a-f]{64}); set_keys (\d+); set_deny_lists (\d+)", t)
    assert m
    keys = re.findall(r"^  \('([a-z0-9_.-]+)', '[a-z0-9_-]+', '(?:int|float|bool)'", t, flags=re.M)
    assert len(keys) == int(m.group(3)) and len(set(keys)) == len(keys)
    assert len(re.findall(r"^  \('[a-z0-9_.-]+\.deny_[a-z_]+', '\^", t, flags=re.M)) == int(m.group(4))
    assert m.group(1) in RBK.read_text(encoding="utf-8")
    assert "INSERT INTO vera_vjay.set_settings " not in t and "INSERT INTO vera_vjay.set_deny " not in t


def test_kit_matches_repo_files_when_present():
    if not KIT.is_dir():
        return
    assert (KIT / "sql" / FWD.name).read_bytes() == FWD.read_bytes()
    assert (KIT / "sql" / RBK.name).read_bytes() == RBK.read_bytes()
    r = subprocess.run([sys.executable, "-I", str(KIT / "scripts" / "gen_seed_026.py"), "--check"], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    sums = (KIT / "scripts" / "SHA256SUMS").read_text()
    common = (KIT / "scripts" / "tax_common_026.sh").read_text()
    assert "9|37|6|18" in common


def test_policy_prefix_layer_and_history_insert_guard():
    t = _code(FWD)
    assert t.count("_policy_prefix_chk CHECK (key !~ '") == 2
    m = re.search(r"set_keys_policy_prefix_chk CHECK \(key !~ '(.*)'\),", t)
    rx = re.compile(m.group(1))
    for k in ("fair_use.a.b", "licence_allowlist.x.y", "source_withdrawals.x.y", "crawl_delay.a.b", "auto_acquire.landing_route.a",
              "reference_portals.adapters.openalex_rw.per_page", "reference_portals.caps.enabled", "reference_portals.x.owner_switch",
              "reference_portals.x.mode", "reference_portals.x.allow_hosts"):
        assert rx.search(k), k
    for k in ("reference_portals.caps.timeout_s", "reference_portals.relevance.min_hits", "auto_acquire.micro_topic.registry_table.cache_ttl_seconds",
              "rate_limits.defaults.increase_frac"):
        assert not rx.search(k), k
    seeded = re.findall(r"^  \('([a-z0-9_.-]+)', '[a-z0-9_-]+', '(?:int|float|bool)'", FWD.read_text(), flags=re.M)
    assert not [k for k in seeded if rx.search(k)]
    assert len(re.findall(r"BEFORE INSERT ON vera_vjay\.set_(settings|deny)_hist", t)) == 2 and "pg_trigger_depth() < 2" in t
    assert "SET LOCAL search_path = pg_catalog, pg_temp" in t and "SET LOCAL search_path = pg_catalog, pg_temp" in _code(RBK)

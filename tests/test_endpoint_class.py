"""safety/endpoint_class.py (PLAN-ENV v2.1 §2, §5): two classes, strict parsing, indicators only add reasons.
Offline: fake hosts (.invalid/.test), fake files under tmp_path, no network."""
import ast
import secrets
from pathlib import Path

import pytest

from safety import endpoint_class as ec

ROOT = Path(__file__).resolve().parents[1]


def _hex():
    return secrets.token_hex(4)


@pytest.fixture
def fake_sources(tmp_path):
    h = _hex()
    prod_host, prod_db, role = f"canary-prod-{h}.invalid", f"canaryprod_{h}", f"canary_rw_{h}"
    vera = tmp_path / ".env-vera"
    vera.write_text(f"VERA_DB_URL_RO=postgresql://{role}:pw{h}@{prod_host}:5432/{prod_db}\n")
    intern = tmp_path / ".env-intern"
    intern.write_text(f"DB_HOST=dpg-{h}\nDB_HOST_EXTERNAL_SUFFIX=.oregon-postgres.render.com\nDB_PORT=5432\n"
                      f"DB_NAME={prod_db}\nDB_ACCOUNT={role}\nDB_ADMIN_ROLE=canary_admin_{h}\n")
    src = ec.Sources.from_files(vera, intern, {})
    return src, dict(h=h, host=prod_host, db=prod_db, role=role)


def test_module_is_stdlib_only():
    tree = ast.parse((ROOT / "safety" / "endpoint_class.py").read_text())
    mods = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            mods |= {a.name.split(".")[0] for a in n.names}
        elif isinstance(n, ast.ImportFrom) and n.module:
            mods.add(n.module.split(".")[0])
    assert not mods & {"dotenv", "sqlalchemy", "psycopg", "psycopg2"}


@pytest.mark.parametrize("scheme", ["postgres", "postgresql", "postgresql+psycopg2", "POSTGRES"])
def test_schemes_normalize(scheme):
    p = ec.parse(f"{scheme}://u:p@db-x.test:5433/vera_test_x")
    assert (p.scheme, p.host, p.port, p.dbname, p.user, p.password) == ("postgresql", "db-x.test", 5433,
                                                                         "vera_test_x", "u", "p")


@pytest.mark.parametrize("url,code", [
    ("mysql://u:p@h.test/db", "bad_scheme"),
    ("h.test/db", "bad_scheme"),
    ("postgresql://u:p@h.test/db?hostaddr=10.0.0.1", "bad_param"),
    ("postgresql://u:p@h.test/db?dbname=other", "bad_param"),
    ("postgresql://u:p@h.test/db?port=5433", "bad_param"),
    ("postgresql://u:p@h.test/db?user=x", "bad_param"),
    ("postgresql://u:p@h.test/db?options=-csearch_path%3Dx", "bad_param"),
    ("postgresql://u:p@h.test/db?passfile=/x", "bad_param"),
    ("postgresql://u:p@h.test/db?sslrootcert=/x", "bad_param"),
    ("postgresql://u:p@h.test/db?target_session_attrs=any", "bad_param"),
    ("postgresql://u:p@h.test/db?host=/var/run/postgresql", "bad_param"),
    ("postgresql://u:p@h.test/db?service=x", "bad_param"),
    ("postgresql://u:p@h.test/db?sslmode=bogus", "bad_param"),
    ("postgresql://u:p@%68.test/db", "percent"),
    ("postgresql://u:p@h.test/d%62", "percent"),
    ("postgresql://u%40x:p@h.test/db", "percent"),
    ("postgresql://u:p@%2Fvar%2Frun/db", "percent"),
    ("postgresql://u:p@/db", "empty_host"),
    ("postgresql://u:p@:5432/db", "empty_host"),
    ("postgresql://u:p@a.test,b.test/db", "multihost"),
    ("postgresql://u:p@[::1]:5432/db", "unparsable"),
    ("postgresql://u:p@h.test:99999/db", "unparsable"),
    ("postgresql://u:p@h.test/", "unparsable"),
    ("postgresql://u:p@h.test", "unparsable"),
    ("postgresql://:p@h.test/db", "unparsable"),
    ("postgresql://u:p@h.test/db#frag", "unparsable"),
    ("postgresql://u:p@h.test/db name", "unparsable"),
    ("", "unparsable"),
])
def test_parse_refusals(url, code):
    with pytest.raises(ec.ParseError) as ei:
        ec.parse(url)
    assert ei.value.code == code
    assert str(ei.value) == code and "h.test" not in repr(ei.value)
    v = ec.classify(url, sources=ec.Sources.empty(), environ={})
    assert v.cls == ec.PRODUCTION and code in v.reasons


def test_percent_password_is_exempt_and_decoded_once():
    p = ec.parse("postgresql://u:a%2540b%3Ac@h.test/db")
    assert p.password == "a%40b:c"
    assert "a%40b" not in repr(p) and "h.test" not in str(p)


def test_trailing_dot_and_case_normalized():
    assert ec.parse("postgresql://u:p@DB.Example.TEST./db").host == "db.example.test"


def test_allowed_params_kept():
    p = ec.parse("postgresql://u:p@h.test/db?sslmode=require&connect_timeout=5&application_name=vera")
    assert p.params == {"sslmode": "require", "connect_timeout": 5, "application_name": "vera"}


@pytest.mark.parametrize("url", [
    "postgresql://u:p@localhost:5433/vera_test_x",
    "postgresql://u:p@127.0.0.1/vera_local_y",
    "postgresql://u:p@fresh-db.test/brand_new",
])
def test_nothing_is_non_production_by_name_or_loopback(url, fake_sources):
    src, _ = fake_sources
    v = ec.classify(url, sources=src, environ={})
    assert v.cls == ec.PRODUCTION and "inconclusive" in v.reasons


def test_no_sources_reason():
    v = ec.classify("postgresql://u:p@h.test/db", sources=ec.Sources.empty(), environ={})
    assert v.cls == ec.PRODUCTION and {"no_sources", "inconclusive"} <= set(v.reasons)


def test_tunnel_to_production_dbname_and_role(fake_sources):
    src, f = fake_sources
    v = ec.classify(f"postgresql://{f['role']}:x@localhost:5433/{f['db']}", sources=src, environ={})
    assert v.cls == ec.PRODUCTION and {"dbname", "role"} <= set(v.reasons)


def test_fingerprint_host_and_trailing_dot(fake_sources):
    src, f = fake_sources
    v = ec.classify(f"postgresql://other:x@{f['host'].upper()}.:5432/{f['db']}", sources=src, environ={})
    assert {"fp", "host_pattern", "dbname"} <= set(v.reasons)


def test_suffix_and_code_floor(fake_sources):
    src, f = fake_sources
    v = ec.classify("postgresql://x:y@dpg-zzz.oregon-postgres.render.com./whatever", sources=src, environ={})
    assert {"suffix", "host_pattern"} <= set(v.reasons)
    v2 = ec.classify("postgresql://x:y@dpg-abc/whatever", sources=ec.Sources.empty(), environ={})
    assert "host_pattern" in v2.reasons and v2.cls == ec.PRODUCTION


def test_ip_indicator_only_with_resolved_set(fake_sources):
    src, f = fake_sources
    src.add_resolved_ips(["192.0.2.7"])
    assert "ip" in ec.classify("postgresql://x:y@192.0.2.7/db", sources=src, environ={}).reasons
    v = ec.classify("postgresql://x:y@alias.test/db", sources=src, environ={}, resolver=lambda h: ["192.0.2.7"])
    assert "ip" in v.reasons


def test_empty_host_with_pghost_is_production():
    v = ec.classify("postgresql://u:p@/db", sources=ec.Sources.empty(), environ={"PGHOST": "/tmp"})
    assert v.cls == ec.PRODUCTION and {"empty_host", "pg_env"} <= set(v.reasons)


def test_ambient_names():
    assert ec.ambient_names({"PGPASSFILE": "x", "KRB5CCNAME": "y", "GSSAPI": "z", "PATH": "p"}) == [
        "GSSAPI", "KRB5CCNAME", "PGPASSFILE"]


def test_verify_disposable_refuses_unconditionally(fake_sources):
    src, _ = fake_sources

    class Conn:
        def cursor(self):
            raise AssertionError("verify_disposable must not touch the connection")

    v = ec.verify_disposable(Conn(), "postgresql://u:p@localhost/vera_test_x", marker_file="/nonexistent",
                             sources=src, environ={})
    assert v.cls == ec.PRODUCTION and ec.VERIFY_DISABLED in v.reasons


def test_disposable_verified_is_never_constructed():
    src = (ROOT / "safety" / "endpoint_class.py").read_text()
    # every Verdict built in endpoint_class is PRODUCTION; no other module builds one
    import re
    calls = re.findall(r"Verdict\(([A-Za-z_]+)", src)
    assert calls and set(calls) == {"PRODUCTION"}
    for f in ("connect_guard.py", "sql_guard.py", "destructive.py"):
        text = (ROOT / "safety" / f).read_text()
        assert "Verdict(" not in text


def test_verdict_and_sources_reprs_are_names_only(fake_sources):
    src, f = fake_sources
    v = ec.classify(f"postgresql://{f['role']}:x@{f['host']}/{f['db']}", sources=src, environ={}, name="VERA_DB_URL_RO")
    for text in (repr(v), str(v.as_dict()), repr(src)):
        for s in (f["host"], f["db"], f["role"], f["h"]):
            assert s not in text
    assert set(v.reasons) <= ec.REASONS

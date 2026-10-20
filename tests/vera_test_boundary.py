"""K2.S-a test boundary: no VERA-DAR test may reach a production database, whatever the shell or any .env holds.

Loaded BY FILE PATH from both conftests (root and tests/) before any application module can be imported, and
installed once (idempotent). It imports only the stdlib, dotenv, sqlalchemy.event/engine and psycopg2 (if
importable); it never imports `main`, `streamlit_app`, `vera.*` or `db`.

What install() does (plan-K2-S 10.3 A1, A2, B1-B4, D1):
  A1/A2  stash every guarded environment name (credentials, DB settings, provider keys) in private memory and
         remove it from os.environ; build deny fingerprints (hashes only) from the outer environment and from the
         .env files; replace dotenv.load_dotenv / dotenv_values with no-ops.
  B1/B2  refuse every non-SQLite database connection BEFORE any socket: a global SQLAlchemy `do_connect` listener and
         wrappers on the driver entry points (psycopg2.connect and, if importable, psycopg/asyncpg/pg8000).
  D1     every refusal is recorded per test item (kind and phase only, never a host or URL).
K2.S-b adds the network boundary: sockets and name resolution to non-loopback hosts are refused (C1-C3), proxy
variables are scrubbed (A1), and live-marked items in a live session get provider keys back and the network lifted
for that item only (A3). The DB boundary is never lifted.
"""
from __future__ import annotations

import atexit
import contextlib
import hashlib
import importlib.machinery
import importlib.util
import inspect
import io
import ipaddress
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from urllib.parse import urlsplit

REPO = Path(__file__).resolve().parent.parent
MODULE_NAME = "tests.vera_test_boundary"

# --- A1: guarded names ---------------------------------------------------------------------------------------
GUARDED_NAMES = frozenset({
    # credentials and keys (read in vera/, main.py, config *_env keys)
    "VERA_API_KEY", "VERA_OWNER_KEY", "OPENAI_API_KEY", "GROQ_API_KEY", "OPENROUTER_API_KEY",
    # database URLs and settings (config db_url_*_env, vera/db.py, week-1v2 db.py, tests/sql_contract_support.py)
    "VERA_DB_URL_RO", "VERA_DB_URL_WO", "VERA_DB_URL_RW", "EXTERNAL_DB_URL", "INTERNAL_DB_URL", "DATABASE_URL",
    "DB_HOST", "DB_PORT", "DB_NAME", "DB_ACCOUNT", "DB_PASSWORD",
    "DB_ADMIN_ROLE", "DB_ADMIN_ROLE_RETRY", "DB_ADMIN_PASSWORD",
    # live hosts and deployment switches
    "VERA_API_BASE_URL", "API_BASE_URL", "VERA_UI_URL", "VERA_LIVE_URL", "RENDER",
    # proxies (S-b): a proxy would route traffic through an allowed loopback address
    "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY", "http_proxy", "https_proxy", "all_proxy", "no_proxy",
    # libpq side files and options (M7): ~/.pgpass, service files and ssl material must not reach an allowed target
    "PGPASSFILE", "PGSERVICEFILE", "PGSYSCONFDIR", "PGSSLROOTCERT", "PGSSLCERT", "PGSSLKEY", "PGOPTIONS",
})
GUARDED_PATTERN = re.compile(
    r"(_KEY|_TOKEN|_SECRET|_PASSWORD|_PASSWD|_CREDENTIALS)$"
    r"|(^|_)(DB|DATABASE|PG)(_[A-Z0-9]+)*_(URL|DSN|HOST|USER|ACCOUNT|NAME|PORT|PASSWORD|ROLE)(_[A-Za-z0-9]+)*$"
    r"|(^|_)(DB|DATABASE)_URL"
    r"|^PG(HOST|USER|PASSWORD|DATABASE|PORT|SERVICE)$"
    r"|^WSL"
    r"|PROXY$", re.I)
PROVIDER_KEY = re.compile(r"(_API_KEY$)|^API_BASE_URL$")  # A3: the only names ever restored, only for live items
PASSTHROUGH = frozenset({"VERA_TEST_DB_URL", "VERA_TEST_DB_DISPOSABLE", "VERA_PUBLIC_MODE"})

# --- B4: deny rules --------------------------------------------------------------------------------------------
DENIED_HOST = re.compile(r"(?i)(\.render\.com$|^dpg-)")
DENIED_USER = re.compile(r"(?i)(admin|_dbadmin|superuser|postgres)$")
_APP_MODULES = ("main", "streamlit_app", "db")


class DBBoundaryError(Exception):
    """A database connection was refused by the test boundary."""


REFUSAL_TEXT = ("test DB boundary: connection refused (an unverified database is production; live DB tests are "
                "disabled: DISPOSABLE-VERIFY-DISABLED-pending-isolation-review)")

_RELOAD_KEEP = ("STATE", "_SEALED", "_ORIG_ENV", "_STASH", "_ORIG")
_RELOAD_PRIOR = {k: globals()[k] for k in _RELOAD_KEEP if k in globals()}   # non-empty only under importlib.reload
STATE: dict = {"reloads": 0, "db_listener": None,"installed": False, "app_modules_at_install": None, "live_enabled": False, "live_item": False,
               "phase": "setup", "violations": [], "deny_fp": frozenset(), "deny_user_fp": frozenset(),
               "deny_db_fp": frozenset(), "deny_suffix": frozenset(), "deny_ip_fp": frozenset(),
               "dotenv_noop": False, "net_lifted": False, "in_item": False, "orphan_violations": [],
               "deny_policy": "unchecked", "deny_all": False,
               "is_child": False, "bootstrap_dir": None,
               "netns": "undecided", "netns_decided": False, "config": None, "unix_roots": [], "unix_check": "unchecked",
               "live_deselected": 0, "seccomp": "none", "sockets_requested": False, "tamper_count": 0}
# K18W M-2 (M2-B): the session's deny decision, sealed by install(). check() combines it monotonically with STATE, so a
# test that writes STATE["deny_all"] = False (or a constant "reset") cannot loosen the DB boundary. A test that really
# needs the allowed path must say so explicitly (monkeypatch both STATE and _SEALED). Session isolation facts (netns,
# seccomp, live_enabled, wrapper identities) are added by seal_session() at pytest_sessionstart. Accident-class control:
# in-process code can still rebind _SEALED deliberately (design 1.6); tamper_message() makes persisting writes visible.
_SEALED: dict = {}
_ORIG_ENV: dict = {}   # snapshot of the environment before install() touched it (deny fingerprints, re-exec)
_STASH: dict = {}   # private; never exposed or logged (live provider keys are restored only by S-b)
_ORIG: dict = {}
# INT2 (verifier LOW, R1-R3): importlib.reload() re-executes this file in the SAME module dict, which used to empty the
# seal, the tamper count and the stash silently. A reload now keeps the live session objects and is itself a tamper:
# it fails the session closed (tamper_count) and is reported by name ("boundary_reloaded").
if (_RELOAD_PRIOR.get("STATE") or {}).get("installed"):
    STATE, _SEALED, _ORIG_ENV, _STASH, _ORIG = (_RELOAD_PRIOR[k] for k in _RELOAD_KEEP)
    STATE["reloads"] = STATE.get("reloads", 0) + 1
    STATE["tamper_count"] += 1
del _RELOAD_PRIOR


def guarded(name: str) -> bool:
    if name in PASSTHROUGH:
        return False
    return name in GUARDED_NAMES or bool(GUARDED_PATTERN.search(name))


def _h(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def fingerprint(host, port, dbname) -> str:
    return _h(f"{(host or '').lower()}|{port or 5432}|{dbname or ''}")


def _from_url(value: str):
    """(host, port, dbname, user) of a postgres-style URL, or None."""
    try:
        u = urlsplit(value)
        if not u.scheme or "://" not in value:
            return None
        return u.hostname or "", u.port or 5432, (u.path or "").lstrip("/"), u.username or ""
    except ValueError:
        return None


# --- E2 (PLAN-ENV v2.1 §3, ENV-D): deny sources are R/config/env, read by a password-dropping parser ---------------
# The legacy policy file (.env.db-accounts in the old course tree beside the main checkout) and the main-checkout .env are
# no longer read. The only file sources are R/config/env/.env-vera and R/config/env/.env-intern, located from this
# file's path (no cwd, $HOME or env override). The runner hides /home (tmpfs), so inside a runner session they are
# invisible and deny_all applies (R-G2). A missing, unsafe or malformed policy source fails closed (deny_all).
ENV_DIR = REPO.parent / "config" / "env"
VERA_ENV_NAME = ".env-vera"
INTERN_ENV_NAME = ".env-intern"              # the policy source: deny_all is False only when it is present and valid
POLICY_STATUSES = ("ok", "missing", "escape", "unreadable", "malformed")
_HOSTNAME = re.compile(r"^[A-Za-z0-9]([A-Za-z0-9.-]*[A-Za-z0-9])?$")
_EC_PRIVATE = "_vera_safety_endpoint_class"
_CG_PRIVATE = "_vera_safety_connect_guard"
# Names kept by the parser. URL-valued names are reduced to a password-less URL at parse time; every other name
# (passwords, keys) is dropped without being kept.
DENY_URL_NAMES = ("EXTERNAL_DB_URL", "INTERNAL_DB_URL", "DATABASE_URL", "VERA_DB_URL_RO", "VERA_DB_URL_RW",
                  "VERA_DB_URL_WO", "VERA_DB_URL_ADMIN")
DENY_PARAM_NAMES = ("DB_HOST", "DB_HOST_EXTERNAL_SUFFIX", "DB_PORT", "DB_NAME", "DB_ACCOUNT", "DB_ADMIN_ROLE",
                    "DB_ADMIN_ROLE_RETRY")


def _gitdir(repo: Path):
    """The git directory of a checkout: repo/.git, or the (repo-relative, never cwd-relative) `gitdir:` target."""
    g = repo / ".git"
    try:
        if g.is_dir():
            return g
        if g.is_file():
            p = Path(g.read_text().split(":", 1)[1].strip())
            return Path(os.path.realpath(p if p.is_absolute() else repo / p))
    except (OSError, IndexError, ValueError):
        pass
    return None


def _main_checkout(repo: Path):
    """The main VERA-DAR checkout, from git metadata only: no environment variable, no cwd, no parent search."""
    repo = Path(repo)
    try:
        if (repo / ".git").is_dir():
            main = repo
        else:
            gd = _gitdir(repo)
            if gd is None:
                return None
            cd = gd / "commondir"
            common = (gd / cd.read_text().strip()) if cd.is_file() else gd.parent.parent
            real = os.path.realpath(common)
            if os.path.basename(real) != ".git":
                return None
            main = Path(real).parent
        if not ((main / ".git").is_dir() and (main / "main.py").is_file() and (main / "vera").is_dir()):
            return None
        return main
    except (OSError, ValueError):
        return None


def _load_safety(name: str, private: str):
    """safety/<name>.py loaded BY FILE PATH under a private module name (never `safety.*`, never `vera.*`)."""
    mod = sys.modules.get(private)
    if mod is None:
        spec = importlib.util.spec_from_file_location(private, REPO / "safety" / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[private] = mod
        try:
            spec.loader.exec_module(mod)
        except BaseException:
            sys.modules.pop(private, None)
            raise
    return mod


def endpoint_class():
    return _load_safety("endpoint_class", _EC_PRIVATE)


def _strip_password(value: str) -> str | None:
    """A password-less form of a URL value (scheme://user@host:port/db), or None if it does not parse. The password is
    never kept, returned or logged. The strict C4 parse is used when it succeeds; the lenient form otherwise, so that a
    URL the strict parser refuses still contributes deny fingerprints (indicators only ever tighten)."""
    try:
        p = endpoint_class().parse(value)
        host, port, db, user = p.host, p.port, p.dbname, p.user
    except Exception:  # noqa: BLE001  (ParseError or any loader failure: fall back to the lenient form)
        t = _from_url(value)
        if not t:
            return None
        host, port, db, user = t
    return f"postgresql://{user + '@' if user else ''}{host}:{port or 5432}/{db}"


def _reduce(mapping) -> dict:
    """Password-dropping reduction of a name->value mapping: URL-valued guarded names become password-less URLs, the
    DB parameter names are kept, everything else is dropped."""
    out = {}
    for k, v in mapping.items():
        if not v or not isinstance(v, str):
            continue
        if k in DENY_PARAM_NAMES:
            out[k] = v.strip()
        elif (k in DENY_URL_NAMES or guarded(k)) and "://" in v:
            r = _strip_password(v.strip())
            if r:
                out[k] = r
    return out


def _parse_env_text(text: str) -> dict:
    """KEY=VALUE lines (optional `export `, quotes stripped, # comments). Only DENY_URL_NAMES and DENY_PARAM_NAMES
    survive, through _reduce; a password is dropped as soon as its line is read."""
    raw: dict = {}
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if s.startswith(("export ", "export\t")):
            s = s[7:].lstrip()
        k, sep, v = s.partition("=")
        k = k.strip()
        if not sep or not k:
            raise ValueError("malformed line")
        if k in DENY_URL_NAMES or k in DENY_PARAM_NAMES:
            v = v.strip()
            if len(v) >= 2 and v[0] == v[-1] and v[0] in "'\"":
                v = v[1:-1]
            raw[k] = v
    out = _reduce(raw)
    raw.clear()
    return out


def _validate_policy(values: dict) -> bool:
    """Structural check of the reduced .env-intern mapping; never returns, prints or logs any value."""
    for k in ("DB_HOST", "DB_NAME", "DB_ACCOUNT", "DB_ADMIN_ROLE"):
        if not (values.get(k) or "").strip():
            return False
    host = values["DB_HOST"].strip()
    try:
        ipaddress.ip_address(host.strip("[]"))
    except ValueError:
        if not _HOSTNAME.match(host):
            return False
    if "DB_PORT" in values:
        try:
            if not 1 <= int(str(values["DB_PORT"]).strip()) <= 65535:
                return False
        except ValueError:
            return False
    return True


def _safe_file(p: Path, env_dir: Path) -> tuple:
    """(status, text | None) for one source file: missing, escape (symlink or a symlinked/foreign ancestor),
    unreadable (not regular, not readable, other owner, group/other writable) or ok with the text read ONCE."""
    import stat as _st
    try:
        if not (p.exists() or p.is_symlink()):
            return "missing", None
        if p.is_symlink() or os.path.realpath(p) != str(p) or os.path.realpath(env_dir) != str(env_dir):
            return "escape", None
        st = p.stat()
        if not (_st.S_ISREG(st.st_mode) and os.access(p, os.R_OK) and st.st_uid == os.getuid()
                and not st.st_mode & 0o022):
            return "unreadable", None
        try:
            return "ok", p.read_text(encoding="utf-8")       # read ONCE: the validated text is what gets parsed (X3)
        except (OSError, UnicodeDecodeError):
            return "unreadable", None
    except (OSError, ValueError):
        return "unreadable", None


def resolve_sources(env_dir: Path | None = None) -> tuple:
    """(status, [reduced mappings]) from R/config/env. status in POLICY_STATUSES; "ok" only when .env-intern is present,
    safe and structurally valid and .env-vera (optional) is absent or safe and parsable. Mappings hold no password."""
    env_dir = Path(env_dir) if env_dir is not None else ENV_DIR
    try:
        if not env_dir.is_absolute():
            return "escape", []
        if env_dir.is_symlink() or not env_dir.is_dir():
            return ("escape" if env_dir.is_symlink() else "missing"), []
    except OSError:
        return "unreadable", []
    out, status = [], "ok"
    for name in (INTERN_ENV_NAME, VERA_ENV_NAME):
        st, text = _safe_file(env_dir / name, env_dir)
        if st == "missing" and name == VERA_ENV_NAME:
            continue
        if st != "ok":
            status = st if status == "ok" else status
            continue
        try:
            vals = _parse_env_text(text)
        except ValueError:
            status = "malformed" if status == "ok" else status
            continue
        finally:
            text = None
        if name == INTERN_ENV_NAME and not _validate_policy(vals):
            status = "malformed" if status == "ok" else status
        if name == VERA_ENV_NAME:
            vals = {k: v for k, v in vals.items() if k in DENY_URL_NAMES}
        out.append(vals)        # even a failed source still adds deny fingerprints (fail closed, tighten only)
    return status, out


def _read_env_files(env_dir: Path | None = None, policy: tuple | None = None) -> list:
    """Reduced (password-less) mappings of R/config/env/.env-intern and .env-vera. Commits the policy status to STATE:
    anything but "ok" sets deny_all (FAIL CLOSED: no usable policy source means maximum deny, never "no deny list").
    `policy` = a precomputed (status, mappings) pair (tests)."""
    status, out = tuple(policy) if policy is not None else resolve_sources(env_dir)
    STATE["deny_policy"] = status
    STATE["deny_all"] = status != "ok"
    return list(out)


def _ip_literal(host: str) -> str | None:
    try:
        return str(ipaddress.ip_address((host or "").strip("[]").split("%")[0]))
    except ValueError:
        return None


def _norm_host(host) -> str:
    h = str(host or "").lower()
    return h[:-1] if h.endswith(".") else h


def _build_deny_sets(sources: list, resolver=None) -> dict:
    """Deny sets as unsalted sha256 hashes (in memory only, never logged):
    fp (host|port|db, incl. DB_HOST and DB_HOST + DB_HOST_EXTERNAL_SUFFIX), user (= PR: every production role/user),
    db (= PD: production dbnames), suffix ((length, hash) of DB_HOST_EXTERNAL_SUFFIX), ip (= PI: IP-literal hosts, plus
    `resolver(host)` results when a resolver is passed; install() passes none, so no DNS happens at install)."""
    fps, users, dbs, suffixes, ips = set(), set(), set(), set(), set()

    def endpoint(host, port, name):
        host = _norm_host(host)
        if host and name:
            fps.add(fingerprint(host, port, name))
        if name:
            dbs.add(_h(name))
        lit = _ip_literal(host)
        if lit:
            ips.add(_h(lit))
        elif host and resolver is not None:
            try:
                for a in resolver(host) or ():
                    ips.add(_h(str(a)))
            except Exception:  # noqa: BLE001  (a failed lookup adds nothing; indicators only tighten)
                pass

    for env in sources:
        env = _reduce(env)
        host, port, name = env.get("DB_HOST"), env.get("DB_PORT"), env.get("DB_NAME")
        suffix = _norm_host(env.get("DB_HOST_EXTERNAL_SUFFIX"))
        try:
            port = int(port) if port else 5432
        except ValueError:
            port = None
        if port is not None and (host or name):
            endpoint(host, port, name)
            if host and suffix:
                endpoint(str(host) + suffix, port, name)
        elif name:
            dbs.add(_h(name))
        if suffix:
            suffixes.add((len(suffix), _h(suffix)))
        for k, v in env.items():
            if "://" in v:
                parsed = _from_url(v)
                if parsed:
                    endpoint(parsed[0], parsed[1], parsed[2])
                    if parsed[3]:
                        users.add(_h(parsed[3].lower()))
            elif k in ("DB_ADMIN_ROLE", "DB_ADMIN_ROLE_RETRY", "DB_ACCOUNT"):
                users.add(_h(v.lower()))
        env.clear()
    return {"fp": frozenset(fps), "user": frozenset(users), "db": frozenset(dbs), "suffix": frozenset(suffixes),
            "ip": frozenset(ips)}


def _build_deny(sources: list) -> tuple[frozenset, frozenset]:
    """(fp, user) of _build_deny_sets (kept for callers that need only those two)."""
    s = _build_deny_sets(sources)
    return s["fp"], s["user"]


def install_connect_guard() -> None:
    """E2: the shared connect guard (safety/connect_guard.py), loaded by path under a private module name, installed in
    the guarded "test" context: every non-SQLite connect it sees refuses with VERIFY_DISABLED before the driver. It is
    installed BEFORE the boundary's own hooks, which therefore stay outermost (the boundary refuses and records first)."""
    _load_safety("connect_guard", _CG_PRIVATE).install("test")


# --- D1: violation recorder --------------------------------------------------------------------------------------
def record(kind: str) -> None:
    if STATE["is_child"] and "_pytest" not in sys.modules and STATE["bootstrap_dir"]:
        try:  # M5: a child process reports to its parent through the bootstrap log (kind only, no host or URL)
            fd = os.open(str(Path(STATE["bootstrap_dir"]) / "violations.log"), os.O_WRONLY | os.O_APPEND)
            try:
                os.write(fd, f"child {kind}\n".encode())
            finally:
                os.close(fd)
        except OSError:
            pass
    if not STATE["in_item"]:  # recorded outside any test item (collection time, session fixtures): fails the session
        STATE["orphan_violations"].append({"kind": kind})
    STATE["violations"].append({"kind": kind, "phase": STATE["phase"]})


def set_phase(phase: str) -> None:
    STATE["phase"] = phase


def begin_item(live_item: bool, sockets: bool = False) -> None:
    """Called from a tryfirst pytest_runtest_setup hook, so refusals during fixture setup are attributed to the item.
    `sockets` (K18W, owner decision OA-2): True only for an item that explicitly requests local sockets
    (marker `vera_sockets`); every other item, collection and session code get no loopback or AF_UNIX connect."""
    STATE["violations"] = []
    STATE["sockets_requested"] = bool(sockets)
    STATE["live_item"] = bool(live_item)
    STATE["net_lifted"] = bool(live_item and STATE["live_enabled"])  # A3/C: network lifted for this item only
    STATE["phase"] = "setup"
    STATE["in_item"] = True


def end_item() -> None:
    STATE["in_item"] = False
    STATE["sockets_requested"] = False
    STATE["live_item"] = False
    STATE["net_lifted"] = False


def provider_env_for_live() -> dict:
    """A3: provider-key names (and API_BASE_URL) from the stash, for a live item in a live session. Never DB or proxy."""
    if not (STATE["live_item"] and STATE["live_enabled"]):
        return {}
    return {k: v for k, v in _STASH.items() if PROVIDER_KEY.search(k)}


def _drain_child_log(phase: str) -> None:
    """Parent only: every line a child appended is a violation of the current item (or an orphan outside any item)."""
    d = STATE["bootstrap_dir"]
    if not d or (STATE["is_child"] and "_pytest" not in sys.modules):   # a nested pytest session drains for its own items
        return
    try:
        fd = os.open(str(Path(d) / "violations.log"), os.O_RDWR)
    except OSError:
        return
    try:
        data = os.read(fd, 1 << 20)
        os.ftruncate(fd, 0)
    finally:
        os.close(fd)
    for _ in range(data.count(b"\n")):
        if STATE["in_item"]:
            STATE["violations"].append({"kind": "child", "phase": phase})
        else:
            STATE["orphan_violations"].append({"kind": "child"})


def orphan_message() -> str | None:
    _drain_child_log("orphan")
    n = len(STATE["orphan_violations"])
    return (f"test boundary violation outside any test: {n} access attempt(s) refused during collection or session "
            "setup (see tests/vera_test_boundary.py)") if n else None


def set_live_enabled(flag: bool) -> None:
    STATE["live_enabled"] = bool(flag)


def violations_message(phase: str, probe: bool = False) -> str | None:
    """Failure text for violations recorded in `phase` of a non-probe item (D2), else None."""
    _drain_child_log(phase)
    mine = [v for v in STATE["violations"] if v["phase"] == phase]
    n = sum(1 for v in mine if v["kind"] == "db")
    m = sum(1 for v in mine if v["kind"] == "net")
    k = sum(1 for v in mine if v["kind"] == "child")
    if probe or not (n or m or k):
        return None
    return (f"test boundary violation: {n} DB / {m} network / {k} child-process access attempt(s) refused; "
            "production or external access is not allowed in tests (see tests/vera_test_boundary.py)")


# --- B: the decision -----------------------------------------------------------------------------------------------
VERIFY_DISABLED = "DISPOSABLE-VERIFY-DISABLED-pending-isolation-review"   # = safety.endpoint_class.VERIFY_DISABLED


def live_preconditions(target: tuple, *, live_item: bool, live_enabled: bool, env, deny_fp, deny_user_fp,
                       deny_all: bool = False, deny_db_fp=frozenset(), deny_suffix=frozenset(),
                       deny_ip_fp=frozenset()) -> str:
    """The first unmet precondition of the live disposable-DB path for target=(host, port, dbname, user), as a code,
    or VERIFY_DISABLED when every static precondition holds. Pure. It never returns an "allowed" value: by binding
    owner decision (2026-10-19) disposable verification is disabled, so the live path ends in VERIFY_DISABLED."""
    host, port, dbname, user = target
    if deny_all:
        return "deny_all"
    if not (live_item and live_enabled):
        return "not_live"
    if env.get("VERA_TEST_DB_DISPOSABLE") != "1":        # an extra precondition only; it can never allow by itself
        return "no_disposable_flag"
    test = _from_url(env.get("VERA_TEST_DB_URL") or "")
    if not test:
        return "no_test_url"
    fp = fingerprint(host, port, dbname)
    tfp = fingerprint(test[0], test[1], test[2])
    if fp != tfp:
        return "not_test_url"
    if fp in deny_fp or tfp in deny_fp:
        return "fp"
    for h in (_norm_host(host), _norm_host(test[0])):
        if DENIED_HOST.search(h):
            return "host_pattern"
        if any(len(h) >= n and _h(h[-n:]) == d for n, d in deny_suffix):
            return "suffix"
        lit = _ip_literal(h)
        if lit and _h(lit) in deny_ip_fp:
            return "ip"
    for d in (dbname or "", test[2] or ""):                # E2: dbname in PD -> refused
        if d and _h(d) in deny_db_fp:
            return "dbname"
    for u in (user or "", test[3] or ""):                  # E2: role in PR -> refused
        if DENIED_USER.search(u) or (u and _h(u.lower()) in deny_user_fp):
            return "role"
    return VERIFY_DISABLED


def decide(target: tuple, *, live_item: bool, live_enabled: bool, env, deny_fp, deny_user_fp, deny_all: bool = False,
           deny_db_fp=frozenset(), deny_suffix=frozenset(), deny_ip_fp=frozenset()) -> bool:
    """True if a non-SQLite connection to target=(host, port, dbname, user) is allowed. Pure.
    Binding owner decision 2026-10-19: an unverified DB is PRODUCTION and disposable verification refuses
    unconditionally (VERIFY_DISABLED); no flag, env var (VERA_TEST_DB_DISPOSABLE=1 included) or argument enables it.
    So this ALWAYS returns False: the live-DB path stays denied. live_preconditions() reports which gate stopped it."""
    live_preconditions(target, live_item=live_item, live_enabled=live_enabled, env=env, deny_fp=deny_fp,
                       deny_user_fp=deny_user_fp, deny_all=deny_all, deny_db_fp=deny_db_fp,
                       deny_suffix=deny_suffix, deny_ip_fp=deny_ip_fp)
    return False


def _target_from(cargs, cparams) -> tuple:
    host, port = cparams.get("host"), cparams.get("port")
    dbname, user = cparams.get("dbname") or cparams.get("database"), cparams.get("user")
    for a in cargs or ():
        if isinstance(a, str) and a:
            try:
                if "://" in a:
                    t = _from_url(a)
                    if t:
                        host, port, dbname, user = host or t[0], port or t[1], dbname or t[2], user or t[3]
                else:
                    kv = dict(p.split("=", 1) for p in a.split() if "=" in p)
                    host, port = host or kv.get("host"), port or kv.get("port")
                    dbname, user = dbname or kv.get("dbname"), user or kv.get("user")
            except ValueError:
                pass
    try:
        port = int(port) if port else 5432
    except (TypeError, ValueError):
        port = 5432
    return str(host or ""), port, str(dbname or ""), str(user or "")


ALLOWED_NETS = tuple(ipaddress.ip_network(n) for n in (
    "127.0.0.0/8", "10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16", "100.64.0.0/10", "::1/128", "fc00::/7"))


def _orig_getaddrinfo(host, port=None):
    """The real resolver (tests monkeypatch this name; never the wrapped socket.getaddrinfo)."""
    return getattr(socket.getaddrinfo, "_vera_original", socket.getaddrinfo)(host, port)


def _addr_allowed(addr: str) -> bool:
    try:
        ip = ipaddress.ip_address(str(addr).split("%")[0])
    except ValueError:
        return False
    return any(ip in n for n in ALLOWED_NETS if n.version == ip.version)


def _checked_address(host: str) -> str:
    """M6: resolve an otherwise-allowed target; every address must be loopback/private/CGNAT/ULA. Returns the first."""
    try:
        if _addr_allowed(host.strip("[]")):
            return host.strip("[]")
        addrs = sorted({ai[4][0] for ai in _orig_getaddrinfo(host)}) if host else []
    except (OSError, ValueError):
        addrs = []
    if not addrs or not all(_addr_allowed(a) for a in addrs):
        record("db")
        raise DBBoundaryError(REFUSAL_TEXT)
    return addrs[0]


def effective_deny_all() -> bool:
    """K18W M-2: monotonic. The sealed session value can only tighten STATE, never loosen it."""
    return bool(STATE["deny_all"]) or bool(_SEALED.get("deny_all", False))


def check(cargs=(), cparams=None) -> None:
    """Raise DBBoundaryError unless the connection is allowed. Called before any socket opens. decide() is always False
    (owner decision 2026-10-19, VERIFY_DISABLED), so every non-SQLite connection is refused and recorded."""
    cparams = {} if cparams is None else cparams
    target = _target_from(cargs, cparams)
    if not decide(target, live_item=STATE["live_item"], live_enabled=STATE["live_enabled"],
                  env=os.environ, deny_fp=STATE["deny_fp"], deny_user_fp=STATE["deny_user_fp"],
                  deny_all=effective_deny_all(), deny_db_fp=STATE["deny_db_fp"],
                  deny_suffix=STATE["deny_suffix"], deny_ip_fp=STATE["deny_ip_fp"]):
        record("db")
        raise DBBoundaryError(REFUSAL_TEXT)
    pin_allowed_target(target, cparams)                    # unreachable while decide() is False (owner decision)


def pin_allowed_target(target: tuple, cparams: dict) -> None:
    """M6/M7 hardening for an allowed target: pin the resolved address (hostaddr; every address must be private or
    loopback, else DBBoundaryError) and disable ~/.pgpass and service files. Kept, and unit-tested directly, for a
    later independently reviewed isolation design; check() never reaches it while decide() is always False."""
    cparams["hostaddr"] = _checked_address(target[0])
    cparams["passfile"] = os.devnull
    cparams.pop("service", None)


def _on_do_connect(dialect, conn_rec, cargs, cparams):
    if getattr(dialect, "name", "") == "sqlite":
        return None
    check(cargs, cparams)
    return None


def _wrap_driver(module, attr: str) -> None:
    orig = getattr(module, attr, None)
    if orig is None or getattr(orig, "_vera_boundary", False):
        return

    def wrapped(*a, **kw):
        check(a, kw)
        _CHECKED.depth = getattr(_CHECKED, "depth", 0) + 1     # INT2 F-1: the raw entry point below is reached only
        try:                                                   # from inside this checked call
            return orig(*a, **kw)
        finally:
            _CHECKED.depth -= 1

    wrapped._vera_boundary = True
    wrapped._vera_original = orig
    setattr(module, attr, wrapped)


# INT2 F-1 (verifier A6/A8/I2): psycopg2.connect() is a thin Python function over the C entry point _connect, which
# test code could call directly (also via SQLAlchemy create_engine(creator=...)), skipping check(). The raw entry point
# is now deny-by-default: it runs only inside a psycopg2.connect() call that check() already allowed (same thread).
# psycopg2.extensions.connection(dsn) (the C type itself) cannot be wrapped in-process (immutable static type;
# replacing it with a subclass would break isinstance); it is covered only by the kernel layer (netns + masked DB
# sockets, see maybe_reexec / scripts/vera_test_runner.py).
_CHECKED = threading.local()
RAW_DB_SITES = (("psycopg2", "_connect"), ("psycopg2._psycopg", "_connect"))


def _wrap_raw_driver(module, attr: str) -> None:
    orig = getattr(module, attr, None)
    if orig is None or getattr(orig, "_vera_boundary", False):
        return

    def wrapped(*a, **kw):
        if getattr(_CHECKED, "depth", 0) <= 0 and not getattr(_INTERNAL, "on", False):   # (kernel_layer_probe only)
            record("db")
            raise DBBoundaryError(REFUSAL_TEXT)
        return orig(*a, **kw)

    wrapped._vera_boundary = True
    wrapped._vera_original = orig
    setattr(module, attr, wrapped)


def _wrap_async(module, attr: str) -> None:
    orig = getattr(module, attr, None)
    if orig is None or getattr(orig, "_vera_boundary", False):
        return

    async def wrapped(*a, **kw):
        check(a, kw)
        return await orig(*a, **kw)

    wrapped._vera_boundary = True
    setattr(module, attr, wrapped)


def _install_db_listener() -> None:
    """The class-level do_connect listener. SQLAlchemy runs dialect do_connect listeners in REGISTRATION order
    (`insert=True` does not reorder them), so install() registers this one before the connect guard's (E2)."""
    from sqlalchemy import event
    from sqlalchemy.engine import Engine
    if STATE["db_listener"] is None:                  # INT2 F-2: one stable listener object (survives a reload)
        STATE["db_listener"] = _on_do_connect
    if not event.contains(Engine, "do_connect", STATE["db_listener"]):
        # a re-add (INT2 F-2 rewrap) must not land behind the connect guard's listener: take the guard's listener out,
        # add the boundary's, then put the guard's back after it
        guard = getattr(sys.modules.get(_CG_PRIVATE), "_GLOBAL", None)
        glis = getattr(guard, "listener", None)
        moved = glis is not None and event.contains(Engine, "do_connect", glis)
        if moved:
            event.remove(Engine, "do_connect", glis)
        event.listen(Engine, "do_connect", STATE["db_listener"])
        if moved:
            event.listen(Engine, "do_connect", glis)


def _install_db_boundary() -> None:
    _install_db_listener()
    for modname, attrs in (("psycopg2", ("connect",)), ("psycopg", ("connect",)), ("pg8000", ("connect",)),
                           ("pg8000.dbapi", ("connect",))):
        if importlib.util.find_spec(modname.split(".")[0]) is None:
            continue
        mod = __import__(modname, fromlist=["x"])
        for a in attrs:
            _wrap_driver(mod, a)
    if importlib.util.find_spec("psycopg2") is not None:
        for modname, attr in RAW_DB_SITES:
            _wrap_raw_driver(__import__(modname, fromlist=["x"]), attr)
    for modname, attrs in (("asyncpg", ("connect", "create_pool")),):
        if importlib.util.find_spec(modname) is not None:
            mod = __import__(modname, fromlist=["x"])
            for a in attrs:
                _wrap_async(mod, a)


def _noop_load_dotenv(*a, **k):
    return False


def _noop_dotenv_values(*a, **k):
    return {}


def _install_dotenv_noop() -> None:
    import dotenv
    import dotenv.main
    _ORIG.update(load=dotenv.load_dotenv, values=dotenv.dotenv_values)
    for mod in (dotenv, dotenv.main):
        mod.load_dotenv = _noop_load_dotenv
        mod.dotenv_values = _noop_dotenv_values
    STATE["dotenv_noop"] = True


def stash_guarded_env(keep_provider: bool = False) -> list:
    """Move guarded names from os.environ into the private stash; returns the names moved (never values).
    A child process (keep_provider) keeps provider keys in its environment as well: they were passed explicitly by
    the parent test, and the network boundary still blocks any provider call from the child."""
    moved = []
    for name in list(os.environ):
        if guarded(name):
            if keep_provider and PROVIDER_KEY.search(name):
                _STASH[name] = os.environ[name]
            else:
                _STASH[name] = os.environ.pop(name)
            moved.append(name)
    return moved


# --- C: network boundary (S-b) -----------------------------------------------------------------------------------
class NetworkBoundaryError(ConnectionRefusedError):
    """A socket connection to a non-loopback address was refused by the test boundary."""


NET_REFUSAL_TEXT = ("test network boundary: connection refused (only loopback and unix sockets are allowed in tests); "
                    "live provider tests need --run-live-stages")
LOCAL_SOCKET_REFUSAL_TEXT = ("test socket boundary: local socket refused (loopback and AF_UNIX connects are denied unless "
                             "the test requests them with @pytest.mark.vera_sockets)")
RESERVED_TLDS = (".invalid", ".test", ".example", ".localhost")


def _is_loopback_host(host) -> bool:
    if host is None:
        return True
    if isinstance(host, (bytes, bytearray)):
        host = bytes(host).decode("ascii", "ignore")
    h = str(host).strip().strip("[]").lower()
    if h in ("", "localhost", "localhost.localdomain", "ip6-localhost"):
        return True
    try:
        return ipaddress.ip_address(h.split("%")[0]).is_loopback
    except ValueError:
        return False


def _reserved_name(host) -> bool:
    if isinstance(host, (bytes, bytearray)):
        host = bytes(host).decode("ascii", "ignore")
    h = str(host or "").lower().rstrip(".")
    return any(h == t[1:] or h.endswith(t) for t in RESERVED_TLDS)


def _wrap_getaddrinfo(orig):
    def getaddrinfo(host, port, *a, **k):
        if STATE["net_lifted"] and not _reserved_name(host):
            return orig(host, port, *a, **k)
        if _is_loopback_host(host):
            return orig(host, port, *a, **k)
        if _reserved_name(host):  # same result the real resolver gives; not a violation
            raise socket.gaierror(socket.EAI_NONAME, "test network boundary")
        record("net")
        raise socket.gaierror(socket.EAI_NONAME, "test network boundary")
    getaddrinfo._vera_boundary = True
    getaddrinfo._vera_original = orig
    return getaddrinfo


def _unix_roots() -> list:
    """Directories a path-based AF_UNIX connect may reach: this session's pytest basetemp and the current process's
    multiprocessing temp dir (plus STATE['unix_roots'], used by tests)."""
    roots = list(STATE["unix_roots"])
    try:
        roots.append(STATE["config"]._tmp_path_factory.getbasetemp())
    except Exception:  # noqa: BLE001
        pass
    try:
        import multiprocessing.process as _mp
        td = _mp.current_process()._config.get("tempdir")
        if td:
            roots.append(td)
    except Exception:  # noqa: BLE001
        pass
    return [os.path.realpath(r) for r in roots]


def _unix_path_ok(address) -> bool:
    if isinstance(address, (bytes, bytearray)):
        if bytes(address)[:1] == b"\0" or not address:   # abstract namespace: per network namespace, isolated by 14
            return True
        address = os.fsdecode(bytes(address))
    if not isinstance(address, str):
        return False
    if address == "" or address.startswith("\0"):
        return True
    real = os.path.realpath(address)
    return any(os.path.commonpath([real, r]) == r for r in _unix_roots() if r)


def _local_target(sock, address) -> bool:
    """True if `address` is a local endpoint (AF_UNIX of any kind, or a loopback host): allowed only on request."""
    if getattr(sock, "family", None) == getattr(socket, "AF_UNIX", object()):
        return True
    return isinstance(address, tuple) and bool(address) and _is_loopback_host(address[0])


def _addr_ok(sock, address) -> bool:
    if getattr(sock, "family", None) == getattr(socket, "AF_VSOCK", object()):
        return False
    if not STATE["sockets_requested"]:          # K18W OA-2: default deny for local sockets too
        return False
    if getattr(sock, "family", None) == getattr(socket, "AF_UNIX", object()):
        return _unix_path_ok(address)
    if isinstance(address, tuple) and address:
        return _is_loopback_host(address[0])
    return False


def _refusal_text(sock, address) -> str:
    return LOCAL_SOCKET_REFUSAL_TEXT if _local_target(sock, address) else NET_REFUSAL_TEXT


def _wrap_connect(orig, ex=False):
    def connect(self, address):
        if STATE["net_lifted"] or _addr_ok(self, address):
            return orig(self, address)
        record("net")
        raise NetworkBoundaryError(_refusal_text(self, address))
    connect._vera_boundary = True
    connect._vera_original = orig
    return connect


def _wrap_create_connection(orig):
    def create_connection(address, *a, **k):
        host = address[0] if isinstance(address, tuple) and address else None
        if STATE["net_lifted"] or _reserved_name(host) or (_is_loopback_host(host) and STATE["sockets_requested"]):
            return orig(address, *a, **k)
        record("net")
        raise NetworkBoundaryError(LOCAL_SOCKET_REFUSAL_TEXT if _is_loopback_host(host) else NET_REFUSAL_TEXT)
    create_connection._vera_boundary = True
    create_connection._vera_original = orig
    return create_connection


def _wrap_send(orig, extract):
    """M8: UDP and datagram sends to a non-loopback destination are refused like connects."""
    def send(self, *a, **k):
        address = extract(a, k)
        if address is not None and not STATE["net_lifted"] and not _addr_ok(self, address):
            record("net")
            raise NetworkBoundaryError(_refusal_text(self, address))
        return orig(self, *a, **k)
    send._vera_boundary = True
    send._vera_original = orig
    return send


def _wrap_socket_init(orig):
    """OD-S6 (C): AF_VSOCK talks to the hypervisor host and is not network-namespaced: refuse it at creation."""
    def __init__(self, *a, **k):
        fam = a[0] if a else k.get("family", -1)
        vs = getattr(socket, "AF_VSOCK", None)
        if vs is not None and fam == vs and not STATE["net_lifted"]:
            record("net")
            raise NetworkBoundaryError(NET_REFUSAL_TEXT)
        return orig(self, *a, **k)
    __init__._vera_boundary = True
    __init__._vera_original = orig
    return __init__


def _install_network_boundary() -> None:
    """Idempotent per site (INT2 F-2: also used to re-install a wrapper that a test removed)."""
    w = lambda f: getattr(f, "_vera_boundary", False)   # noqa: E731
    if not w(socket.getaddrinfo):
        socket.getaddrinfo = _wrap_getaddrinfo(socket.getaddrinfo)
    if not w(socket.socket.__init__):
        socket.socket.__init__ = _wrap_socket_init(socket.socket.__init__)
    if not w(socket.socket.connect):
        socket.socket.connect = _wrap_connect(socket.socket.connect)
    if not w(socket.socket.connect_ex):
        socket.socket.connect_ex = _wrap_connect(socket.socket.connect_ex)
    if not w(socket.create_connection):
        socket.create_connection = _wrap_create_connection(socket.create_connection)
    if not w(socket.socket.sendto):
        socket.socket.sendto = _wrap_send(                                      # sendto(data, addr) / (data, flags, addr)
            socket.socket.sendto, lambda a, k: a[-1] if len(a) >= 2 else k.get("address"))
    if not w(socket.socket.sendmsg):
        socket.socket.sendmsg = _wrap_send(                                     # sendmsg(buffers, anc, flags, addr)
            socket.socket.sendmsg, lambda a, k: a[3] if len(a) > 3 else k.get("address"))


# --- INT2 F-3: audit hook (PEP 578) -------------------------------------------------------------------------------
# The Python wrappers above sit on socket.socket; the C methods of _socket.socket (the base class) are reachable from
# pure Python (`_socket.socket.connect(sock, addr)`, a raw `_socket.socket`), and ctypes can look up libc connect().
# CPython raises an audit event INSIDE the C implementation of connect/connect_ex/sendto/sendmsg/getaddrinfo and for
# every ctypes symbol lookup, and an audit hook can never be removed once added. The hook applies the SAME decision as
# the wrappers (so an allowed call passes both), refusing with the same error and recording the same violation.
# Honest limit (stated): native code that does not go through these entry points (libpq inside psycopg2, a ctypes
# function pointer obtained before install() or by address, cffi, a C extension's own socket calls) raises no such
# event. Only the kernel layer covers it: the network namespace and the masked DB socket directories (maybe_reexec,
# scripts/vera_test_runner.py). The hook also consults mutable STATE, so deliberate in-process tampering stays the
# accepted residual of design 1.6 (detected per item by tamper_keys()).
_AUDIT_SOCKET = frozenset({"socket.connect", "socket.sendto", "socket.sendmsg"})
CTYPES_DENIED_SYMBOLS = frozenset({
    "connect", "socket", "sendto", "sendmsg", "sendmmsg", "bind", "getaddrinfo", "gethostbyname", "system", "popen",
    "fork", "vfork", "clone", "clone3", "execv", "execve", "execvp", "execvpe", "execl", "execle", "execlp",
    "fexecve", "posix_spawn", "posix_spawnp", "syscall", "PQconnectdb", "PQconnectdbParams", "PQconnectStart",
    "PQconnectStartParams", "PQsetdbLogin"})
_INTERNAL = threading.local()       # set only around the boundary's own probes (netns UDP check, seccomp install)


def _audit_hook(event, args):
    if event not in _AUDIT_SOCKET and event != "socket.getaddrinfo" and event != "ctypes.dlsym":
        return
    if not STATE.get("installed") or getattr(_INTERNAL, "on", False) or STATE.get("net_lifted"):
        return
    if event == "ctypes.dlsym":
        name = args[1] if len(args) > 1 else None
        if isinstance(name, str) and name in CTYPES_DENIED_SYMBOLS:
            record("net")
            raise NetworkBoundaryError(NET_REFUSAL_TEXT + " (ctypes lookup of " + name + ")")
        return
    if event == "socket.getaddrinfo":
        host = args[0] if args else None
        if _is_loopback_host(host):
            return
        if _reserved_name(host):
            raise socket.gaierror(socket.EAI_NONAME, "test network boundary")
        record("net")
        raise socket.gaierror(socket.EAI_NONAME, "test network boundary")
    sock = args[0] if args else None
    address = args[1] if len(args) > 1 else None
    if event != "socket.connect" and address is None:
        return
    if not _addr_ok(sock, address):
        record("net")
        raise NetworkBoundaryError(_refusal_text(sock, address))


@contextlib.contextmanager
def kernel_layer_probe():
    """TEST-ONLY and explicit: steps the audit hook aside so a test can prove the KERNEL layer (netns, seccomp, masks)
    with raw _socket / ctypes calls. Call sites are pinned by test_kernel_layer_probe_call_sites_are_pinned."""
    _INTERNAL.on = True
    try:
        yield
    finally:
        _INTERNAL.on = False


def _install_audit_hook() -> None:
    if not STATE.get("audit_hook"):
        sys.addaudithook(_audit_hook)
        STATE["audit_hook"] = True


# --- F2: child processes (S-b revision, plan section 13) ---------------------------------------------------------
MARKER = "VERA_TEST_BOUNDARY_CHILD"
STRIP_ENV = {"PATH": ""}
# Reviewed allowlist (ALLOWED_CHILDREN). Each entry is an exact argv shape plus pinned content; anything else that is
# not an ordinary (non-isolated) Python child is refused. Re-review this list when a hash below fails.
#   1. [<REPO>/start.sh, "nope"]: exits 2 before any exec (tests/test_item73_deploy.py:48). start.sh is pinned by hash.
START_SH_SHA = "e689b25d7913e72053ec67d5179c14487925373b0cf230384e9135d903ee70f1"   # ENV-A C11 start.sh (2026-10-19, SEC)
#   2. [<abs python>, "-I", "-B", "-c", P] with env == {"PATH": ""}: P is the program at tests/test_answer_contract.py:126,
#      hashed with the checkout path replaced by <ROOT>. It runs WITHOUT the boundary (-I ignores PYTHONPATH): accepted
#      residual. -B is required (DEF-1 R1) so the child writes no bytecode into the checkout; the old -I -c shape is refused.
ANSWER_CONTRACT_CHILD_SHA = "129913de7c5aa1f191c43bd37a4b795f4ca4e25b58ccc599c975c67ee94733e9"
#   3. Read-only git calls made by production code: vera/m6/fingerprint.py:156 (rev-parse HEAD; status --porcelain --
#      vera config), vera/run_mvp.py:159 (rev-parse HEAD; status --porcelain), vera/mvp_preflight.py:73
#      (check-ignore -q <relative path>). Added in the S-b revision because the first full run showed the spawn guard
#      refusing them (30 tests); exact argv shapes only, env scrubbed. NEEDS REVIEW by the vetter.
ALLOWED_GIT = (("git", "rev-parse", "HEAD"), ("git", "status", "--porcelain"),
               ("git", "status", "--porcelain", "--", "vera", "config"))
GIT_CONFIG_OVERRIDES = (("core.fsmonitor", "false"), ("core.hooksPath", "/dev/null"), ("core.pager", "cat"),
                        ("credential.helper", ""), ("protocol.allow", "never"), ("core.untrackedCache", "false"))
_GIT_REL_PATH = re.compile(r"^[A-Za-z0-9._][A-Za-z0-9._/-]*$")
CHILD_REFUSAL_TEXT = ("test child-process boundary: spawn refused (only Python children that inherit the boundary, or "
                      "reviewed ALLOWED_CHILDREN entries, may start); see tests/vera_test_boundary.py")
CHANGED_TEXT = "allowlisted child changed; re-review ALLOWED_CHILDREN"


class ChildProcessBoundaryError(PermissionError):
    """A child process was refused by the test boundary."""


CHILD_BOOTSTRAP = """\
# Written by tests/vera_test_boundary.py for child Python processes: installs the same boundary first thing.
import importlib.machinery, importlib.util, os, sys
try:
    _B = @@BOUNDARY@@
    if "tests.vera_test_boundary" not in sys.modules:
        _s = importlib.util.spec_from_file_location("tests.vera_test_boundary", _B)
        _m = importlib.util.module_from_spec(_s)
        sys.modules["tests.vera_test_boundary"] = _m
        _s.loader.exec_module(_m)
    sys.modules["tests.vera_test_boundary"].install()
except BaseException:  # fail closed and loud: a child that cannot install the boundary must not run unguarded
    os.write(2, b"test child-process boundary: install failed in child\\n")
    os._exit(98)
_here = os.path.dirname(os.path.abspath(__file__))
_s2 = importlib.machinery.PathFinder.find_spec("sitecustomize", [p for p in sys.path if os.path.abspath(p or ".") != _here])
if _s2 is not None and _s2.loader is not None:
    _m2 = importlib.util.module_from_spec(_s2)
    _s2.loader.exec_module(_m2)
"""


def _under(path: Path, base: Path) -> bool:
    try:
        return os.path.commonpath([os.path.realpath(path), os.path.realpath(base)]) == os.path.realpath(base)
    except ValueError:
        return False


def _setup_child_bootstrap() -> None:
    """M4: the bootstrap directory lives under the checkout's own git directory (never the system tmp)."""
    STATE["is_child"] = os.environ.get(MARKER) == "1"
    gd = _gitdir(REPO)
    if gd is None:
        raise RuntimeError("test boundary: cannot locate the git directory for the child bootstrap")
    base = Path(gd) / "vera-test-boundary"
    old = os.environ.get("PYTHONPATH") or ""
    first = old.split(os.pathsep)[0]
    if STATE["is_child"] and first and Path(first).is_dir() and _under(Path(first), base):
        STATE["bootstrap_dir"] = Path(first)          # reuse; create nothing, remove nothing
        STATE["bootstrap_reused"] = True              # INT2: a real child of a boundary session (launch_sanctioned)
        return
    os.makedirs(base, mode=0o700, exist_ok=True)
    d = Path(tempfile.mkdtemp(prefix="run-", dir=str(base)))
    os.chmod(d, 0o700)
    (d / "sitecustomize.py").write_text(CHILD_BOOTSTRAP.replace("@@BOUNDARY@@", repr(str(Path(__file__).resolve()))))
    os.close(os.open(str(d / "violations.log"), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600))
    os.environ["PYTHONPATH"] = str(d) + (os.pathsep + old if old else "")
    os.environ[MARKER] = "1"
    STATE["bootstrap_dir"] = d
    atexit.register(shutil.rmtree, str(d), True)      # removes only the directory this process created


def _child_exit_hook() -> None:
    """M5: a non-pytest child that recorded a refusal exits non-zero even if its own code swallowed the error."""
    if STATE["violations"] and "_pytest" not in sys.modules:
        for stream in (sys.stdout, sys.stderr):
            try:
                stream.flush()
            except Exception:  # noqa: BLE001
                pass
        os._exit(97)


PYTHON_ALLOWLIST: frozenset = frozenset()   # M1-B: further reviewed interpreter real paths (none at present)


def _resolve_program(prog, env=None, search: bool = False, cwd=None):
    """M1-A: the program the kernel will actually run, or None if it cannot be determined (the caller refuses).
    `search`: resolve a bare name through PATH (of `env`, else os.environ), as the exec*p/posix_spawnp variants and
    Popen do; otherwise the path as given, made absolute as the kernel resolves it. S-4: `cwd` is the directory the
    child starts in (Popen cwd=): relative paths and relative PATH entries (an empty entry is the cwd) resolve there."""
    if isinstance(prog, int) or prog is None:          # fexecve by fd, or no program: cannot be classified
        return None
    p = os.fsdecode(prog)
    if not p:
        return None
    base = os.fsdecode(cwd) if cwd is not None else os.getcwd()
    if search and os.sep not in p:
        e = os.environ if env is None else env
        entries = (e.get("PATH", os.defpath) or "").split(os.pathsep)
        path = os.pathsep.join(x if os.path.isabs(x) else os.path.join(base, x) for x in entries)
        return shutil.which(p, path=path)
    return os.path.normpath(os.path.join(base, p))


def _is_python(argv0, env=None, cwd=None) -> bool:
    """M1-B (PF-18 M-1, L-16): a Python child is the running interpreter, identified by the real path of the program
    that will run, never by a basename such as "python3" (which names whatever binary is first on PATH)."""
    a = _resolve_program(argv0, env, search=True, cwd=cwd)
    if a is None:
        return False
    real = os.path.realpath(a)
    return real == os.path.realpath(sys.executable) or real in PYTHON_ALLOWLIST


def _isolated(argv, env) -> bool:
    """M3/X2: -I, -E or -S anywhere in the interpreter flags (also combined, e.g. -IS, -sS, and after options that take
    a value such as -W, -X, --check-hash-based-pycs), or a user-site-off + cleared PYTHONPATH environment, makes a
    Python child 'isolated' (sitecustomize would not load)."""
    e = env if env is not None else os.environ
    env_isolated = "PYTHONNOUSERSITE" in e and not e.get("PYTHONPATH")
    i, toks = 1, list(argv)
    while i < len(toks):
        tok = toks[i]
        if tok == "--" or not tok.startswith("-") or tok == "-":
            break                                    # script path, stdin or end of options
        if tok.startswith("--"):
            if tok.split("=", 1)[0] == "--check-hash-based-pycs" and "=" not in tok:
                i += 1                               # long option with a separate argument
            i += 1
            continue
        flags = tok[1:]
        for j, ch in enumerate(flags):
            if ch in "IES":
                return True
            if ch in "cm":
                return env_isolated
            if ch in "WX":
                if j == len(flags) - 1:
                    i += 1                           # argument is the next token
                break                                # rest of the cluster is the argument
        i += 1
    return env_isolated


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _match_allowed(argv, env_given, env):
    """'ok' | 'changed' | None for the reviewed ALLOWED_CHILDREN."""
    if list(argv) == [str(REPO / "start.sh"), "nope"]:
        try:
            return "ok" if _sha((REPO / "start.sh").read_bytes()) == START_SH_SHA else "changed"
        except OSError:
            return "changed"
    if _is_allowed_git(argv):
        return "ok"
    # S-4: the pinned child must name the interpreter by ABSOLUTE path (its env PATH is empty, so a bare name would
    # be looked up in the cwd)
    if len(argv) == 5 and os.path.isabs(argv[0]) and _is_python(argv[0]) and argv[1:4] == ["-I", "-B", "-c"]:
        norm = argv[4].replace(repr(str(REPO)), "<ROOT>")
        if _sha(norm.encode()) == ANSWER_CONTRACT_CHILD_SHA and env_given and dict(env) == STRIP_ENV:
            return "ok"
    return None


def _refuse_spawn(text: str = CHILD_REFUSAL_TEXT):
    record("child")
    raise ChildProcessBoundaryError(text)


def _is_allowed_git(argv) -> bool:
    return tuple(argv) in ALLOWED_GIT or (len(argv) == 4 and tuple(argv[:3]) == ("git", "check-ignore", "-q")
                                          and bool(_GIT_REL_PATH.match(argv[3])) and ".." not in argv[3].split("/"))


def _pinned_git():
    """The system git, resolved on a fixed PATH and required to live under /usr/ (never a PATH-shadowed binary)."""
    p = shutil.which("git", path="/usr/bin:/bin")
    if p is None:
        return None
    real = os.path.realpath(p)
    return p if real.startswith("/usr/") else None


def _git_env() -> dict:
    """Hardened environment for the allowlisted git children: no inherited GIT_*, fixed PATH, no user or system
    config, no hooks, no fsmonitor, no prompts, no optional locks."""
    home = "/nonexistent"
    bd = STATE.get("bootstrap_dir")
    if bd:
        try:
            home = tempfile.mkdtemp(prefix="githome-", dir=str(bd))
        except OSError:
            pass
    env = {"PATH": "/usr/bin:/bin", "HOME": home, "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": "/dev/null",
           "GIT_TERMINAL_PROMPT": "0", "GIT_OPTIONAL_LOCKS": "0", "GIT_NO_LAZY_FETCH": "1",
           "GIT_CONFIG_COUNT": str(len(GIT_CONFIG_OVERRIDES))}
    for i, (k, v) in enumerate(GIT_CONFIG_OVERRIDES):
        env[f"GIT_CONFIG_KEY_{i}"], env[f"GIT_CONFIG_VALUE_{i}"] = k, v
    return env


def _inject(env):
    e = {k: v for k, v in dict(os.environ if env is None else env).items() if not k.upper().startswith("WSL")}
    bd = str(STATE["bootstrap_dir"])
    rest = [x for x in (e.get("PYTHONPATH") or "").split(os.pathsep) if x and x != bd]
    e["PYTHONPATH"] = os.pathsep.join([bd] + rest)
    e[MARKER] = "1"
    e["PYTHONDONTWRITEBYTECODE"] = "1"   # DEF-1 R2: constant, no passthrough; injected children write no bytecode in the checkout
    return e


def _spawn_env(argv, env, shell=False, via_popen=False, cwd=None):
    """The boundary decision for one spawn. Returns the env to use (a copy), or None to leave the env alone.
    Raises ChildProcessBoundaryError (recorded) for a refused spawn."""
    if shell or not argv:
        _refuse_spawn()
    argv = [os.fsdecode(a) for a in argv]
    allowed = _match_allowed(argv, env is not None, env or {})
    if allowed == "changed":
        _refuse_spawn(CHANGED_TEXT)
    if allowed == "ok":
        if _is_allowed_git(argv):
            if not via_popen or _pinned_git() is None:    # git only through Popen, only the pinned system binary
                _refuse_spawn(CHANGED_TEXT if via_popen else CHILD_REFUSAL_TEXT)
            return _git_env()
        base = dict(os.environ if env is None else env)
        return {k: v for k, v in base.items() if not guarded(k)}
    if _is_python(argv[0], env, cwd) and not _isolated(argv, env):
        return _inject(env)
    _refuse_spawn()


def _bootstrap_inherited() -> bool:
    """M1-A: an env-less exec/spawn inherits os.environ; the child gets the boundary only if os.environ still carries
    the bootstrap (first PYTHONPATH entry is this session's bootstrap dir, MARKER set)."""
    bd = STATE.get("bootstrap_dir")
    first = (os.environ.get("PYTHONPATH") or "").split(os.pathsep)[0]
    return bool(bd) and first == str(bd) and os.environ.get(MARKER) == "1"


_POPEN_SIG = inspect.signature(getattr(subprocess.Popen.__init__, "_vera_original", subprocess.Popen.__init__))


def _wrap_popen() -> None:
    orig = subprocess.Popen.__init__
    if getattr(orig, "_vera_boundary", False):
        return

    def __init__(self, *a, **kw):
        ba = _POPEN_SIG.bind(self, *a, **kw)
        args, exe = ba.arguments.get("args"), ba.arguments.get("executable")
        shell = bool(ba.arguments.get("shell"))
        if isinstance(args, (str, bytes, os.PathLike)):
            argv = [os.fsdecode(args)]
            shell = shell or (isinstance(args, (str, bytes)) and " " in os.fsdecode(args).strip())
        else:
            argv = [os.fsdecode(x) for x in args]
        if exe:
            argv = [os.fsdecode(exe)] + argv[1:]
        new_env = _spawn_env(argv, ba.arguments.get("env"), shell=shell, via_popen=True, cwd=ba.arguments.get("cwd"))
        if _is_allowed_git(argv):                      # run the pinned binary, never whatever PATH resolves
            ba.arguments["args"] = [_pinned_git()] + argv[1:]
            ba.arguments["executable"] = None
        if new_env is not None:
            ba.arguments["env"] = new_env
        return orig(*ba.args, **ba.kwargs)

    __init__._vera_boundary = True
    __init__._vera_original = orig
    subprocess.Popen.__init__ = __init__


def _wrap_os_call(owner, name: str, argv_i, env_i, path_i=None, search=False) -> None:
    orig = getattr(owner, name, None)
    if orig is None or getattr(orig, "_vera_boundary", False):
        return

    def wrapped(*a, **k):
        a = list(a)
        if argv_i is None:                                   # os.system(command): never allowed
            _refuse_spawn()
        argv = a[argv_i] if len(a) > argv_i else k.get("argv", k.get("args"))
        env = a[env_i] if env_i is not None and len(a) > env_i else k.get("env")
        if path_i is not None:                               # M1-A: classify the program that runs, not argv[0]
            argv = list(argv or ())
            prog = a[path_i] if len(a) > path_i else k.get("path", k.get("file"))
            resolved = _resolve_program(prog, None if search == "parent" else env, search=bool(search))
            if not argv or resolved is None:
                _refuse_spawn()
            argv = [resolved] + argv[1:]
        new_env = _spawn_env(list(argv), env)
        if new_env is not None and env_i is None and not _bootstrap_inherited():
            _refuse_spawn()                                  # M1-A: env-less site, bootstrap no longer in os.environ
        if new_env is not None and env_i is not None:
            if len(a) > env_i:
                a[env_i] = new_env
            elif env is not None or "env" in k:
                k["env"] = new_env
        return orig(*a, **k)

    wrapped._vera_boundary = True
    wrapped._vera_original = orig
    setattr(owner, name, wrapped)


SPAWN_SITES = (("system", None, None), ("execv", 1, None), ("execve", 1, 2), ("execvp", 1, None), ("execvpe", 1, 2),
               ("spawnv", 2, None), ("spawnve", 2, 3), ("spawnvp", 2, None), ("spawnvpe", 2, 3),
               ("posix_spawn", 1, 2), ("posix_spawnp", 1, 2))
# M1-A: where each site's program argument is (index), and how a bare name is resolved: False = not searched (path
# as given), "env" = PATH of the env passed (else os.environ; os._execvpe semantics), "parent" = PATH of this
# process (glibc posix_spawnp searches the caller's PATH, not the child env)
SPAWN_PROGRAM = {"execv": (0, False), "execve": (0, False), "execvp": (0, "env"), "execvpe": (0, "env"),
                 "spawnv": (1, False), "spawnve": (1, False), "spawnvp": (1, "env"), "spawnvpe": (1, "env"),
                 "posix_spawn": (0, False), "posix_spawnp": (0, "parent")}


# INT2 F-5 (verifier 2c): under `python -I` (the release gate's own pytest argv, release_gate.py:59) multiprocessing
# starts its children (spawn workers, the resource tracker) with the parent's flags, i.e. with -I, so the bootstrap
# sitecustomize cannot load and the child was refused (fail closed, but it made every gate run fail). For exactly the
# two stdlib child programs below, the boundary is now installed EXPLICITLY at the top of the child's -c program,
# loaded by file path; -I and every other flag stay. Any other isolated Python child is still refused.
_MP_CHILD_CODE = re.compile(
    r"from multiprocessing\.spawn import spawn_main; spawn_main\((?:(?:tracker_fd|pipe_handle|parent_pid)=\d+)"
    r"(?:, (?:tracker_fd|pipe_handle|parent_pid)=\d+)*\)"
    r"|from multiprocessing\.resource_tracker import main;main\(\d+\)")
_MP_FLAG = re.compile(r"-[BIEsSPOqbuvd]+|-W.+")
MP_CHILD_PREFIX = (
    "import importlib.util as _vu, os as _vo, sys as _vs\n"
    "try:\n"
    "    if 'tests.vera_test_boundary' not in _vs.modules:\n"
    "        _vsp = _vu.spec_from_file_location('tests.vera_test_boundary', @@BOUNDARY@@)\n"
    "        _vm = _vu.module_from_spec(_vsp); _vs.modules['tests.vera_test_boundary'] = _vm; _vsp.loader.exec_module(_vm)\n"
    "    _vs.modules['tests.vera_test_boundary'].install()\n"
    "except BaseException:\n"
    "    _vo.write(2, b'test child-process boundary: install failed in child\\n'); _vo._exit(98)\n")


def mp_child_argv(argv):
    """The rewritten argv for an ISOLATED multiprocessing stdlib child of the exact shapes above, else None."""
    argv = [os.fsdecode(a) for a in argv]
    if len(argv) < 3 or not _is_python(argv[0]) or "-c" not in argv:
        return None
    ci = argv.index("-c")
    flags, rest = argv[1:ci], argv[ci + 2:]
    i = 0
    while i < len(flags):                                   # only interpreter flags, -X takes a separate value
        if flags[i] == "-X" and i + 1 < len(flags):
            i += 2
            continue
        if not _MP_FLAG.fullmatch(flags[i]):
            return None
        i += 1
    if ci + 1 >= len(argv) or not _MP_CHILD_CODE.fullmatch(argv[ci + 1]) or rest not in ([], ["--multiprocessing-fork"]):
        return None
    if not _isolated(argv, None):
        return None                                         # a non-isolated child gets the boundary via sitecustomize
    prefix = MP_CHILD_PREFIX.replace("@@BOUNDARY@@", repr(str(Path(__file__).resolve())))
    return argv[:ci + 1] + [prefix + argv[ci + 1]] + rest


def _wrap_mp_spawnv_passfds(mpu) -> None:
    """Layer over the generic spawnv_passfds wrapper (installed by _install_spawn_guard, unchanged)."""
    checked = getattr(mpu, "spawnv_passfds", None)
    if checked is None or getattr(checked, "_vera_mp_layer", False) or not getattr(checked, "_vera_boundary", False):
        return
    generic = getattr(checked, "_vera_original", None)
    if generic is None:
        return

    def spawnv_passfds(path, args, passfds):
        rewritten = mp_child_argv(args)
        if rewritten is None:
            return checked(path, args, passfds)             # the generic rule (refuses an isolated Python child)
        if os.path.realpath(os.fsdecode(path)) != os.path.realpath(rewritten[0]) or not _bootstrap_inherited():
            _refuse_spawn()
        return generic(path, rewritten, passfds)

    spawnv_passfds._vera_boundary = True
    spawnv_passfds._vera_mp_layer = True
    spawnv_passfds._vera_original = generic
    mpu.spawnv_passfds = spawnv_passfds


def _install_spawn_guard() -> None:
    _wrap_popen()
    for name, argv_i, env_i in SPAWN_SITES:
        _wrap_os_call(os, name, argv_i, env_i, *SPAWN_PROGRAM.get(name, (None, False)))
    import multiprocessing.util as mpu
    _wrap_os_call(mpu, "spawnv_passfds", 1, None, 0, False)
    # NOTE: native code (raw _socket, ctypes, cffi) and processes started by it are NOT intercepted here; that needs
    # OS-level isolation and is planned separately. A later network-isolation check can hook in at this point.


# --- 14: OS-level network isolation (kernel-enforced; the Python boundaries above stay as defence in depth) ----------
EXIT_ISOLATION = 78            # EX_CONFIG: distinct from pytest's own exit codes 0-5
REEXEC_MARKER = "VERA_TEST_NETNS_REEXEC"   # loop guard only; it never grants or skips anything
NETNS_MESSAGE = ("test network isolation unavailable: the run could not enter an unprivileged network namespace "
                 "(unshare -rn) and is not in a --network none container. Tests do not run without OS-level isolation.")
ESCAPE_MESSAGE = ("test isolation: a host control socket (container runtime, user service bus or terminal multiplexer) "
                  "is reachable inside the test namespace; refusing to run")
STDIN_MESSAGE = "test network isolation: re-exec impossible (script read from stdin)"
ESCAPE_DEVICE_FILES = ("/dev/vsock",)          # bind-mounted over with /dev/null in the mount namespace (OD-S6 C)
WSL_INTEROP_BINFMT = "/proc/sys/fs/binfmt_misc/WSLInterop"
WSL_INTEROP_DIR = "/run/WSL"
ESCAPE_SOCKET_FILES = ("/run/dbus/system_bus_socket", "/var/run/docker.sock", "/run/docker.sock", "/run/containerd/containerd.sock",
                       "/run/containerd/containerd.sock.ttrpc", "/run/podman/podman.sock", "/run/crio/crio.sock",
                       "/run/user/{uid}/docker.sock", "/run/user/{uid}/podman/podman.sock")
ESCAPE_SOCKET_DIRS = ("/run/user/{uid}", "/tmp/tmux-{uid}", "/run/screen/S-{user}", "/run/WSL",
                      # INT2 F-1: libpq's default Unix socket directories (a LIVE host Postgres listens there)
                      "/run/postgresql", "/var/run/postgresql")
# INT2 F-1: Postgres sockets outside a socket directory (unix_socket_directories = '/tmp' and similar): every matching
# file found at re-exec time is bound over with /dev/null. Checked (stat only, never connected) by db_socket_visible().
DB_SOCKET_SCAN_DIRS = ("/run/postgresql", "/var/run/postgresql", "/tmp", "/var/tmp", "/dev/shm", "/var/pgsql_socket",
                       "/var/lib/postgresql")      # INT3 MED-B: a socket dir some packagings use (listed, never opened)
DB_SOCKET_PREFIX = ".s.PGSQL."
DB_SOCKET_MESSAGE = ("test isolation: a PostgreSQL socket (.s.PGSQL.*) is visible inside the test namespace; refusing to "
                     "run. Launch through scripts/vera_test_runner.py, which masks the socket directories")
# INT2 F-1/F-4: sanctioned launches. scripts/vera_test_runner.py sets RUNNER_MARKER (it pins argv, env -i, masks
# /home, the socket dirs and credential files); the release gate sets a 32-hex VERA_GATE_NONCE; OFFLINE_ESCAPE is the
# opt-in for an already offline-only container (for example `docker run --network none`). These names are
# accident-class signals, not authentication.
# INT3 MED-A: every signal (runner marker, gate nonce, OFFLINE_ESCAPE, child marker, the old re-exec marker) is
# honoured ONLY when the session is verifiably isolated ALREADY (session_isolation_problems() == []): only `lo`, no
# route and no reachable UDP; no Postgres socket and no host control socket in the listed places; no known credential
# file readable. Otherwise the session is refused (exit 78) with a message naming the failed check. There is NO
# fallback to the in-repo netns re-exec chain any more (reexec_argv/SCRIPT/OUTER stay as library code only; nothing
# in maybe_reexec calls them), because that chain left HOME and checkout credential files readable (WS-8g GB2/GB5/GB6).
# Inherent residual (documented, not fixable in-repo): on a BARE launch inside an isolated host, test-tree code that
# runs before maybe_reexec (a conftest writing the snapshot, a `-p` plugin) can still self-sanction; the kernel facts
# checked here are then the only control (WS-8g GA2/GA3).
RUNNER_MARKER = "VERA_TEST_RUNNER"
OFFLINE_ESCAPE = "VERA_TEST_OFFLINE_ESCAPE"
RUNNER_MESSAGE = ("test isolation: tests must be launched through the sanctioned runner: "
                  "`python -I scripts/vera_test_runner.py -- tests/...` (the release gate must launch through it too). "
                  "See tests/vera_test_boundary.py (INT2 F-1, INT3 MED-A)")
ISOLATION_REQUIRED_MESSAGE = ("test isolation: this session is not network-isolated (an interface other than lo, a "
                              "route, or a reachable UDP destination is present). A launch signal is honoured only "
                              "inside an already isolated session and there is no in-repo fallback; launch through "
                              "`python -I scripts/vera_test_runner.py -- tests/...` (INT3 MED-A)")
CREDENTIAL_MESSAGE = ("test isolation: a known credential file (~/.pgpass, ~/.pg_service.conf, a checkout or "
                      "main-checkout .env, or a credential-type file in the checkout) is readable inside the test "
                      "session; refusing to run. scripts/vera_test_runner.py masks them (INT3 MED-A)")
CRED_FILE_GLOBS = (".env", ".env.*", "*.env", "*.env.*", "*-prv.env", "*secret*", "*.pem", "*.key", ".pgpass",
                   ".pg_service.conf", ".env.db-accounts", ".env-*", "*.env-*")
# = scripts/vera_test_runner.py CRED_MASK_GLOBS (E1b/C10: equality is asserted by tests/test_env_guard.py)
CRED_FILE_EXCLUDE = ("*.example",)
CRED_ROOT_SKIP = (".git",)   # INT4 (WS-8h LOW-MED): only the root .git and the pinned venv are skipped, no name below


def _readable_nonempty(path) -> bool:
    """A regular file (after symlinks) that this process can read and that holds bytes. /dev/null binds (the runner's
    masks) and absent files are not visible. stat/access only; the content is never opened."""
    import stat as _st
    try:
        st = os.stat(path)
    except OSError:
        return False
    return _st.S_ISREG(st.st_mode) and st.st_size > 0 and os.access(path, os.R_OK)


def _pinned_venv():
    """Real path of the running venv (sys.prefix), or None outside a venv (= scripts/vera_test_runner.py)."""
    pre, base = os.path.realpath(sys.prefix), os.path.realpath(sys.base_prefix)
    return pre if pre != base else None


def credential_files_visible(homes=None, repo=None, venv=None) -> list:
    """INT3 MED-A: known credential files readable in this session (paths; callers report names/counts only).
    INT4: the checkout walk skips only <repo>/.git and the pinned venv (default: sys.prefix), so a credential-type
    file under tests/node_modules, tests/venv or __pycache__ is seen (and is masked by the runner, same rule)."""
    import fnmatch
    import pwd
    if homes is None:
        homes = [_ORIG_ENV.get("HOME") or os.environ.get("HOME") or ""]
        try:
            homes.append(pwd.getpwuid(os.getuid()).pw_dir)
        except (KeyError, OSError):
            pass
    repo = Path(repo or REPO)
    cands = [os.path.join(h, n) for h in dict.fromkeys(homes) if h for n in (".pgpass", ".pg_service.conf")]
    cands.append(str(repo / ".env"))
    try:
        main = _main_checkout(repo)
    except Exception:  # noqa: BLE001
        main = None
    if main is not None and main != repo:
        cands.append(str(Path(main) / ".env"))
    root = os.path.realpath(str(repo))
    venv = os.path.realpath(venv) if venv else _pinned_venv()
    for d, dirs, files in os.walk(root):
        dirs[:] = [x for x in dirs if not ((d == root and x in CRED_ROOT_SKIP)
                                           or (venv and os.path.realpath(os.path.join(d, x)) == venv))]
        for f in files:
            if any(fnmatch.fnmatchcase(f, x) for x in CRED_FILE_EXCLUDE):
                continue
            if any(fnmatch.fnmatchcase(f, g) for g in CRED_FILE_GLOBS):
                cands.append(os.path.join(d, f))
    return sorted({c for c in cands if _readable_nonempty(c)})


def checkout_db_sockets(repo=None) -> list:
    """INT3 MED-B: Postgres sockets (S_ISSOCK, .s.PGSQL.*) anywhere in the checkout (WS-8g GC1/GC2: a socket in an
    unlisted directory was reachable by native libpq). lstat only; never opens or connects."""
    import stat as _st
    out = []
    for d, dirs, files in os.walk(str(Path(repo or REPO))):
        dirs[:] = [x for x in dirs if x != ".git"]
        for f in files:
            if f.startswith(DB_SOCKET_PREFIX) and not f.endswith(".lock"):
                pth = os.path.join(d, f)
                try:
                    if _st.S_ISSOCK(os.lstat(pth).st_mode):
                        out.append(pth)
                except OSError:
                    continue
    return sorted(out)


def session_isolation_problems(uid=None, user=None, live=False) -> list:
    """INT3 MED-A: [] when the session is verifiably isolated already; else the names of the failed checks, in order:
    'network', 'db-socket', 'escape-socket', 'credential-file'. Reads /proc and lstat/access only."""
    uid = os.getuid() if uid is None else uid
    user = (os.environ.get("USER", "") or _ORIG_ENV.get("USER", "")) if user is None else user
    if netns_status() != "isolated":
        return ["network"]
    out = []
    if db_socket_visible() or checkout_db_sockets():
        out.append("db-socket")
    if escape_capable(*escape_lists(uid, user)):
        out.append("escape-socket")
    if not live and credential_files_visible():
        out.append("credential-file")
    return out


# Masks the known escape sockets inside the new mount namespace (positional data only, nothing interpolated), then
# drops to the real uid/gid in a nested user namespace. Any failure stops the chain: the run does not start.
SCRIPT = (
    'fail() { echo "' + NETNS_MESSAGE + '" >&2; exit ' + str(EXIT_ISOLATION) + '; }\n'
    'ip link set lo up || fail\n'
    'uid=$1; gid=$2; shift 2\n'
    'mode=files\n'
    'while [ "$#" -gt 0 ]; do\n'
    '  a=$1; shift\n'
    '  if [ "$a" = "--dirs" ]; then mode=dirs; continue; fi\n'
    '  if [ "$a" = "--cmd" ]; then break; fi\n'
    '  if [ -e "$a" ]; then\n'
    '    if [ "$mode" = files ]; then mount --bind /dev/null "$a" || fail; else mount -t tmpfs -o mode=0700 tmpfs "$a" || fail; fi\n'
    '  fi\n'
    'done\n'
    'unset WSL_INTEROP WSL_DISTRO_NAME WSLENV\n'
    'exec unshare --user --map-user="$uid" --map-group="$gid" -- "$@" || fail\n'
)


# The outer shell checks that unshare itself works (so a failing unshare exits 78, not 1), then execs it.
OUTER = ('unshare -rnm true >/dev/null 2>&1 || { echo "' + NETNS_MESSAGE + '" >&2; exit ' + str(EXIT_ISOLATION) + '; }\n'
         'exec unshare "$@"\n')


def _proc_net(name: str) -> str:
    return (Path("/proc/self/net") / name).read_text()


def _udp_unreachable() -> bool:
    """A UDP connect sends no packet; with no route it fails at once with ENETUNREACH (raw socket, no wrapper)."""
    import _socket
    import errno
    s = _socket.socket(_socket.AF_INET, _socket.SOCK_DGRAM)
    _INTERNAL.on = True                     # the boundary's own probe: not a test connect (INT2 audit hook)
    try:
        s.connect(("192.0.2.1", 9))
    except OSError as exc:
        return exc.errno == errno.ENETUNREACH
    finally:
        _INTERNAL.on = False
        s.close()
    return False


def netns_status(read=_proc_net, udp=_udp_unreachable) -> str:
    """'isolated' | 'not_isolated' | 'unknown'. Reads only /proc/self/net/*; sends no packet."""
    try:
        dev, route, r6 = read("dev"), read("route"), read("ipv6_route")
    except OSError:
        return "unknown"
    names = {ln.split(":", 1)[0].strip() for ln in dev.splitlines()[2:] if ":" in ln}
    if names != {"lo"}:
        return "not_isolated"
    if [ln for ln in route.splitlines()[1:] if ln.strip()]:
        return "not_isolated"
    if any(ln.split() and ln.split()[-1] != "lo" for ln in r6.splitlines()):
        return "not_isolated"
    try:
        return "isolated" if udp() else "not_isolated"
    except OSError:
        return "not_isolated"


def db_socket_files(dirs=None) -> list:
    """Postgres socket paths (S_ISSOCK, name .s.PGSQL.*) in `dirs`. lstat only; never opens or connects."""
    import stat as _st
    dirs = DB_SOCKET_SCAN_DIRS if dirs is None else dirs
    out = []
    for d in dirs:
        try:
            names = sorted(os.listdir(d))
        except OSError:
            continue
        for n in names:
            if n.startswith(DB_SOCKET_PREFIX) and not n.endswith(".lock"):
                pth = os.path.join(d, n)
                try:
                    if _st.S_ISSOCK(os.lstat(pth).st_mode):
                        out.append(pth)
                except OSError:
                    continue
    return out


def db_socket_visible(dirs=None) -> bool:
    return bool(db_socket_files(dirs))


def launch_sanctioned(env) -> bool:
    """A child pytest started by a test of a sanctioned session (spawn guard: MARKER + reused bootstrap dir,
    STATE["is_child"]) runs inside that session's namespaces and counts as sanctioned too."""
    return (env.get(RUNNER_MARKER) == "1" or env.get(REEXEC_MARKER) == "1" or env.get(OFFLINE_ESCAPE) == "1"
            or bool(re.fullmatch(r"[0-9a-f]{32}", env.get("VERA_GATE_NONCE") or ""))
            or (env.get(MARKER) == "1" and bool(STATE.get("is_child")) and bool(STATE.get("bootstrap_reused"))))


FOREIGN_FSTYPES = ("9p", "drvfs", "virtiofs", "fuse.drvfs")


def foreign_mounts(read=lambda: Path("/proc/self/mountinfo").read_text(), visible_only: bool = False) -> list:
    """INT2 (WSL): host-shared mountpoints (Windows drives, 9p shares); they hold Windows executables that the WSL
    interop binfmt handler would run OUTSIDE the network namespace. Read-only parse; never executes anything.
    visible_only: drop an entry that a LATER mount at the same path or at an ancestor path covers (masked)."""
    try:
        text = read()
    except OSError:
        return []
    rows = []
    for line in text.splitlines():
        f = line.split()
        if "-" in f[6:]:
            i = f.index("-", 6)
            mp = re.sub(r"\\([0-7]{3})", lambda m: chr(int(m.group(1), 8)), f[4])
            rows.append((mp, len(f) > i + 1 and f[i + 1] in FOREIGN_FSTYPES))
    out = []
    for n, (mp, foreign) in enumerate(rows):
        if not foreign:
            continue
        if visible_only and any(later == mp or mp.startswith(later.rstrip("/") + "/") for later, _f in rows[n + 1:]):
            continue
        out.append(mp)
    return sorted(set(out))


def escape_lists(uid: int, user: str) -> tuple:
    sub = lambda t: t.replace("{uid}", str(uid)).replace("{user}", user or "nouser")   # noqa: E731
    dirs = [sub(d) for d in ESCAPE_SOCKET_DIRS]
    # INT2 (WSL): the netns re-exec also hides /mnt (Windows drives) and every other host-shared mount that does not
    # contain this checkout (a checkout on a 9p mount stays visible; scripts/vera_test_runner.py refuses that case)
    for m in ["/mnt"] + [m for m in foreign_mounts() if not m.startswith("/mnt/")]:
        if m not in dirs and not _under(REPO, Path(m)):
            dirs.append(m)
    loose = db_socket_files(tuple(d for d in DB_SOCKET_SCAN_DIRS if d not in ESCAPE_SOCKET_DIRS))
    return [sub(f) for f in (*ESCAPE_SOCKET_FILES, *ESCAPE_DEVICE_FILES)] + loose, dirs


def vsock_reachable(path: str = "/dev/vsock", null: str = "/dev/null") -> bool:
    """True if the vsock device node is present and NOT masked by /dev/null (OD-S6 C). Never opens or connects."""
    try:
        st = os.stat(path)
    except OSError:
        return False
    try:
        return st.st_rdev != os.stat(null).st_rdev
    except OSError:
        return True


def wsl_interop_open(binfmt: str = WSL_INTEROP_BINFMT, interop_dir: str = WSL_INTEROP_DIR, environ=None) -> bool:
    """S-b-F3: True if WSL interop is enabled AND still reachable (sockets in /run/WSL, or WSL_INTEROP in the env).
    Interop lets a process start Windows programs, which have the host's full network."""
    try:
        with open(binfmt) as f:
            enabled = f.readline().strip() == "enabled"
    except OSError:
        return False
    if not enabled:
        return False
    env = os.environ if environ is None else environ
    if env.get("WSL_INTEROP"):
        return True
    try:
        return bool(os.listdir(interop_dir))
    except OSError:
        return False


def escape_capable(files, dirs, binfmt: str = WSL_INTEROP_BINFMT, interop_dir: str = WSL_INTEROP_DIR) -> bool:
    """True if any listed control socket could be connected to (a socket we can write to), WSL interop is open, or the
    vsock device is unmasked. Never connects."""
    import stat as _st
    if wsl_interop_open(binfmt, interop_dir) or vsock_reachable():
        return True
    paths = list(files)
    for d in dirs:
        try:
            names = os.listdir(d)
        except OSError:
            continue
        paths += [os.path.join(d, n) for n in names
                  if n.endswith(".sock") or n.endswith("_interop") or n == "bus" or "tmux" in d or d.rstrip("/").endswith("/run/WSL")]
    for path in paths:
        try:
            st = os.stat(path)
        except OSError:
            continue
        if _st.S_ISSOCK(st.st_mode) and os.access(path, os.W_OK):
            return True
    return False


def reexec_argv(orig_argv, uid: int, gid: int, user: str = "") -> list:
    files, dirs = escape_lists(uid, user)
    inner = ["-rnm", "sh", "-c", SCRIPT, "sh", str(uid), str(gid), *files, "--dirs", *dirs, "--cmd", *orig_argv]
    return ["sh", "-c", OUTER, "sh", *inner]


def _stop_capture(config) -> None:
    try:
        config.pluginmanager.getplugin("capturemanager").stop_global_capturing()
    except Exception:  # noqa: BLE001
        pass
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except Exception:  # noqa: BLE001
            pass


def _isolation_fail(config, text: str):
    _stop_capture(config)
    os.write(2, (text + "\n").encode())
    os._exit(EXIT_ISOLATION)


# --- S-b-F5: kernel-enforced refusal of native AF_VSOCK and io_uring (seccomp, ctypes only) ----------------------
SECCOMP_MESSAGE = "test isolation: cannot install the socket-family filter (seccomp)"
AF_VSOCK_NR = 40
SECCOMP_RET_ALLOW, SECCOMP_RET_KILL_PROCESS = 0x7FFF0000, 0x80000000
_ARCHES = {   # machine -> (AUDIT_ARCH, socket nr, io_uring_setup nr, seccomp(2) nr, rejects x32-style nr bit)
    "x86_64": (0xC000003E, 41, 425, 317, True),
    "aarch64": (0xC00000B7, 198, 425, 277, False),
}
X32_SYSCALL_BIT = 0x40000000


def _ret_errno(errno_value: int) -> int:
    return 0x00050000 | errno_value


def build_socket_filter(machine: str) -> list:
    """Classic BPF [(code, jt, jf, k), ...]. Allows everything except socket(AF_VSOCK) -> EACCES and io_uring_setup ->
    EPERM (io_uring has IORING_OP_SOCKET, which would bypass the socket() rule). Any other architecture/ABI (32-bit
    compat via int 0x80, and on x86_64 the x32 ABI whose numbers carry bit 30) is killed, so no alternative ABI can
    reach socket(2). Raises KeyError for an unsupported machine."""
    audit_arch, nr_socket, nr_uring, _nr_seccomp, reject_x32 = _ARCHES[machine]
    LD_W_ABS, JEQ_K, JGE_K, RET_K = 0x20, 0x15, 0x35, 0x06
    ins = [("LD", LD_W_ABS, 4), ("JEQ", JEQ_K, audit_arch), ("LD", LD_W_ABS, 0)]
    if reject_x32:
        ins.append(("JGE", JGE_K, X32_SYSCALL_BIT))
    ins += [("JEQ", JEQ_K, nr_uring), ("JEQ", JEQ_K, nr_socket), ("LD", LD_W_ABS, 16), ("JEQ", JEQ_K, AF_VSOCK_NR),
            ("RET", RET_K, SECCOMP_RET_ALLOW), ("RET", RET_K, _ret_errno(1)), ("RET", RET_K, _ret_errno(13)),
            ("RET", RET_K, SECCOMP_RET_KILL_PROCESS)]
    n = len(ins)
    idx = {"allow": n - 4, "eperm": n - 3, "eacces": n - 2, "kill": n - 1}
    out = []
    for i, (kind, code, k) in enumerate(ins):
        jt = jf = 0
        if kind == "JEQ" and k == audit_arch:
            jf = idx["kill"] - i - 1                    # wrong architecture: kill
        elif kind == "JGE":
            jt = idx["kill"] - i - 1                    # x32 ABI: kill
        elif kind == "JEQ" and k == nr_uring:
            jt, jf = idx["eperm"] - i - 1, 0
        elif kind == "JEQ" and k == nr_socket:
            jt, jf = 0, idx["allow"] - i - 1            # socket: look at args[0]; any other syscall: allow
        elif kind == "JEQ" and k == AF_VSOCK_NR:
            jt, jf = idx["eacces"] - i - 1, idx["allow"] - i - 1
        out.append((code, jt, jf, k))
    return out


def run_socket_filter(prog, arch: int, nr: int, arg0: int = 0) -> int:
    """Tiny classic-BPF interpreter (tests only): the verdict the kernel would give for one syscall."""
    mem = {0: nr, 4: arch, 16: arg0 & 0xFFFFFFFF}
    a, pc = 0, 0
    while True:
        code, jt, jf, k = prog[pc]
        if code == 0x20:
            a = mem.get(k, 0)
        elif code == 0x15:
            pc += jt if a == k else jf
        elif code == 0x35:
            pc += jt if a >= k else jf
        elif code == 0x06:
            return k
        pc += 1


def install_socket_filter() -> None:
    """Install the seccomp filter on the whole process (all threads, inherited by every child). Raises OSError or
    KeyError on failure; the caller turns any failure into a fail-closed exit 78."""
    import ctypes
    import platform
    if STATE.get("seccomp") == "installed":
        return
    machine = platform.machine()
    prog = build_socket_filter(machine)
    nr_seccomp = _ARCHES[machine][3]

    class SockFilter(ctypes.Structure):
        _fields_ = [("code", ctypes.c_ushort), ("jt", ctypes.c_ubyte), ("jf", ctypes.c_ubyte), ("k", ctypes.c_uint32)]

    class SockFprog(ctypes.Structure):
        _fields_ = [("len", ctypes.c_ushort), ("filter", ctypes.POINTER(SockFilter))]

    arr = (SockFilter * len(prog))(*[SockFilter(*i) for i in prog])
    fprog = SockFprog(len(prog), ctypes.cast(arr, ctypes.POINTER(SockFilter)))
    libc = ctypes.CDLL(None, use_errno=True)
    _INTERNAL.on = True                     # the boundary's own libc lookups (prctl, syscall): INT2 audit hook exempt
    try:
        prctl, sys_call = libc.prctl, libc.syscall
    finally:
        _INTERNAL.on = False
    if prctl(38, 1, 0, 0, 0) != 0:                                         # PR_SET_NO_NEW_PRIVS
        raise OSError(ctypes.get_errno(), "prctl(PR_SET_NO_NEW_PRIVS)")
    if sys_call(nr_seccomp, 1, 1, ctypes.byref(fprog)) != 0:                # SET_MODE_FILTER, FILTER_FLAG_TSYNC
        raise OSError(ctypes.get_errno(), "seccomp")
    STATE["seccomp"] = "installed"


def _install_filter_or_fail(config) -> None:
    try:
        install_socket_filter()
    except Exception:  # noqa: BLE001  (any failure, including an unsupported architecture, is fail closed)
        _isolation_fail(config, SECCOMP_MESSAGE)


def reexec_env(orig_env) -> dict:
    """K18W M-2/L-4 (credential inheritance): the environment for the netns re-exec. No WSL_* names (interop) and no
    guarded name (DB URLs/settings, passwords, provider keys, proxies): an isolated session never needs them (live
    sessions do not re-exec), so they must not cross into the namespace where test code runs."""
    return {k: v for k, v in dict(orig_env).items() if not k.upper().startswith("WSL") and not guarded(k)}


def maybe_reexec(config, live: bool) -> None:
    """Called once from a tryfirst pytest_configure (before any test module is imported). Makes the kernel enforce
    'no non-loopback network': the session must ALREADY be isolated (runner, or an offline-only container), else it
    fails closed with exit 78. INT3 MED-A: there is no re-exec fallback any more; no signal overrides the checks."""
    if STATE["netns_decided"]:
        return
    STATE["netns_decided"] = True
    STATE["config"] = config
    uid, gid, user = os.getuid(), os.getgid(), os.environ.get("USER", "") or _ORIG_ENV.get("USER", "")
    if not live and not launch_sanctioned(_ORIG_ENV):          # INT2 F-1/F-4: runner (or gate, or explicit escape)
        _isolation_fail(config, RUNNER_MESSAGE)
    if netns_status() == "isolated":
        STATE["netns"] = "isolated"
        if db_socket_visible() or checkout_db_sockets():       # INT2 F-1 / INT3 MED-B: kernel fact, whatever launched
            _isolation_fail(config, DB_SOCKET_MESSAGE)
        if escape_capable(*escape_lists(uid, user)):
            _isolation_fail(config, ESCAPE_MESSAGE)
        if not live and credential_files_visible():            # INT3 MED-A: "isolated" includes no readable credential
            _isolation_fail(config, CREDENTIAL_MESSAGE)
        STATE["unix_check"] = "clean"
        _install_filter_or_fail(config)
        return
    if live:                                   # live sessions run live-marked items only (see tests/conftest.py)
        STATE["netns"], STATE["unix_check"] = "live-session", "skipped-live"
        _install_filter_or_fail(config)
        return
    # INT3 MED-A: a sanctioned signal in a session that is NOT already isolated is refused. Never re-exec into the
    # in-repo chain (it hid neither HOME nor checkout credential files).
    STATE["netns"] = "refused-not-isolated"
    _isolation_fail(config, ISOLATION_REQUIRED_MESSAGE)


def install() -> dict:
    """Idempotent. Raises RuntimeError if an application module is already imported."""
    if STATE["installed"]:
        return STATE
    if not _ORIG_ENV:
        _ORIG_ENV.update(os.environ)             # first statement: before the stash and any PYTHONPATH injection
    loaded = sorted(n for n in sys.modules if n in _APP_MODULES or n == "vera" or n.startswith("vera."))
    STATE["app_modules_at_install"] = loaded
    if loaded:
        raise RuntimeError("test boundary installed after application import")
    # (i) deny fingerprints from the outer environment and R/config/env (E2), password-dropped, hashes only
    sources = [_reduce(os.environ)] + _read_env_files()
    deny = _build_deny_sets(sources)
    STATE["deny_fp"], STATE["deny_user_fp"] = deny["fp"], deny["user"]
    STATE["deny_db_fp"], STATE["deny_suffix"], STATE["deny_ip_fp"] = deny["db"], deny["suffix"], deny["ip"]
    sources.clear()
    _SEALED["deny_policy"], _SEALED["deny_all"] = STATE["deny_policy"], bool(STATE["deny_all"])
    # (ii) stash, always; (ii-a) child bootstrap (sitecustomize) and (ii-b) spawn guard
    _setup_child_bootstrap()
    stash_guarded_env(keep_provider=STATE["is_child"])
    _install_spawn_guard()
    _wrap_mp_spawnv_passfds(sys.modules["multiprocessing.util"])      # INT2 F-5 layer
    if STATE["is_child"]:
        atexit.register(_child_exit_hook)
    # (iii) dotenv no-op, always
    _install_dotenv_noop()
    # (iv) E2: the boundary's do_connect listener, then the shared connect guard (context "test", always
    # VERIFY_DISABLED), then the boundary's driver wrappers: the boundary listener runs first and its wrappers are
    # outermost, so the boundary refuses and records before the guard is reached (the guard is defence in depth)
    _install_db_listener()
    install_connect_guard()
    _install_db_boundary()
    # (v) network boundary (S-b)
    _install_network_boundary()
    # (vi) INT2 LOW F-6: the snapshot keeps no credential values; the stash keeps only provider keys (the one thing a
    # live item gets back). Deny fingerprints were already built above from the full environment.
    drop_credentials_from_snapshot()
    STATE["installed"] = True
    _install_audit_hook()                   # INT2 F-3: after installed=True so the hook is active from here on
    return STATE


def drop_credentials_from_snapshot() -> list:
    """Names (never values) removed from _ORIG_ENV and _STASH. /proc/self/environ of THIS process still holds the
    launch environment: only a clean launch (scripts/vera_test_runner.py env -i, or the netns re-exec, whose env is
    reexec_env()) removes it there."""
    gone = sorted(k for k in _ORIG_ENV if guarded(k))
    for k in gone:
        del _ORIG_ENV[k]
    for k in [k for k in _STASH if not PROVIDER_KEY.search(k)]:
        del _STASH[k]
        gone.append(k)
    return sorted(set(gone))


_SEAL_KEYS = ("installed", "netns", "seccomp", "live_enabled")


DB_WRAP_SITES = (("psycopg2", "connect"), ("psycopg", "connect"), ("pg8000", "connect"), ("pg8000.dbapi", "connect"),
                 ("asyncpg", "connect"), ("asyncpg", "create_pool")) + RAW_DB_SITES


def _db_listener_present():
    try:
        from sqlalchemy import event
        from sqlalchemy.engine import Engine
        return STATE["db_listener"] is not None and event.contains(Engine, "do_connect", STATE["db_listener"])
    except Exception:  # noqa: BLE001
        return False


def _wrapper_ids() -> dict:
    ids = {"getaddrinfo": id(socket.getaddrinfo), "connect": id(socket.socket.connect),
           "connect_ex": id(socket.socket.connect_ex), "create_connection": id(socket.create_connection),
           "sendto": id(socket.socket.sendto), "sendmsg": id(socket.socket.sendmsg),
           "socket_init": id(socket.socket.__init__), "popen_init": id(subprocess.Popen.__init__)}
    # INT2 F-2 (verifier P1-P3): the DB driver wrappers, the SQLAlchemy listener, the os/multiprocessing spawn wrappers
    # and the dotenv no-ops are isolation state too. A module in sys.modules is compared by its CURRENT attribute, so a
    # raw unwrap or importlib.reload(psycopg2) changes the identity and is reported.
    for modname, attr in DB_WRAP_SITES:
        mod = sys.modules.get(modname)
        if mod is not None and hasattr(mod, attr):
            ids[f"{modname}.{attr}"] = id(getattr(mod, attr))
    for name, _a, _e in SPAWN_SITES:
        if hasattr(os, name):
            ids["os." + name] = id(getattr(os, name))
    mpu = sys.modules.get("multiprocessing.util")
    if mpu is not None:
        ids["multiprocessing.util.spawnv_passfds"] = id(getattr(mpu, "spawnv_passfds", None))
    for modname in ("dotenv", "dotenv.main"):
        mod = sys.modules.get(modname)
        if mod is not None:
            ids[modname + ".load_dotenv"] = id(getattr(mod, "load_dotenv", None))
            ids[modname + ".dotenv_values"] = id(getattr(mod, "dotenv_values", None))
    ids["sqlalchemy.do_connect_listener"] = _db_listener_present()
    return ids


def reinstall_wrappers() -> None:
    """INT2 F-2 fail closed: put back every wrapper/listener a test removed (each install step is idempotent)."""
    _install_spawn_guard()
    _wrap_mp_spawnv_passfds(sys.modules["multiprocessing.util"])
    import dotenv
    import dotenv.main
    if STATE["dotenv_noop"] and not (dotenv.load_dotenv is dotenv.main.load_dotenv is _noop_load_dotenv
                                     and dotenv.dotenv_values is dotenv.main.dotenv_values is _noop_dotenv_values):
        for mod in (dotenv, dotenv.main):          # never re-read the originals into _ORIG (they would be the no-ops)
            mod.load_dotenv, mod.dotenv_values = _noop_load_dotenv, _noop_dotenv_values
    _install_db_boundary()
    _install_network_boundary()


def seal_session() -> None:
    """K18W M-2: called from pytest_sessionstart (after the netns gate). Records the session's isolation facts."""
    _SEALED["session"] = {k: STATE[k] for k in _SEAL_KEYS}
    _SEALED["wrappers"] = _wrapper_ids()
    _SEALED["reloads"] = STATE.get("reloads", 0)


def tamper_keys() -> list:
    """Names (never values) of isolation facts that differ from the sealed session state. Called after each item's
    teardown, when monkeypatch has already undone its changes, so only writes that PERSIST past an item are reported."""
    bad = []
    if "deny_all" in _SEALED:
        if bool(STATE["deny_all"]) != _SEALED["deny_all"]:
            bad.append("deny_all")
        if STATE["deny_policy"] != _SEALED["deny_policy"]:
            bad.append("deny_policy")
    for k, v in (_SEALED.get("session") or {}).items():
        if STATE.get(k) != v:
            bad.append(k)
    now = _wrapper_ids()
    for k, v in (_SEALED.get("wrappers") or {}).items():
        if now.get(k) != v:
            bad.append("wrapper:" + k)
    if "session" in _SEALED and STATE.get("reloads", 0) != _SEALED.get("reloads", 0):
        bad.append("boundary_reloaded")
    return bad


def tamper_message() -> str | None:
    bad = tamper_keys()
    if not bad:
        return None
    STATE["tamper_count"] += 1
    if "deny_all" in _SEALED:             # fail closed: put the sealed deny decision back for the rest of the session
        STATE["deny_all"], STATE["deny_policy"] = _SEALED["deny_all"], _SEALED["deny_policy"]
    if any(b.startswith("wrapper:") for b in bad) and "wrappers" in _SEALED:   # INT2 F-2: fail closed, re-wrap
        reinstall_wrappers()
        _SEALED["wrappers"] = _wrapper_ids()
    if "boundary_reloaded" in bad:
        _SEALED["reloads"] = STATE.get("reloads", 0)
    return ("test boundary tamper: isolation state changed and was not restored after the test (" + ", ".join(bad)
            + "); use monkeypatch so the change is undone, the session's isolation evidence is withheld")


def load_by_path():
    """Used by the conftests: load this file by path (no package import) and register it under MODULE_NAME."""
    mod = sys.modules.get(MODULE_NAME)
    if mod is None:
        spec = importlib.util.spec_from_file_location(MODULE_NAME, Path(__file__).resolve())
        mod = importlib.util.module_from_spec(spec)
        sys.modules[MODULE_NAME] = mod
        spec.loader.exec_module(mod)
    return mod

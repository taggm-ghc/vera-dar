"""Process-global connect guard (PLAN-ENV v2.1 §3 E0/C5, V1, V2(b), L1; guard file).

`install(context)` registers a class-level SQLAlchemy `do_connect` hook (inserted first), records each engine's raw URL
and connect_args by wrapping `create_engine`, wraps `psycopg2.connect` / `psycopg.connect`, and installs the SQL guard.
On every new DBAPI connection (pool reconnects included, no cache):
  1. ambient PG*/KRB5*/GSS* names refuse (`pg_env`), in every context;
  2. strict C4 parse; a failure refuses with its code, in every context (V2(b));
  3. the connect kwargs are rebuilt: endpoint fields only from the parse, code-supplied args only through the V1
     allowlist, then passfile=/dev/null, sslcertmode=disable, gssencmode=disable;
  4. guarded contexts (test, calibration, migration, check) ALWAYS refuse with VERIFY_DISABLED before the driver is
     called (binding owner decision 2026-10-19; no flag enables verification);
  5. app contexts (production, local_dev, agent) connect with the rebuilt kwargs and tag the connection with its
     PRODUCTION Verdict, so the SQL guard (C7) applies.
ConnectRefused carries a code only.
"""
from __future__ import annotations

import importlib.util
import os
import re
import sys
import threading
import types
import weakref
from pathlib import Path
from urllib.parse import quote


def _sibling(name: str):
    try:
        return __import__(f"safety.{name}", fromlist=["x"])
    except ImportError:
        priv = f"_vera_safety_{name}"
        if priv in sys.modules:
            return sys.modules[priv]
        spec = importlib.util.spec_from_file_location(priv, Path(__file__).with_name(f"{name}.py"))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[priv] = mod
        spec.loader.exec_module(mod)
        return mod


_ec = _sibling("endpoint_class")
sql_guard = _sibling("sql_guard")

APP_CONTEXTS = ("production", "local_dev", "agent")
GUARDED_CONTEXTS = ("test", "calibration", "migration", "check")
CONTEXTS = APP_CONTEXTS + GUARDED_CONTEXTS
VERDICT_KEY = sql_guard.VERDICT_KEY

ENDPOINT_KEYS = ("host", "hostaddr", "port", "dbname", "database", "user", "username", "password", "passfile",
                 "service", "servicefile")
ALLOWED_CODE_KEYS = ("connect_timeout", "application_name", "sslmode", "options")
PASSTHROUGH_KEYS = ("cursor_factory", "row_factory", "autocommit")   # driver-side objects, not connection params
OPTIONS_RE = re.compile(r"(-c (statement_timeout|lock_timeout|idle_in_transaction_session_timeout|"
                        r"default_transaction_read_only)=[A-Za-z0-9_]+ ?)+")
_APPNAME_RE = re.compile(r"[A-Za-z0-9_.-]{1,63}")


class ConnectRefused(Exception):
    """A refused connection. Carries a code only."""

    def __init__(self, code: str):
        super().__init__(code)
        self.code = code

    def __str__(self) -> str:
        return f"connect refused: {self.code}"

    def __repr__(self) -> str:
        return f"ConnectRefused({self.code!r})"


_SHARED = sys.modules.setdefault("_vera_safety_shared_state", types.ModuleType("_vera_safety_shared_state"))
ConnectRefused = _SHARED.__dict__.setdefault("ConnectRefused", ConnectRefused)


def rebuild_kwargs(parsed, code_args=None) -> dict:
    """V1: endpoint fields only from the parse; code-supplied args merged through the allowlist."""
    kw = {"host": parsed.host, "port": parsed.port, "dbname": parsed.dbname, "user": parsed.user}
    if parsed.password:
        kw["password"] = parsed.password
    kw.update(parsed.params)
    base_ssl = kw.get("sslmode", "prefer")
    for k, v in dict(code_args or {}).items():
        if k not in ALLOWED_CODE_KEYS:
            raise ConnectRefused("bad_param")
        if k == "connect_timeout":
            if isinstance(v, bool) or not (isinstance(v, int) or (isinstance(v, str) and v.isdigit())):
                raise ConnectRefused("bad_param")
            v = int(v)
        elif k == "application_name":
            if not isinstance(v, str) or _APPNAME_RE.fullmatch(v) is None:
                raise ConnectRefused("bad_param")
        elif k == "sslmode":
            if v not in _ec.SSLMODES or _ec.SSLMODES.index(v) < _ec.SSLMODES.index(base_ssl):
                raise ConnectRefused("bad_param")
        elif k == "options":
            if not isinstance(v, str) or OPTIONS_RE.fullmatch(v) is None:
                raise ConnectRefused("bad_param")
        kw[k] = v
    kw["passfile"] = os.devnull
    kw["sslcertmode"] = "disable"
    kw["gssencmode"] = "disable"
    return kw


class Guard:
    """One guard state. The process-global instance is shared by every copy of this module (package import and
    by-path loads under private names) through a synthetic registry module in sys.modules. Tests may build private
    instances and install them on fake driver modules."""

    def __init__(self):
        self.context = None
        self.wrapped: list = []
        self.resolver = None
        self.dialect_info = weakref.WeakKeyDictionary()
        self.local = threading.local()
        self.lock = threading.RLock()
        self.listener = self.on_do_connect          # one stable bound-method object for event.contains/remove

    def __repr__(self) -> str:
        return f"Guard(context={self.context!r})"

    # -- decision ---------------------------------------------------------------------------------------------
    def guard_url(self, url, code_args=None, *, context=None):
        """Returns (rebuilt kwargs, Verdict) in app contexts; raises ConnectRefused otherwise."""
        ctx = context or self.context
        code = None
        if ctx not in CONTEXTS:
            code = "bad_context"
        elif not url or not isinstance(url, str):
            code = "no_url"
        elif _ec.ambient_names(os.environ):
            code = "pg_env"
        parsed = None
        if code is None:
            try:
                parsed = _ec.parse(url)
            except _ec.ParseError as e:
                code = e.code
        if code is not None:
            raise ConnectRefused(code)
        kwargs = rebuild_kwargs(parsed, code_args)
        verdict = _ec.classify(url, environ=os.environ, resolver=self.resolver)
        if ctx in GUARDED_CONTEXTS:
            # Owner decision 2026-10-19: verification refuses unconditionally; the driver is never called.
            _ec.verify_disposable(None, url)
            raise ConnectRefused(_ec.VERIFY_DISABLED)
        return kwargs, verdict

    def _enter(self):
        self.local.depth = getattr(self.local, "depth", 0) + 1

    def _leave(self):
        self.local.depth -= 1

    def _active(self) -> bool:
        return getattr(self.local, "depth", 0) > 0

    # -- SQLAlchemy -------------------------------------------------------------------------------------------
    def on_do_connect(self, dialect, conn_rec, cargs, cparams):
        if getattr(dialect, "name", "") == "sqlite":
            return None
        info = self.dialect_info.get(dialect)
        if info is not None:
            url, code_args = info
        else:
            url, code_args = _url_from_cparams(cargs, cparams)
        kwargs, verdict = self.guard_url(url, code_args)
        conn_rec.info[VERDICT_KEY] = verdict
        self._enter()
        try:
            return dialect.connect(**kwargs)
        finally:
            self._leave()

    def wrap_create_engine(self, module, attr):
        orig = getattr(module, attr, None)
        if orig is None or getattr(orig, "_vera_guard_owner", None) is self:
            return
        guard = self

        def create_engine(url, *a, **kw):
            eng = orig(url, *a, **kw)
            try:
                raw = url if isinstance(url, str) else url.render_as_string(hide_password=False)
                guard.dialect_info[eng.dialect] = (raw, dict(kw.get("connect_args") or {}))
            except Exception:
                pass                    # untagged engines use the cparams fallback
            return eng

        create_engine._vera_guard = True
        create_engine._vera_guard_owner = self
        create_engine._vera_original = orig
        setattr(module, attr, create_engine)
        self.wrapped.append((module, attr, orig))

    # -- raw drivers ------------------------------------------------------------------------------------------
    def wrap_driver(self, module, attr):
        orig = getattr(module, attr, None)
        if orig is None or getattr(orig, "_vera_guard_owner", None) is self:
            return
        guard = self

        def connect(*a, **kw):
            if guard._active():         # L1: this guard already decided this connect
                return orig(*a, **kw)
            if len(a) > 1:
                raise ConnectRefused("bad_param")
            dsn = a[0] if a else kw.pop("dsn", kw.pop("conninfo", None))
            passthrough = {k: kw.pop(k) for k in PASSTHROUGH_KEYS if k in kw}
            kwargs, verdict = guard.guard_url(dsn, kw)
            guard._enter()
            try:
                raw = orig(**kwargs, **passthrough)
            finally:
                guard._leave()
            return sql_guard.wrap_connection(raw, verdict)

        connect._vera_guard = True
        connect._vera_guard_owner = self
        connect._vera_original = orig
        setattr(module, attr, connect)
        self.wrapped.append((module, attr, orig))

    # -- lifecycle --------------------------------------------------------------------------------------------
    def install(self, context: str, *, sqlalchemy: bool = True, drivers=None) -> None:
        if context not in CONTEXTS:
            raise ConnectRefused("bad_context")
        with self.lock:
            cur = self.context
            if cur is not None and cur != context and not (cur in APP_CONTEXTS and context in GUARDED_CONTEXTS):
                raise ConnectRefused("context_conflict")
            self.context = context
            if sqlalchemy and importlib.util.find_spec("sqlalchemy") is not None:
                from sqlalchemy import event
                from sqlalchemy.engine import Engine
                if not event.contains(Engine, "do_connect", self.listener):
                    event.listen(Engine, "do_connect", self.listener, insert=True)
                import sqlalchemy as _sa
                import sqlalchemy.engine
                import sqlalchemy.engine.create
                for mod in (_sa, sqlalchemy.engine, sqlalchemy.engine.create):
                    self.wrap_create_engine(mod, "create_engine")
                sql_guard.install_listener()
            if drivers is None:
                drivers = [(__import__(m), "connect") for m in ("psycopg2", "psycopg")
                           if importlib.util.find_spec(m) is not None]
            for module, attr in drivers:
                self.wrap_driver(module, attr)

    def uninstall(self) -> None:
        """TEST HELPER ONLY (private instances): restores wrapped attributes and removes this guard's listener."""
        with self.lock:
            for module, attr, orig in reversed(self.wrapped):
                if getattr(getattr(module, attr, None), "_vera_guard_owner", None) is self:
                    setattr(module, attr, orig)
            self.wrapped.clear()
            if importlib.util.find_spec("sqlalchemy") is not None:
                from sqlalchemy import event
                from sqlalchemy.engine import Engine
                if event.contains(Engine, "do_connect", self.listener):
                    event.remove(Engine, "do_connect", self.listener)
            self.context = None
            self.resolver = None


_GLOBAL = _SHARED.__dict__.setdefault("connect_guard", Guard())


def install(context: str) -> None:
    """Process-global and idempotent. Re-install with the same context is a no-op; app -> guarded tightens; any
    other change raises ConnectRefused('context_conflict')."""
    _GLOBAL.install(context)


def installed_context():
    return _GLOBAL.context


def set_resolver(resolver) -> None:
    """Optional resolver(host)->iterable[ip] used to match PI on every connect (no cache). Default: none."""
    _GLOBAL.resolver = resolver


def guard_url(url, code_args=None, *, context=None):
    return _GLOBAL.guard_url(url, code_args, context=context)


def uninstall() -> None:
    """TEST HELPER ONLY: uninstalls the process-global guard. Never call it in a process whose boundary installed it."""
    _GLOBAL.uninstall()


def _url_from_cparams(cargs, cparams):
    """Fallback for an engine whose create_engine call predates install(): treat the endpoint keys SQLAlchemy built
    as the URL and every other key as code-supplied (limitation: a code-supplied host/port/dbname/user override is
    indistinguishable here; hostaddr, passfile, service and unknown keys still refuse)."""
    if cargs:
        raise ConnectRefused("bad_param")
    cp = dict(cparams)
    host = str(cp.pop("host", "") or "")
    port = cp.pop("port", None)
    db = str(cp.pop("dbname", cp.pop("database", "")) or "")
    user = str(cp.pop("user", cp.pop("username", "")) or "")
    pw = cp.pop("password", None)
    auth = user + (":" + quote(str(pw), safe="") if pw else "")
    url = f"postgresql://{auth}@{host}{':' + str(port) if port else ''}/{db}"
    return url, cp


def verdict_of(conn):
    """The Verdict tagged on a SQLAlchemy Connection or a guarded raw connection, else None."""
    try:
        v = object.__getattribute__(conn, "_vera_verdict")
        return v
    except AttributeError:
        pass
    try:
        return conn.info.get(VERDICT_KEY)
    except Exception:
        return None

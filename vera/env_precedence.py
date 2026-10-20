"""PRB-DARINT-002: deterministic .env* precedence. Replaces the bare load_dotenv() calls.

Decision tree (full text: kraken-DARINT/problems/PRB-DARINT-002-env-precedence.md; data: config/env_precedence.json):
  1. Context. A name counts as "present" when it is in the environment, whatever its value.
     First, a VERA_ENV_CONTEXT that is present but names no configured context (exact match) fails closed (F2).
     RENDER present: it must be exactly "true", which selects production; any other value fails closed.
     A test marker present, or the test boundary module loaded, selects test.
     An agent marker present selects agent. An explicit VERA_ENV_CONTEXT may only pick from agent_may_select; a
     file-loading pick (local_dev, OD-DARINT-AGENTDEV) is selected_by=agent_explicit: only agent_explicit.files
     load and agent_explicit.excluded_names (VERA_OWNER_KEY, every DB URL name) are skipped by name.
     Otherwise VERA_ENV_CONTEXT selects the context.
     Otherwise default_context. An explicit value that disagrees with RENDER or a test signal fails closed.
  2. Files. Only the paths the context lists are read, anchored at ENV_DIR (R/config/env, PLAN-ENV v2.1 1.2), in
     list order.
     There is no upward search, so ~/.env and every unlisted file are never opened.
     production, test and agent list no files.
  3. Precedence (PLAN-ENV v2.1 1.3, C3/OR-3; replaces "process env wins"). In a human local_dev context the
     inherited STRIP names (_INHERITED_STRIP) are removed first, then the file wins over the inherited process env.
     A name in both places with different values logs WARNING "env: file <rel> overrides inherited NAME" (name
     only). Between files the first listed wins. A name assigned twice in one file fails closed. Values are literal
     (no ${VAR} interpolation). Production, test and agent contexts load no file and are unchanged.
  4. Fail closed (EnvPrecedenceError; nothing is applied). Causes:
     - bad config, a context conflict or unknown context, or (file-loading contexts only) a parser version
       mismatch;
     - a listed file that is a symlink, hard-linked (st_nlink != 1), the user's ~/.env, not regular, outside the root, not owned by the euid,
       group/other-writable, too large, not UTF-8, BOM-prefixed or unparseable, or that has a duplicate
       or an invalid name;
     - a missing required file;
     - a restricted (DB URL) name in a file not allowed to supply it;
     - a control name (RENDER, VERA_ENV_CONTEXT, markers, PATH, ...), a TLS/proxy/libpq/Kerberos name
       (SSL_CERT_*, *_PROXY, PG*, KRB5*, GSS*) or an admin-floor name in any file (F7).
     Allowed restricted names load only when the context was selected explicitly. Otherwise they are
     skipped by name.
  4b. Agent process env (D-DARINT-036). When an agent marker is present, inherited VERA_OWNER_KEY and every DB
     credential name except the standing RO/WO capabilities (agent_process_env.keep) are removed from the process
     environment and recorded in audit.stripped by name. Human local_dev strips _INHERITED_STRIP instead (C3).
  5. Audit. Names and sources only. No value, value length or hash is ever logged, raised or returned.
"""
from __future__ import annotations

import fnmatch
import io
import json
import logging
import os
import re
import stat
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, MutableMapping

REPO_ROOT = Path(__file__).resolve().parent.parent
# Same value as vera.env_files.ENV_DIR (R/config/env); computed here to avoid a circular import.
ENV_DIR = REPO_ROOT.parent / "config" / "env"
CONFIG_PATH = REPO_ROOT / "config" / "env_precedence.json"
logger = logging.getLogger("vera.env_precedence")

_CONFIG_KEYS = {"schema_version", "_doc", "parser_version", "context_var", "render_signal", "test_markers",
                "test_modules", "test_context", "agent_markers", "agent_context", "agent_may_select", "agent_explicit",
                "agent_process_env",
                "default_context", "contexts", "precedence", "file_limits", "file_denied_names", "restricted_keys"}
_CONFLICT_MODES = ("first_wins", "error")
_DUP_MODES = ("error", "warn")            # "warn" = transition only: python-dotenv last-wins, reported by name
_NAME_RE = re.compile(r"\A[A-Za-z_][A-Za-z0-9_]*\Z")
# Code-level filename denylist (basename globs, case-insensitive): an edited config can never list these.
_AGENT_MUST_EXCLUDE = ("VERA_OWNER_KEY",)    # code floor: config can never hand an agent the owner key
_FLOOR_DB_PATTERNS = ("*DB_URL*", "DATABASE_URL", "PG*")   # code floor: restricted / excluded / stripped always
_AGENT_MARKER_FLOOR = ("CLAUDECODE", "AI_AGENT")         # code floor: agent detection (and G1) cannot be disabled
_AGENT_KEEP_MAX = ("VERA_DB_URL_RO", "VERA_DB_URL_WO")     # the only inherited DB names an agent may keep
_FORBIDDEN_FILE_GLOBS = ("-prv.env", "*-prv.env", "*-prv.env.*", ".env.render.*", "*.example", ".env.db-accounts",
                         ".env-intern")       # .env-intern: fingerprinted only, never loaded by VERA-DAR (D-ENV-4)
_PROXY_NAMES = ("HTTP_PROXY", "HTTPS_PROXY", "HTTP*_PROXY", "ALL_PROXY", "NO_PROXY")   # matched case-insensitively
_ADMIN_FLOOR = ("DB_ADMIN_*", "*_DBADMIN*", "VERA_DB_URL_ADMIN", "VERA_MIGRATE_PRODUCTION")
# Code floor (C3): a file may never set these, now that files win (TLS, proxies, libpq, Kerberos, admin floor).
_FILE_DENIED_FLOOR = ("SSL_CERT_*", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE") + _PROXY_NAMES + (
    "PG*", "KRB5*", "GSS*") + _ADMIN_FLOOR
# PLAN-ENV v2.1 1.3 STRIP set: removed from the inherited env in human local_dev contexts before files apply.
_INHERITED_STRIP = ("*DB_URL*", "DATABASE_URL", "PG*", "KRB5*", "GSS*", "*_DBADMIN*", "DB_ADMIN_*",
                    "VERA_MIGRATE_PRODUCTION", "SSL_CERT_*", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE") + _PROXY_NAMES
_PROCESS_AUDIT: list = []                     # once-per-process latch for load_env(os.environ)


class EnvPrecedenceError(RuntimeError):
    """Fail-closed refusal. The message carries names, repo-relative paths, line numbers and case codes only."""

    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code


@dataclass
class EnvAudit:
    context: str
    selected_by: str
    files: list = field(default_factory=list)       # [{"path": rel, "status": "loaded"|"missing"}]
    sources: dict = field(default_factory=dict)     # NAME -> "process" | rel path (winning source)
    shadowed: dict = field(default_factory=dict)    # NAME -> [{"source": rel, "by": rel}] (cross-file only)
    skipped: dict = field(default_factory=dict)     # NAME -> reason (restricted name, context not explicit)
    notes: dict = field(default_factory=dict)       # NAME -> ["interpolation_not_applied@rel", "no_value@rel", ...]
    applied: list = field(default_factory=list)     # names written into the environment by this call
    stripped: list = field(default_factory=list)    # inherited names removed (D-DARINT-036 agent; C3 local_dev)
    overridden: list = field(default_factory=list)  # inherited names a file replaced with a different value (C3)

    def as_dict(self) -> dict:
        return {"context": self.context, "selected_by": self.selected_by, "files": list(self.files),
                "sources": dict(sorted(self.sources.items())), "shadowed": dict(sorted(self.shadowed.items())),
                "skipped": dict(sorted(self.skipped.items())), "notes": dict(sorted(self.notes.items())),
                "applied": sorted(self.applied), "stripped": sorted(self.stripped),
                "overridden": sorted(self.overridden)}


# --- config ----------------------------------------------------------------------------------------------------
def load_config(path: Path = CONFIG_PATH) -> dict:
    try:
        cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise EnvPrecedenceError("F0", f"config unreadable or invalid JSON ({type(e).__name__})") from None
    validate_config(cfg)
    return cfg


def _bad(msg: str):
    raise EnvPrecedenceError("F0", msg)


def _safe_rel(p) -> bool:
    if not isinstance(p, str) or not p or "\x00" in p or "\\" in p or p.startswith("~"):
        return False
    pp = Path(p)
    if not pp.parts or pp.is_absolute() or ".." in pp.parts or "." in pp.parts or p != pp.as_posix():
        return False
    return not any(fnmatch.fnmatchcase(part.lower(), g.lower()) for part in pp.parts for g in _FORBIDDEN_FILE_GLOBS)


def _names(v) -> bool:
    return isinstance(v, list) and all(isinstance(x, str) and x for x in v)


def validate_config(cfg) -> None:
    if not isinstance(cfg, dict):
        _bad("config is not an object")
    unknown, missing = set(cfg) - _CONFIG_KEYS, _CONFIG_KEYS - {"_doc"} - set(cfg)
    if unknown or missing:
        _bad(f"config keys unknown={sorted(unknown)} missing={sorted(missing)}")
    if cfg["schema_version"] != 2 or not isinstance(cfg["parser_version"], str):
        _bad("unsupported schema_version or parser_version")
    ctxs = cfg["contexts"]
    if not isinstance(ctxs, dict) or not ctxs:
        _bad("contexts must be a non-empty object")
    rs = cfg["render_signal"]
    if not (isinstance(rs, dict) and set(rs) == {"var", "value", "context"}):
        _bad("render_signal must have var, value, context")
    for name in (rs["context"], cfg["test_context"], cfg["agent_context"], cfg["default_context"]):
        if name not in ctxs:
            _bad(f"context {name!r} is not defined")
    for k in ("test_markers", "test_modules", "agent_markers", "agent_may_select", "file_denied_names"):
        if not _names(cfg[k]):
            _bad(f"{k} must be a list of names")
    for name in cfg["agent_may_select"]:
        if name not in ctxs:
            _bad("agent_may_select may name only defined contexts")
    ae = cfg["agent_explicit"]
    if not (isinstance(ae, dict) and set(ae) - {"_doc"} == {"files", "excluded_names"}
            and _names(ae["files"]) and _names(ae["excluded_names"])):
        _bad("agent_explicit must have files and excluded_names")
    for name in cfg["agent_may_select"]:
        listed = {f["path"] for f in ctxs[name].get("files", []) if isinstance(f, dict)}
        if listed and not set(ae["files"]) <= listed:
            _bad(f"agent_explicit.files must be listed by context {name!r}")
    rk0 = cfg["restricted_keys"] if isinstance(cfg["restricted_keys"], dict) else {}
    for pat in _FLOOR_DB_PATTERNS:
        if pat not in (rk0.get("patterns") or []) or pat not in ae["excluded_names"]:
            _bad(f"floor pattern {pat} must stay in restricted_keys.patterns and agent_explicit.excluded_names")
    for must in _FILE_DENIED_FLOOR:
        if must not in cfg["file_denied_names"]:
            _bad(f"file_denied_names must include floor name {must}")
    for m in _AGENT_MARKER_FLOOR:
        if m not in cfg["agent_markers"]:
            _bad(f"agent_markers must include floor marker {m}")
    ap = cfg["agent_process_env"]
    if not (isinstance(ap, dict) and set(ap) - {"_doc"} == {"strip", "keep"} and _names(ap["strip"])
            and isinstance(ap["keep"], list) and all(isinstance(x, str) for x in ap["keep"])):
        _bad("agent_process_env must have strip and keep")
    for must in _AGENT_MUST_EXCLUDE + _FLOOR_DB_PATTERNS:
        if must not in ap["strip"]:
            _bad(f"agent_process_env.strip must include floor name {must}")
    if not set(ap["keep"]) <= set(_AGENT_KEEP_MAX):
        _bad(f"agent_process_env.keep may name only {list(_AGENT_KEEP_MAX)}")
    for must in _AGENT_MUST_EXCLUDE:
        if not _match(must, ae["excluded_names"]):
            _bad(f"agent_explicit.excluded_names must exclude {must}")
    for pat in cfg["restricted_keys"]["patterns"] if isinstance(cfg["restricted_keys"], dict) else []:
        if pat not in ae["excluded_names"]:
            _bad(f"agent_explicit.excluded_names must include every restricted pattern ({pat})")
    for fixed in (rs["context"], cfg["test_context"], cfg["agent_context"]):
        if ctxs[fixed].get("files"):
            _bad(f"context {fixed!r} must list no files")
    for cname, c in ctxs.items():
        if not isinstance(c, dict) or set(c) != {"files"} or not isinstance(c["files"], list):
            _bad(f"context {cname!r} must be {{files: [...]}}")
        seen = set()
        for f in c["files"]:
            if not (isinstance(f, dict) and set(f) == {"path", "required"} and _safe_rel(f["path"])
                    and isinstance(f["required"], bool)):
                _bad(f"context {cname!r} has an invalid or forbidden file entry (repo-relative path, required bool)")
            if f["path"] in seen:
                _bad(f"context {cname!r} lists {f['path']} twice")
            seen.add(f["path"])
    pr = cfg["precedence"]
    if not (isinstance(pr, dict) and set(pr) == {"cross_file_conflict", "duplicate_key_in_file", "interpolate"}):
        _bad("precedence must have cross_file_conflict, duplicate_key_in_file, interpolate")
    if pr["cross_file_conflict"] not in _CONFLICT_MODES or pr["duplicate_key_in_file"] not in _DUP_MODES:
        _bad("precedence mode not supported")
    if pr["interpolate"] is not False:
        _bad("interpolate must be false (values are literal)")
    fl = cfg["file_limits"]
    if not (isinstance(fl, dict) and set(fl) == {"max_bytes", "deny_group_other_write", "require_owner_euid"}
            and isinstance(fl["max_bytes"], int) and fl["max_bytes"] > 0
            and fl["deny_group_other_write"] is True and fl["require_owner_euid"] is True):
        _bad("file_limits must set max_bytes > 0 and both checks true")
    rk = cfg["restricted_keys"]
    if not (isinstance(rk, dict) and set(rk) == {"patterns", "require_explicit_context", "allowed"}
            and _names(rk["patterns"]) and rk["require_explicit_context"] is True and isinstance(rk["allowed"], dict)):
        _bad("restricted_keys must have patterns, require_explicit_context=true and allowed")
    for cname, per_file in rk["allowed"].items():
        if cname not in ctxs or not isinstance(per_file, dict):
            _bad(f"restricted_keys.allowed names unknown context {cname!r}")
        listed = {f["path"] for f in ctxs[cname]["files"]}
        for rel, names in per_file.items():
            if rel not in listed or not _names(names) and names != []:
                _bad(f"restricted_keys.allowed.{cname} names a file the context does not list")


# --- branch 1: context -----------------------------------------------------------------------------------------
def select_context(cfg: dict, environ: Mapping[str, str], modules=None) -> tuple[str, str]:
    """Return (context, selected_by). Deterministic: depends only on environ, the loaded module names and cfg."""
    modules = sys.modules if modules is None else modules
    var = cfg["context_var"]
    explicit = environ[var] if var in environ else None          # exact: no strip, case-sensitive
    if explicit is not None and explicit not in cfg["contexts"]:
        raise EnvPrecedenceError("F2", f"{var} names an unknown context")
    rs = cfg["render_signal"]
    if rs["var"] in environ:
        if environ[rs["var"]] != rs["value"]:
            raise EnvPrecedenceError("F1", f"{rs['var']} is present but not exactly {rs['value']!r}")
        if explicit not in (None, rs["context"]):
            raise EnvPrecedenceError("F1", f"{rs['var']} conflicts with {var}")
        return rs["context"], "render"
    if any(m in environ for m in cfg["test_markers"]) or any(m in modules for m in cfg["test_modules"]):
        if explicit not in (None, cfg["test_context"]):
            raise EnvPrecedenceError("F1", f"test signal conflicts with {var}")
        return cfg["test_context"], "test_marker"
    if any(m in environ for m in cfg["agent_markers"]):
        if explicit is None:
            return cfg["agent_context"], "agent_marker"
        if explicit not in cfg["agent_may_select"]:
            raise EnvPrecedenceError("F1", f"agent marker present: {var} may only narrow to {cfg['agent_may_select']}")
        return explicit, ("agent_explicit" if cfg["contexts"][explicit]["files"] else "explicit")
    if explicit is not None:
        return explicit, "explicit"
    return cfg["default_context"], "default"


# --- branch 2: files ---------------------------------------------------------------------------------------------
def _is_home_env(fst) -> bool:
    """True when fst is the same inode as ~/.env (home from the passwd entry and from HOME)."""
    homes = set()
    try:
        import pwd
        homes.add(pwd.getpwuid(os.geteuid()).pw_dir)
    except (ImportError, KeyError):
        pass
    if os.environ.get("HOME"):
        homes.add(os.environ["HOME"])
    for h in homes:
        try:
            hs = os.stat(os.path.join(h, ".env"))
        except OSError:
            continue
        if (hs.st_dev, hs.st_ino) == (fst.st_dev, fst.st_ino):
            return True
    return False


def _read_listed_file(root: Path, rel: str, limits: dict) -> str | None:
    """Text of root/rel, or None when absent. All checks run on the opened fd."""
    real_root = Path(os.path.realpath(root))
    path = real_root / rel
    try:
        st = os.lstat(path)
    except FileNotFoundError:
        return None
    except OSError as e:
        raise EnvPrecedenceError("F3", f"{rel}: cannot stat ({type(e).__name__})") from None
    if stat.S_ISLNK(st.st_mode):
        raise EnvPrecedenceError("F3", f"{rel}: is a symlink")
    if os.path.realpath(path) != str(path):
        raise EnvPrecedenceError("F3", f"{rel}: resolves outside the repo root or through a symlink")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        fd = os.open(path, flags)
    except OSError as e:
        raise EnvPrecedenceError("F3", f"{rel}: cannot open ({type(e).__name__})") from None
    try:
        fst = os.fstat(fd)
        if not stat.S_ISREG(fst.st_mode) or (fst.st_dev, fst.st_ino) != (st.st_dev, st.st_ino):
            raise EnvPrecedenceError("F3", f"{rel}: not a regular file or changed during open")
        if fst.st_nlink != 1:
            raise EnvPrecedenceError("F3", f"{rel}: has {fst.st_nlink} hard links (must be 1)")
        if _is_home_env(fst):
            raise EnvPrecedenceError("F3", f"{rel}: is the user's ~/.env")
        if limits["require_owner_euid"] and fst.st_uid != os.geteuid():
            raise EnvPrecedenceError("F3", f"{rel}: not owned by the current user")
        if limits["deny_group_other_write"] and fst.st_mode & 0o022:
            raise EnvPrecedenceError("F3", f"{rel}: group- or other-writable")
        cap = limits["max_bytes"]
        chunks, total = [], 0
        while True:
            b = os.read(fd, 65536)
            if not b:
                break
            total += len(b)
            if total > cap:
                raise EnvPrecedenceError("F3", f"{rel}: larger than {cap} bytes")
            chunks.append(b)
    finally:
        os.close(fd)
    raw = b"".join(chunks)
    if raw.startswith(b"\xef\xbb\xbf"):
        raise EnvPrecedenceError("F3", f"{rel}: starts with a byte-order mark")
    try:
        return raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        raise EnvPrecedenceError("F3", f"{rel}: not valid UTF-8") from None


def _check_parser(cfg: dict) -> None:
    from dotenv.version import __version__ as v
    if v != cfg["parser_version"]:
        raise EnvPrecedenceError("F0", "python-dotenv version differs from config parser_version")


def _parse(rel: str, text: str, dup_mode: str) -> tuple[dict, dict]:
    """{NAME: value} and {NAME: [notes]}. Uses only the parser's key/value/line number, never Binding.original text."""
    from dotenv.parser import parse_stream  # low-level parser: no file search, no interpolation
    out, notes, seen = {}, {}, set()
    for b in parse_stream(io.StringIO(text)):
        line = b.original.line
        if b.error:
            raise EnvPrecedenceError("F3", f"{rel}: unparseable line {line}")
        if b.key is None:
            continue
        if not _NAME_RE.match(b.key):
            raise EnvPrecedenceError("F3", f"{rel}: invalid variable name on line {line}")
        if b.key in seen:
            if dup_mode == "error":
                raise EnvPrecedenceError("F3", f"{rel}: {b.key} assigned more than once (line {line})")
            notes.setdefault(b.key, []).append(f"duplicate_last_wins@{rel}")
        seen.add(b.key)
        if b.value is None:
            notes.setdefault(b.key, []).append(f"no_value@{rel}")
            out.pop(b.key, None)
            continue
        if "${" in b.value:
            notes.setdefault(b.key, []).append(f"interpolation_not_applied@{rel}")
        out[b.key] = b.value
    return out, notes


def _match(name: str, patterns) -> bool:
    up = name.upper()
    return any(fnmatch.fnmatchcase(up, p.upper()) for p in patterns)


# --- branches 3-5 --------------------------------------------------------------------------------------------------
def resolve(environ: Mapping[str, str], *, cfg: dict | None = None, root: Path = ENV_DIR,
            modules=None) -> tuple[dict, EnvAudit]:
    """Compute (names_to_set, audit) without touching any environment. Raises EnvPrecedenceError."""
    if cfg is None:
        cfg = load_config()
    else:
        validate_config(cfg)
    context, how = select_context(cfg, environ, modules)
    audit = EnvAudit(context=context, selected_by=how)
    files = cfg["contexts"][context]["files"]
    agent = any(m in environ for m in cfg["agent_markers"])
    if agent:
        ap = cfg["agent_process_env"]
        audit.stripped = sorted(n for n in environ if _match(n, ap["strip"]) and n not in ap["keep"])
    elif files:                                   # human local_dev (C3): strip inherited STRIP names first
        audit.stripped = sorted(n for n in environ if _match(n, _INHERITED_STRIP))
    for name in environ:
        if name not in audit.stripped:
            audit.sources[name] = "process"
    agent_x = how == "agent_explicit"
    excluded = cfg["agent_explicit"]["excluded_names"] if agent_x else []
    if agent_x:
        files = [f for f in files if f["path"] in cfg["agent_explicit"]["files"]]
    if not files:
        return {}, audit
    _check_parser(cfg)
    rk = cfg["restricted_keys"]
    allowed = rk["allowed"].get(context, {})
    explicit_ok = how == "explicit" or not rk["require_explicit_context"]
    conflict, dup_mode = cfg["precedence"]["cross_file_conflict"], cfg["precedence"]["duplicate_key_in_file"]
    to_set: dict = {}
    for entry in files:
        rel = entry["path"]
        text = _read_listed_file(root, rel, cfg["file_limits"])
        if text is None:
            if entry["required"]:
                raise EnvPrecedenceError("F5", f"required file {rel} is missing")
            audit.files.append({"path": rel, "status": "missing"})
            continue
        values, notes = _parse(rel, text, dup_mode)
        names = set(values) | set(notes)
        denied = sorted(n for n in names if _match(n, cfg["file_denied_names"]))
        if denied:
            raise EnvPrecedenceError("F7", f"{rel} may not set control names {denied}")
        ok_restricted = set(allowed.get(rel, []))
        restricted = {n for n in names if _match(n, rk["patterns"])}
        bad = sorted(restricted - ok_restricted)
        if bad:
            raise EnvPrecedenceError("F4", f"{rel} may not supply {bad} in context {context}")
        for n, ns in notes.items():
            audit.notes.setdefault(n, []).extend(ns)
        for name, value in values.items():
            if agent_x and _match(name, excluded) or agent and name in audit.stripped:
                audit.skipped[name] = f"excluded_for_agent@{rel}"
                continue
            if name in restricted and not explicit_ok:
                audit.skipped[name] = f"restricted@{rel}: context not selected explicitly"
                continue
            if name in to_set:
                if conflict == "error":
                    raise EnvPrecedenceError("F6", f"{name} is supplied by {audit.sources[name]} and {rel}")
                audit.shadowed.setdefault(name, []).append({"source": rel, "by": audit.sources[name]})
            else:                                 # the file wins over the inherited process env (C3, OR-3)
                to_set[name] = value
                audit.sources[name] = rel
                if name in environ and environ[name] != value:
                    audit.overridden.append(name)
        audit.files.append({"path": rel, "status": "loaded"})
    return to_set, audit


def load_env(environ: MutableMapping[str, str] | None = None, *, cfg: dict | None = None, root: Path = ENV_DIR,
             modules=None, log: bool = True) -> EnvAudit:
    """The single sanctioned loader. Resolves fully, then applies: strips, then file values win (C3).
    For the real os.environ it runs once per process (Streamlit reruns, re-imports return the first audit)."""
    real = environ is None or environ is os.environ
    if real and _PROCESS_AUDIT:
        return _PROCESS_AUDIT[0]
    environ = os.environ if environ is None else environ
    to_set, audit = resolve(environ, cfg=cfg, root=root, modules=modules)
    for name in audit.stripped:
        environ.pop(name, None)
    for name, value in to_set.items():
        environ[name] = value
        audit.applied.append(name)
    for name in audit.overridden:             # always surfaced, by NAME only (OR-3)
        logger.warning("env: file %s overrides inherited %s", audit.sources[name], name)
    if log:
        d = audit.as_dict()
        logger.info("env precedence: context=%s selected_by=%s files=%s file_names=%s shadowed=%s skipped=%s "
                    "notes=%s stripped=%s overridden=%s", d["context"], d["selected_by"], d["files"],
                    {n: s for n, s in d["sources"].items() if s != "process"}, sorted(d["shadowed"]),
                    sorted(d["skipped"]), d["notes"], d["stripped"], d["overridden"])
    if real:
        _PROCESS_AUDIT.append(audit)
    return audit


def check_files(*, context: str | None = None, cfg: dict | None = None, root: Path = ENV_DIR) -> dict:
    """Read-only migration checker: for each file the context lists, report NAMES that the loader would refuse or
    treat differently. Never returns values. Applies nothing."""
    cfg = load_config() if cfg is None else cfg
    validate_config(cfg)
    context = context or cfg["default_context"]
    report = {}
    for entry in cfg["contexts"][context]["files"]:
        rel = entry["path"]
        try:
            text = _read_listed_file(root, rel, cfg["file_limits"])
            if text is None:
                report[rel] = {"status": "missing"}
                continue
            values, notes = _parse(rel, text, "warn")
            names = set(values) | set(notes)
            allowed = set(cfg["restricted_keys"]["allowed"].get(context, {}).get(rel, []))
            report[rel] = {
                "status": "readable",
                "duplicates": sorted(n for n, ns in notes.items() if any(x.startswith("duplicate") for x in ns)),
                "interpolation": sorted(n for n, ns in notes.items() if any(x.startswith("interp") for x in ns)),
                "control_names": sorted(n for n in names if _match(n, cfg["file_denied_names"])),
                "restricted_not_allowed": sorted(n for n in names
                                                 if _match(n, cfg["restricted_keys"]["patterns"]) and n not in allowed),
            }
        except EnvPrecedenceError as e:
            report[rel] = {"status": "refused", "code": e.code, "reason": str(e)}
    return report


def main(argv=None) -> int:
    """Names-only migration check. Run as a FILE (not -m: the checkout is not on sys.path under -I):
        <venv>/bin/python -I <checkout>/vera/env_precedence.py --check [--root <env-dir-to-check>]
    Exit 0 = clean; 1 = something the loader would refuse or treat differently; 2 = usage."""
    import argparse
    ap = argparse.ArgumentParser(prog="env_precedence.py")
    ap.add_argument("--check", action="store_true", required=True)
    ap.add_argument("--root", type=Path, default=ENV_DIR)
    try:
        args = ap.parse_args(argv)
    except SystemExit:
        return 2
    try:
        rep = check_files(root=args.root)
    except EnvPrecedenceError as e:
        print(json.dumps({"error": str(e)}))
        return 1
    print(json.dumps(rep, indent=2, sort_keys=True))
    clean = all(r.get("status") in ("missing", "readable") and not any(
        r.get(k) for k in ("duplicates", "interpolation", "control_names", "restricted_not_allowed"))
        for r in rep.values())
    return 0 if clean else 1


if __name__ == "__main__":
    sys.exit(main())

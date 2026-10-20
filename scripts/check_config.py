"""Non-destructive configuration check (PLAN-ENV v2.1 §4; ENV-B).

    <venv>/bin/python -I -B scripts/check_config.py [--json] [--resolve] [--verify-disposable]

Default mode: no network, no DB connection, writes nothing, loads nothing into os.environ. It reports NAMES, file
metadata, parse codes and classifications only; it never prints a value, host, port, user, dbname, IP or hash.
- `--resolve` uses DNS (only) to build the PI fingerprint set.
- `--verify-disposable` refuses unconditionally (binding owner decision 2026-10-19): exit 3 with
  DISPOSABLE-VERIFY-DISABLED-pending-isolation-review. It never opens a connection.
- `.test-db-marker` is stat()ed only; its contents are never read (owner decision).
- DB-side preconditions (no CREATE for app roles; marker provenance) are always reported UNVERIFIED, never passed.
- `--root` / `--repo` exist for tests on a synthetic layout.
Exit codes: 0 clean, 1 findings, 2 usage error, 3 precondition refused.
"""
from __future__ import annotations

import argparse
import ast
import fnmatch
import json
import os
import re
import stat
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
REPO = HERE.parents[1]                       # R/vera
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from safety import endpoint_class as ec      # noqa: E402  (stdlib-only module)

try:                                         # single source of the R root when ENV-A's module is present
    from vera.env_files import R_ROOT as _R_ROOT   # noqa: E402
except Exception:
    _R_ROOT = REPO.parent

FILES = (".env-vera", ".env-intern", ".env-local-pg", ".test-db-marker")
NOT_READ = (".test-db-marker",)
EXPECTED = {".env-vera": ("VERA_DB_URL_RO",), ".env-intern": (), ".env-local-pg": ()}
ADMIN_FLOOR = ("DB_ADMIN_*", "*_DBADMIN*", "VERA_DB_URL_ADMIN", "VERA_MIGRATE_PRODUCTION")
CONTROL_NAMES = ("VERA_ENV_CONTEXT", "VERA_TEST_DB_URL", "VERA_TEST_DB_DISPOSABLE", "VERA_TEST_RUNNER",
                 "PYTEST_CURRENT_TEST", "RENDER", "CLAUDECODE", "AI_AGENT", "VERA_OWNER_KEY")
C3_DENIED = ("SSL_CERT_*", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE", "HTTP_PROXY", "HTTPS_PROXY", "HTTP*_PROXY",
             "ALL_PROXY", "NO_PROXY", "PG*", "KRB5*", "GSS*")
STRIP = ("*DB_URL*", "DATABASE_URL", "PG*", "KRB5*", "GSS*", "*_DBADMIN*", "DB_ADMIN_*", "VERA_MIGRATE_PRODUCTION",
         "SSL_CERT_*", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE", "HTTP_PROXY", "HTTPS_PROXY", "HTTP*_PROXY",
         "ALL_PROXY", "NO_PROXY")
CONFIG_ENV_ALLOWED = set(FILES) | {"README.md"}
LITERALS = ("AI-Internship", "week-1v2", ".env.db-accounts")
NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _match(name: str, patterns) -> bool:
    return any(fnmatch.fnmatchcase(name.upper(), p.upper()) for p in patterns)


def _names(path: Path) -> tuple[list, list]:
    """(names in order, duplicate names). Values are never kept."""
    names, dups, seen = [], set(), set()
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        s = raw.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        if s.startswith("export ") or s.startswith("export\t"):
            s = s[7:].lstrip()
        k = s.partition("=")[0].strip()
        if NAME_RE.fullmatch(k) is None:
            k = "<malformed>"
        if k in seen:
            dups.add(k)
        seen.add(k)
        names.append(k)
    return names, sorted(dups)


def _url_values(path: Path) -> dict:
    """URL-valued names -> value, held only for parse/classify in this process; never printed."""
    if path is None or not path.is_file() or path.is_symlink():
        return {}
    return ec._read_names_values(path, ec.URL_NAMES + ("VERA_TEST_DB_URL",))


_GIT = {"enabled": True}


def _git(root: Path, *args) -> str:
    if not _GIT["enabled"]:
        return "skipped"
    try:
        r = subprocess.run(["git", "-C", str(root), *args], capture_output=True, timeout=20,
                           env={"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "GIT_TERMINAL_PROMPT": "0"})
    except (OSError, subprocess.SubprocessError):
        return "error"
    return {0: "yes", 1: "no"}.get(r.returncode, "error")


def file_report(root: Path, env_dir: Path, name: str, findings: list) -> dict:
    p = env_dir / name
    rep: dict = {"name": name, "exists": p.exists() or p.is_symlink()}
    if not rep["exists"]:
        if name == ".env-vera":
            findings.append(f"{name}: missing (required, D-ENV-1)")
        return rep
    st = p.lstat()
    rep.update(symlink=stat.S_ISLNK(st.st_mode), regular=stat.S_ISREG(st.st_mode),
               owner_is_euid=st.st_uid == os.geteuid(), mode=oct(stat.S_IMODE(st.st_mode)), size=st.st_size)
    if rep["symlink"] or not rep["regular"]:
        findings.append(f"{name}: not a regular file")
    if not rep["owner_is_euid"]:
        findings.append(f"{name}: owner is not the euid")
    if stat.S_IMODE(st.st_mode) != 0o600:
        findings.append(f"{name}: mode is not 0600")
    rel = p.relative_to(root).as_posix() if p.is_relative_to(root) else str(p)
    rep["git_ignored"] = _git(root, "check-ignore", "-q", rel)
    rep["git_tracked"] = _git(root, "ls-files", "--error-unmatch", rel)
    if rep["git_tracked"] == "yes":
        findings.append(f"{name}: tracked by git")
    if name in NOT_READ:
        rep["contents"] = "not read (owner decision 2026-10-19)"
        return rep
    if rep["symlink"] or not rep["regular"]:
        return rep
    names, dups = _names(p)
    present = sorted(set(names))
    rep["names"] = present
    rep["expected_missing"] = sorted(set(EXPECTED.get(name, ())) - set(present))
    rep["duplicates"] = dups
    rep["admin_floor"] = sorted(n for n in present if _match(n, ADMIN_FLOOR))
    rep["control_names"] = sorted(n for n in present if n in CONTROL_NAMES)
    rep["c3_denied"] = sorted(n for n in present if _match(n, C3_DENIED))
    if dups:
        findings.append(f"{name}: duplicate names")
    if rep["admin_floor"] and name == ".env-vera":
        findings.append(f"{name}: admin-floor names present")
    if rep["control_names"]:
        findings.append(f"{name}: control names present")
    if rep["c3_denied"]:
        findings.append(f"{name}: C3-denied names present")
    return rep


def _loader_preview(env_dir: Path) -> dict:
    try:
        from vera import env_precedence
        return {"status": "ok", "report": env_precedence.check_files(root=env_dir, context="local_dev")}
    except Exception as e:                     # names-only: the exception type, never its text
        return {"status": "unavailable", "error": type(e).__name__}


def _string_constants(path: Path):
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except (SyntaxError, OSError):
        return
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            yield node.value


def _tuple_constant(path: Path, name: str):
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except (SyntaxError, OSError):
        return None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == name for t in node.targets):
            try:
                return tuple(ast.literal_eval(node.value))
            except ValueError:
                return None
    return None


class _Skip(Exception):
    pass


def layout_checks(root: Path, repo: Path, env_dir: Path, findings: list) -> dict:
    out: dict = {}
    bad = []
    for top in (repo, root / "work", root / "course-service"):
        if not top.is_dir():
            continue
        for dirpath, dirnames, filenames in os.walk(top):
            dirnames[:] = [d for d in dirnames if d not in (".git", "__pycache__", ".venv", "node_modules")]
            for f in filenames:
                if f.startswith(".env") and not f.endswith(".example"):
                    bad.append(Path(dirpath, f).relative_to(root).as_posix())
            if Path(dirpath).name == "env" and Path(dirpath).parent.name == "config":
                bad.append(Path(dirpath).relative_to(root).as_posix() + "/")
    extra = []
    if env_dir.is_dir():
        extra = sorted(f.name for f in env_dir.iterdir()
                       if f.name not in CONFIG_ENV_ALLOWED and not f.name.endswith(".example"))
    out["R-G1"] = {"stray_env_paths": sorted(bad), "unexpected_in_config_env": extra}
    if bad or extra:
        findings.append("R-G1: stray env files or unexpected entries")
    try:
        if not _GIT["enabled"]:
            raise _Skip()
        r = subprocess.run(["git", "-C", str(repo), "remote", "-v"], capture_output=True, timeout=20,
                           env={"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "GIT_TERMINAL_PROMPT": "0"})
        remotes = "none" if r.returncode == 0 and not r.stdout.strip() else ("present" if r.returncode == 0
                                                                              else "error")
    except _Skip:
        remotes = "skipped"
    except (OSError, subprocess.SubprocessError):
        remotes = "error"
    out["R-G5"] = {"remotes": remotes, "git_dir_is_directory": (repo / ".git").is_dir() and
                   not (repo / ".git").is_symlink()}
    if remotes not in ("none", "skipped") or not out["R-G5"]["git_dir_is_directory"]:
        findings.append("R-G5: remote present, git error, or .git is not a directory")
    count = 0
    targets = [repo / "main.py"] + [p for d in ("vera", "scripts", "db") if (repo / d).is_dir()
                                    for p in sorted((repo / d).rglob("*.py"))]
    for p in targets:
        if p.is_file() and p.resolve() != HERE:
            count += sum(1 for s in _string_constants(p) for lit in LITERALS if lit in s)
    out["legacy_literal_count"] = count
    if count:
        findings.append("legacy path literals present")
    a = _tuple_constant(repo / "scripts" / "vera_test_runner.py", "CRED_MASK_GLOBS")
    b = _tuple_constant(repo / "tests" / "vera_test_boundary.py", "CRED_FILE_GLOBS")
    out["cred_globs_equal"] = a is not None and a == b
    if not out["cred_globs_equal"]:
        findings.append("C10: CRED_MASK_GLOBS and CRED_FILE_GLOBS differ or are missing")
    return out


def build_report(root: Path, environ, *, resolve: bool = False, repo: Path | None = None,
                 git: bool = True) -> tuple[dict, list]:
    _GIT["enabled"] = bool(git)
    root = Path(root)
    repo = Path(repo) if repo is not None else root / "vera"
    env_dir = root / "config" / "env"
    findings: list = []
    report: dict = {"files": [file_report(root, env_dir, n, findings) for n in FILES]}
    report["loader_preview"] = _loader_preview(env_dir)
    vera_vals = _url_values(env_dir / ".env-vera")
    intern_vals = _url_values(env_dir / ".env-intern")
    vera_names = set(vera_vals) | set(next((f.get("names", []) for f in report["files"]
                                            if f["name"] == ".env-vera"), []))
    report["c3_preview"] = {"would_strip": sorted(n for n in environ if _match(n, STRIP)),
                            "would_override": sorted(n for n in environ if n in vera_names)}
    proc_vals = {k: v for k, v in environ.items() if k in ec.URL_NAMES or k == "VERA_TEST_DB_URL"}
    parse_status = []
    for source, vals in ((".env-vera", vera_vals), ("process", proc_vals)):
        for name in sorted(vals):
            try:
                ec.parse(vals[name])
                code = "ok"
            except ec.ParseError as e:
                code = e.code
            parse_status.append({"name": name, "source": source, "status": code})
            if code != "ok":
                findings.append(f"{name} ({source}): parse {code}")
    report["parse_status"] = parse_status
    resolver = None
    if resolve:
        import socket

        def resolver(host):
            return {ai[4][0] for ai in socket.getaddrinfo(host, None)}
    vf = env_dir / ".env-vera"
    inf = env_dir / ".env-intern"
    sources = ec.Sources.from_files(vf if vf.is_file() and not vf.is_symlink() else None,
                                    inf if inf.is_file() and not inf.is_symlink() else None,
                                    dict(environ), resolver=resolver)
    classes = []
    for source, vals in ((".env-vera", vera_vals), (".env-intern", intern_vals), ("process", proc_vals)):
        for name in sorted(vals):
            v = ec.classify(vals[name], sources=sources, environ=environ, name=name, resolver=resolver)
            classes.append({"name": name, "source": source, "cls": v.cls, "reasons": list(v.reasons)})
    for d in (vera_vals, intern_vals, proc_vals):
        d.clear()
    report["classification"] = classes
    report["layout"] = layout_checks(root, repo, env_dir, findings)
    sql = repo / "db" / "accounts" / "create_vera_accounts.sql"
    try:
        static_revoke = bool(re.search(r"REVOKE\s+CREATE\s+ON\s+DATABASE", sql.read_text(errors="replace"), re.I))
    except OSError:
        static_revoke = None
    report["db_side"] = [
        {"precondition": "app roles RO/RW/WO have no CREATE on the production database", "status": "UNVERIFIED",
         "static_reading": {"file": "db/accounts/create_vera_accounts.sql",
                            "revoke_create_on_database_text_present": static_revoke},
         "note": "pending the owner (D-ENV-6); a static reading is never a pass"},
        {"precondition": "disposable marker provenance", "status": "UNVERIFIED",
         "note": ec.VERIFY_DISABLED},
    ]
    report["findings"] = findings
    return report, findings


def _print_text(report: dict, out) -> None:
    for f in report["files"]:
        meta = ", ".join(f"{k}={v}" for k, v in f.items() if k not in ("name", "names"))
        print(f"file {f['name']}: {meta}", file=out)
        if "names" in f:
            print(f"  names: {' '.join(f['names']) or '-'}", file=out)
    lp = report["loader_preview"]
    print(f"loader_preview: {lp['status']}" + (f" ({lp.get('error')})" if lp["status"] != "ok" else ""), file=out)
    if lp["status"] == "ok":
        print(f"  {json.dumps(lp['report'], sort_keys=True)}", file=out)
    c3 = report["c3_preview"]
    print(f"c3 would_strip: {' '.join(c3['would_strip']) or '-'}", file=out)
    print(f"c3 would_override: {' '.join(c3['would_override']) or '-'}", file=out)
    for p in report["parse_status"]:
        print(f"parse {p['name']} {p['source']} {p['status']}", file=out)
    for c in report["classification"]:
        print(f"class {c['name']} {c['source']} {c['cls']} {','.join(c['reasons'])}", file=out)
    print(f"layout {json.dumps(report['layout'], sort_keys=True)}", file=out)
    for d in report["db_side"]:
        print(f"db-side {d['status']}: {d['precondition']}", file=out)
    for f in report["findings"]:
        print(f"FINDING {f}", file=out)
    print("result: " + ("findings" if report["findings"] else "clean"), file=out)


def main(argv=None, environ=None, out=None) -> int:
    out = sys.stdout if out is None else out
    ap = argparse.ArgumentParser(prog="check_config.py", description="Names-only VERA-DAR configuration check.")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--resolve", action="store_true")
    ap.add_argument("--verify-disposable", action="store_true")
    ap.add_argument("--root", type=Path, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--repo", type=Path, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--no-git", action="store_true", help=argparse.SUPPRESS)   # tests: no git child processes
    args = ap.parse_args(argv)                 # usage errors exit 2
    environ = dict(os.environ if environ is None else environ)
    if args.verify_disposable:
        # Owner decision 2026-10-19: refuses unconditionally, before any install or connection.
        msg = {"verify_disposable": "refused", "code": ec.VERIFY_DISABLED}
        print(json.dumps(msg) if args.json else f"verify-disposable refused: {ec.VERIFY_DISABLED}", file=out)
        return 3
    root = args.root if args.root is not None else Path(_R_ROOT)
    report, findings = build_report(root, environ, resolve=args.resolve, repo=args.repo,
                                     git=not args.no_git)
    if args.json:
        print(json.dumps(report, sort_keys=True, indent=1), file=out)
    else:
        _print_text(report, out)
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())

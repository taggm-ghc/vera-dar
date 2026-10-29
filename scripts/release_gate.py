"""VERA-DAR release gate (item #72 W5). Offline by default; exits non-zero on any failure.

Checks: (a) full test suite, (b) frozen-file integrity (sha256), (c) static safety invariants,
(d) optional --live: scope router layer-1-only regression (zero cost, no network).
Scope sets are directional, small n (R72-g): no statistical claim is made. Never prints .env values.
"""
from __future__ import annotations

import argparse
import errno
import ast
import hashlib
import json
import os
import re
import shutil
import signal
import stat
import subprocess
import sys
import time
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PYTEST_TIMEOUT_S = 1200
# unsafe_allow_html is permitted only on this escaped line (same allowlist as tests/test_item72_public_mode.py)
UNSAFE_HTML_ALLOWED = {"vera/m7/demo_ui.py": "html.escape(span_text)"}
UI_GLOBS = ("streamlit_app.py", "pages/*.py", "vera/m7/*.py")
ENV_ATTRS = {"getenv", "environ"}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ui_files(root: Path) -> list[Path]:
    out: list[Path] = []
    for g in UI_GLOBS:
        out.extend(sorted(root.glob(g)))
    return out


# L4-A (PF-18 L-4): the pytest child gets an ALLOWLISTED environment, never the caller's (PYTHONPATH/sitecustomize,
# PYTHONWARNINGS, LD_*, PYTEST_*, VERA_INTEGRATION, VERA_TEST_BOUNDARY_CHILD, DB/provider settings would all act before
# the boundary is installed). `-I` ignores PYTHON* variables and the user site and keeps the cwd off sys.path; `-B`
# writes no bytecode (PYTHONDONTWRITEBYTECODE=1 is ignored by the -I child itself and only reaches its non-isolated Python
# children); plugin autoload is off (anyio.pytest_plugin would load pre-boundary).
# S-2: PATH is FIXED (the boundary's re-exec resolves `sh`/`unshare` through it), and TMPDIR is a fresh per-run directory.
CHILD_ENV_NAMES = frozenset({"HOME", "USER", "LANG", "TERM"})
CHILD_PATH = "/usr/bin:/bin"
GATE_TMP_REL = "tmp/release-gate"   # per-run TMPDIR parent under the repo (gitignored `tmp/`); explicit, no /tmp fallback
# S-1: the tracked root pytest.ini is pinned (`-c`, `--rootdir`), so no ancestor or untracked pytest config is read.
PYTEST_CONFIG_NAMES = ("pytest.toml", ".pytest.toml", "pytest.ini", ".pytest.ini", "pyproject.toml", "tox.ini", "setup.cfg")
# GATEFIX-1029: the suite is launched THROUGH the sanctioned runner (tests/vera_test_boundary.py RUNNER_MESSAGE: "the
# release gate must launch through it too"). The boundary honours the gate nonce only inside an already isolated
# session and has no re-exec fallback, so a bare `python -m pytest` child exits 78 on any host with a non-lo interface.
# The runner builds the namespaces and masks, forwards only a valid 32-hex VERA_GATE_NONCE (sandbox_env), and itself
# pins `-I -B -X pycache_prefix=... -m pytest -c <root>/pytest.ini --rootdir <root> -p no:cacheprovider` (S-1).
RUNNER_SCRIPT = "scripts/vera_test_runner.py"
SUITE_ARGS = ("-q", "tests", "--ignore=tests/test_m2_integration.py")


def suite_argv(root, python=None) -> list:
    """Pure: the gate's suite command, `<python> -I -B <root>/scripts/vera_test_runner.py -- <SUITE_ARGS>`."""
    root = Path(root)
    return [python or sys.executable, "-I", "-B", str(root / RUNNER_SCRIPT), "--", *SUITE_ARGS]


def inherited_pytest_configs(root, is_file=os.path.isfile) -> list:
    """S-1: every pytest config file pytest's upward search could find for `tests` OTHER than the tracked root
    pytest.ini (in tests/, the repo root and every ancestor), plus the tracked file itself if it is missing. Paths only."""
    root = Path(root)
    found = [] if is_file(str(root / "pytest.ini")) else [f"{root / 'pytest.ini'} (tracked root config) missing"]
    for base in (root / "tests", root, *root.parents):
        for name in PYTEST_CONFIG_NAMES:
            p = base / name
            if p != root / "pytest.ini" and is_file(str(p)):
                found.append(str(p))
    return found


def gate_tmp_refusal(root, must_exist: bool):
    """INT-1 M-9: why the per-run TMPDIR base must not be used, or None. Checks `<root>/tmp` and `<root>/tmp/release-gate`
    with lstat (never following a link): each must be a real directory (not a symlink) owned by this uid, and the base
    must not be group- or other-writable. A missing component is allowed only before makedirs (must_exist False)."""
    root = Path(root)
    for rel, strict in (("tmp", False), (GATE_TMP_REL, True)):
        try:
            st = os.lstat(str(root / rel))
        except FileNotFoundError:
            if must_exist:
                return f"{rel} is missing after makedirs"
            return None
        except OSError as exc:
            return f"{rel} cannot be checked ({type(exc).__name__})"
        if stat.S_ISLNK(st.st_mode):
            return f"{rel} is a symlink"
        if not stat.S_ISDIR(st.st_mode):
            return f"{rel} is not a directory"
        if st.st_uid != os.getuid():
            return f"{rel} is not owned by the gate's uid"
        if strict and st.st_mode & 0o022:
            return f"{rel} is group- or other-writable"
    return None


def child_env(environ, nonce: str, tmpdir: str) -> dict:
    """Pure: the pytest child's environment. Allowlisted names only (plus LC_*); any value that looks like a URL or
    carries credentials (`://`, `@`) is dropped. PATH is fixed and TMPDIR is the given per-run dir (S-2). Never prints or
    returns a dropped value."""
    env = {k: v for k, v in environ.items()
           if (k in CHILD_ENV_NAMES or k.startswith("LC_")) and "://" not in v and "@" not in v}
    env["PATH"] = CHILD_PATH
    env["TMPDIR"] = str(tmpdir)
    env["VERA_GATE_NONCE"] = nonce
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


# L4-D (PF-18 L-4, SEC R6/R11): the gate refuses an unsafe launch of ITSELF. Honest limit: this catches accidental
# launches only; code from a hostile caller environment has already run in this process. The control against that is
# the documented owner invocation:
#   env -i PATH=/usr/bin:/bin HOME="$HOME" USER="$USER" LANG=C.UTF-8 <venv>/bin/python -I scripts/release_gate.py ...
_UNSAFE_PREFIXES = ("LD_", "PYTHON", "PYTEST_", "GIT_")          # S-10: GIT_CONFIG_* can make the gate's git run code
_UNSAFE_NAMES = frozenset({"VERA_INTEGRATION", "VERA_TEST_BOUNDARY_CHILD", "BASH_ENV", "ENV"})


def launch_refusals(environ, flags) -> list[str]:
    """Pure: reasons the gate must not start (variable NAMES only, never values). Empty list = safe to start."""
    out = []
    if not getattr(flags, "isolated", 0):
        out.append("the gate must be started with `python -I` (see the owner invocation in scripts/release_gate.py)")
    bad = sorted(k for k in environ if k.startswith(_UNSAFE_PREFIXES) or k in _UNSAFE_NAMES)
    if bad:
        out.append("unsafe variables in the gate's environment: " + ", ".join(bad))
    return out


def check_tests(root: Path = ROOT) -> list[str]:
    """Run the suite and require nonce-bound proof of OS-level network isolation (K2.S-b): the run's pytest child must
    print exactly one VERA-ISOLATION-EVIDENCE line carrying this run's random nonce. The child gets an allowlisted
    environment (L4-A, S-2) and is launched through scripts/vera_test_runner.py (GATEFIX-1029), which pins the root
    pytest.ini (S-1) and builds the isolated session; the run is REFUSED, before anything is spawned, if
    PYTEST_ADDOPTS is set or any other pytest config file could be found. The child runs in its own session and its
    whole process group is SIGKILLed before the leader is reaped (D-B5-1 S-2, via _run); the per-run TMPDIR is removed
    only after that, and echoed child output is made printable (G-8)."""
    import secrets
    import shutil
    import tempfile
    refused = (["PYTEST_ADDOPTS is set in the gate's environment"] if "PYTEST_ADDOPTS" in os.environ else [])
    refused += [f"inherited pytest config found: {c}" for c in inherited_pytest_configs(root)]
    if refused:
        return ["test run refused (S-1): " + r for r in refused]
    nonce = secrets.token_hex(16)
    base = Path(root) / GATE_TMP_REL                 # project-owned (gitignored tmp/), never the system /tmp
    why = gate_tmp_refusal(root, must_exist=False)   # INT-1 M-9: no symlinked/foreign/writable base, before and after
    if why is None:
        os.makedirs(base, mode=0o700, exist_ok=True)
        why = gate_tmp_refusal(root, must_exist=True)
    if why:
        return [f"test run refused (S-2): per-run TMPDIR base {why}"]
    tmpdir = tempfile.mkdtemp(prefix="vera-gate-", dir=str(base))
    env = child_env(_child_env(os.environ), nonce, tmpdir)   # INT-1 M-1: L4-A/S-2 allowlist over the G-3 deny filter
    cmd = suite_argv(root)                                   # GATEFIX-1029: through the sanctioned runner
    try:  # D-B5-1 G-3/S-2: own session; the whole group is killed before the leader is reaped (INT-1 M-2)
        p = _run(cmd, cwd=root, timeout=PYTEST_TIMEOUT_S, env=env, tmpdir=tmpdir)
    except subprocess.TimeoutExpired:
        return [f"test suite exceeded {PYTEST_TIMEOUT_S}s"]
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    if p.returncode in (64, 78):   # vera_test_runner EXIT_REFUSED / EXIT_ISOLATION (boundary refusals are 78 too)
        what = ("test run refused by the sanctioned runner (exit 64)" if p.returncode == 64 else
                "test isolation was not established for this run (exit 78: the runner or the test boundary refused)")
        return [what] + [_printable(x) for x in (p.stderr or "").strip().splitlines()[-5:]]
    lines = [ln for ln in (p.stdout or "").strip().splitlines() if ln.strip()]
    summary = lines[-1] if lines else "(no output)"
    if p.returncode != 0:
        return [_printable(f"pytest exit {p.returncode}: {summary}")] + [_printable(x) for x in (
            lines[-15:-1] + (p.stderr or "").strip().splitlines()[-5:])]
    evidence = f"VERA-ISOLATION-EVIDENCE netns=isolated nonce={nonce}"
    if [ln.strip() for ln in (p.stdout or "").splitlines()].count(evidence) != 1:
        return ["test run did not prove OS-level network isolation (missing or invalid isolation evidence line)"]
    print(f"      {_printable(summary)}")
    return []


# D-B5-1 G-3: variables a child running candidate code must not inherit (agent socket, DB URLs, libpq settings).
_CHILD_ENV_DENY = re.compile(r"^(SSH_AUTH_SOCK|SSH_AGENT_PID|DATABASE_URL|PG[A-Z_]*)$|DB_URL")
_UNPRINTABLE = re.compile(r"[^\x20-\x7e]")


def _child_env(env: dict[str, str]) -> dict[str, str]:
    return {k: v for k, v in env.items() if not _CHILD_ENV_DENY.search(k)}


def _printable(text: str) -> str:
    """G-8: echo child output without control characters (no ANSI repaint of a fake RESULT line)."""
    return _UNPRINTABLE.sub("?", str(text))


_STDLIB_RUN = subprocess.run
WAIT_POLL_S = 0.05


def _run(cmd, cwd, timeout, env, input=None, tmpdir=None):
    """Run a child for the gate. Normally _run_session (S-2). If subprocess.run has been replaced in this process
    (only the guarded unit tests do that, e.g. tests/test_env_guard.py `_fake_run` and the gate tests after it), call
    the replacement with subprocess.run's own keywords, so those tests keep faking the suite and never spawn a real one.
    `tmpdir` (INT-1 M-3): where the session runner keeps its unnamed capture files; None = the gate's default."""
    if subprocess.run is not _STDLIB_RUN:
        return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout, env=env, input=input)
    return _run_session(cmd, cwd, timeout, env, input, tmpdir)


def _run_session(cmd, cwd, timeout, env, input=None, tmpdir=None):
    """S-2: Popen in a new session (the leader is the process group); output goes to unnamed temp files (no pipe a
    detached grandchild can hold open). Wait for exit WITHOUT reaping (waitid WNOWAIT), SIGKILL the whole group while
    the zombie leader still pins the pgid (no pgid reuse), then reap. On timeout: kill the group, reap, raise."""
    import tempfile
    with tempfile.TemporaryFile(dir=tmpdir) as out, tempfile.TemporaryFile(dir=tmpdir) as err:
        p = subprocess.Popen(cmd, cwd=cwd, env=env, stdin=subprocess.PIPE if input is not None else subprocess.DEVNULL,
                             stdout=out, stderr=err, start_new_session=True)
        try:
            if input is not None:
                try:
                    p.stdin.write(input.encode())
                    p.stdin.close()
                except OSError:
                    pass
            deadline = time.monotonic() + timeout
            while os.waitid(os.P_PID, p.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT) is None:
                if time.monotonic() > deadline:
                    raise subprocess.TimeoutExpired(cmd, timeout)
                time.sleep(WAIT_POLL_S)
        finally:
            _kill_group(p.pid)
            p.wait()
        out.seek(0)
        err.seek(0)
        return subprocess.CompletedProcess(cmd, p.returncode, out.read().decode("utf-8", "replace"),
                                           err.read().decode("utf-8", "replace"))


def _kill_group(pid) -> None:
    if pid:
        try:
            os.killpg(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass


CATALOG_SCRIPT = "scripts/live_catalog_check.py"
_RESULT_LINE = re.compile(r"^RESULT: [A-Z_]+( \d+)?$")
# D-B5-1: (a2) binds the evidence to the commit under test through the B5 checker, pinned by sha256 so that a
# checker edit cannot silently weaken the gate. Change either constant only by a guarded, reviewed edit.
CATALOG_CHECKER = "scripts/check_catalog_evidence_freshness.py"
CATALOG_CHECKER_SHA256 = "48d4ac087179b42ff64bd49581684c5b543a74afc6aae5f85e626fa9f5734aa2"
CATALOG_EVIDENCE_MAX_AGE_HOURS = None  # DV-6: no limit until a calibration note on real runs exists (RC-7)
GIT_TIMEOUT_S = 10
_FULL_SHA = re.compile(r"^[0-9a-f]{40}$")


def _git_env() -> dict[str, str]:
    """Environment for the read-only git calls: no GIT_* redirection (GIT_DIR, GIT_WORK_TREE, GIT_INDEX_FILE ...)."""
    env = _child_env({k: v for k, v in os.environ.items() if not k.startswith("GIT_")})
    env["GIT_OPTIONAL_LOCKS"] = "0"
    env["GIT_CONFIG_NOSYSTEM"] = "1"      # G-6: no system or global config (include.path, filters, autocrlf)
    env["GIT_CONFIG_GLOBAL"] = os.devnull
    return env


GIT_SEARCH_PATH = "/usr/bin:/bin"  # G-6: git is resolved on a fixed PATH, not the caller's


def _git_exe() -> str | None:
    return shutil.which("git", path=GIT_SEARCH_PATH)


def _index_flags_ok(ls_files_v: str) -> bool:
    """G-2: every index entry must be tag H (no assume-unchanged, skip-worktree/sparse, unmerged or other tag)."""
    entries = [e for e in ls_files_v.split("\0") if e]
    return bool(entries) and all(e.startswith("H ") for e in entries)


def _tree_matches(root: Path, ls_tree_z: bytes) -> bool:
    """G-2: the on-disk bytes of every tracked path equal the HEAD tree (`git ls-tree -r -z --full-tree HEAD`),
    computed here from the files, never from the index. Only regular blobs (100644/100755) and symlinks (120000);
    anything else (submodule, unknown mode, malformed line) fails closed."""
    entries = [e for e in ls_tree_z.split(b"\0") if e]
    if not entries:
        return False
    for e in entries:
        meta, sep, rel = e.partition(b"\t")
        parts = meta.split(b" ")
        if not sep or len(parts) != 3 or parts[1] != b"blob" or not rel:
            return False
        mode, oid = parts[0], parts[2].decode("ascii", "replace")
        algo = {40: hashlib.sha1, 64: hashlib.sha256}.get(len(oid))
        path = Path(root) / os.fsdecode(rel)
        try:
            if mode == b"120000":
                if not path.is_symlink():
                    return False
                data = os.fsencode(os.readlink(path))
            elif mode in (b"100644", b"100755"):
                st = path.lstat()
                if not stat.S_ISREG(st.st_mode) or bool(st.st_mode & 0o100) != (mode == b"100755"):
                    return False
                data = path.read_bytes()
            else:
                return False
        except OSError:
            return False
        if algo is None or algo(b"blob %d\0" % len(data) + data).hexdigest() != oid:
            return False
    return True


def _candidate_state(root: Path = ROOT) -> tuple[str | None, bool]:
    """Read-only: (HEAD of the repository whose top level IS `root`, working tree clean). Any doubt -> (None, False).

    Runs git rev-parse, ls-files, ls-tree and status (no shell, no write, no replace refs, no fsmonitor or untracked
    cache, no system/global config). "Clean" means: every index entry is tag H, the on-disk bytes of every tracked
    path equal HEAD's tree (independent of the index), and `git status` reports nothing. A `root` nested inside some
    other repository (top level != root) gives None. Called only by run_offline_checks, never by
    catalog_evidence_outcome. Ignored paths (.venv/, __pycache__/) are NOT covered: release runs use a fresh clone."""
    git = _git_exe()
    if not git:
        return None, False
    base = [git, "--no-replace-objects", "-c", "core.fsmonitor=false", "-c", "core.untrackedCache=false",
            "-C", str(root)]
    kw = {"shell": False, "timeout": GIT_TIMEOUT_S, "env": _git_env(), "capture_output": True}
    try:
        p = subprocess.run(base + ["rev-parse", "--show-toplevel", "HEAD"], text=True, **kw)
        out = (p.stdout or "").splitlines()
        if p.returncode != 0 or len(out) != 2 or not _FULL_SHA.match(out[1].strip()):
            return None, False
        if Path(out[0].strip()).resolve() != Path(root).resolve():
            return None, False
        head = out[1].strip()
        v = subprocess.run(base + ["ls-files", "-v", "-z"], text=True, **kw)
        t = subprocess.run(base + ["ls-tree", "-r", "-z", "--full-tree", head], **kw)
        s = subprocess.run(base + ["status", "--porcelain=v1", "--untracked-files=all"], text=True, **kw)
        clean = (v.returncode == 0 and _index_flags_ok(v.stdout or "") and t.returncode == 0
                 and _tree_matches(Path(root), t.stdout or b"") and s.returncode == 0 and (s.stdout or "") == "")
        return head, clean
    except Exception:  # git missing, timeout, odd path: fail closed
        return None, False


def _bound_candidate(pre, post, expect=None) -> tuple[str | None, str]:
    """The commit (a2) may bind to: HEAD read before AND after the suite, equal, both clean, and equal to
    --expect-candidate-sha when given. Otherwise (None, fixed reason)."""
    (sha_a, clean_a), (sha_b, clean_b) = pre, post
    if sha_a is None or sha_b is None:
        return None, "HEAD of the gate worktree could not be read"
    if sha_a != sha_b:
        return None, "HEAD moved during the test suite"
    if not clean_a:
        return None, "working tree not clean before the test suite"
    if not clean_b:
        return None, "working tree not clean after the test suite"
    if expect is not None and (expect or "").strip().lower() != sha_a:
        return None, "HEAD is not the --expect-candidate-sha commit"
    return sha_a, ""


def _load_checker(root: Path = ROOT):
    """Load the B5 checker from the bytes that were hashed (no re-read between pin check and execution)."""
    path = root / CATALOG_CHECKER
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError("checker")
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != CATALOG_CHECKER_SHA256:
        raise ImportError("checker is not the pinned version")
    mod = types.ModuleType("vera_gate_catalog_checker")
    mod.__file__ = str(path)
    exec(compile(data, str(path), "exec"), mod.__dict__)
    return mod


EVIDENCE_MAX_BYTES = 1 << 20


def _read_evidence_file(path) -> tuple[str, bytes]:
    """Q-1 / G-7: ("ok", bytes) only for a regular, non-symlink file of at most EVIDENCE_MAX_BYTES, opened with
    O_NOFOLLOW|O_NONBLOCK and checked with fstat (a FIFO or device never blocks the gate). Otherwise
    ("missing" | "symlink" | "irregular" | "too large", b"")."""
    p = Path(path)
    try:
        if p.is_symlink():
            return "symlink", b""
        fd = os.open(p, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0))
    except OSError as exc:
        return ("symlink" if getattr(exc, "errno", None) == errno.ELOOP else "missing"), b""
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            return "irregular", b""
        if st.st_size > EVIDENCE_MAX_BYTES:
            return "too large", b""
        chunks, total = [], 0
        while True:
            b = os.read(fd, 65536)
            if not b:
                break
            total += len(b)
            if total > EVIDENCE_MAX_BYTES:
                return "too large", b""
            chunks.append(b)
        return "ok", b"".join(chunks)
    except OSError:
        return "missing", b""
    finally:
        os.close(fd)


def _evidence_snapshot(path) -> tuple:
    """G-3: (kind, sha256) of the evidence file and of its default sidecar; compared before and after the suite."""
    if not path:
        return ()
    out = []
    for q in (Path(path), Path(str(path) + ".sidecar")):
        kind, data = _read_evidence_file(q)
        out.append((kind, hashlib.sha256(data).hexdigest() if kind == "ok" else ""))
    return tuple(out)


def catalog_evidence_outcome(path, root: Path = ROOT, candidate_sha: str | None = None) -> tuple[str, str]:
    """Offline check of the owner-run live-catalog evidence (OD-S1, D-B5-1). Reads only `path`, its sidecar, the
    catalog script's bytes and the pinned B5 checker's bytes; never connects to a database, never runs the script,
    the suite or any child process, reads no environment variable. `candidate_sha` is the commit under test, read by
    the caller; None fails closed. Only PASSED succeeds; its reason is the checker's note (12-char prefixes only)."""
    missing = ("MISSING", "live catalog check required: the owner runs `python scripts/live_catalog_check.py "
               "--i-confirm-owner-run > catalog-evidence-<sha12>.txt` (read-only, operator-run) and writes its "
               ".sidecar per the owner runbook, then passes --catalog-evidence catalog-evidence-<sha12>.txt; "
               "the gate never connects to a database")
    if not path:
        return missing
    kind, data = _read_evidence_file(path)
    if kind == "symlink":
        return "STALE", "evidence file is a symlink (refused)"
    if kind == "too large":
        return "STALE", "evidence file is too large"
    if kind != "ok":  # missing, directory, FIFO, device
        return missing
    try:
        lines = [ln.strip() for ln in data.decode("utf-8").splitlines()]
    except ValueError:  # not UTF-8 text
        return "STALE", "evidence is not UTF-8 text"
    stale = ("STALE", "evidence is for a different version of scripts/live_catalog_check.py")
    try:
        want = f"script_sha256={sha256_file(root / CATALOG_SCRIPT)}"
    except OSError:
        return stale
    if not any(ln == want or ln.endswith(" " + want) for ln in lines):
        return stale
    results = [ln for ln in lines if ln.startswith("RESULT:")]
    if results != ["RESULT: PASS"]:
        shown = results[-1] if results and _RESULT_LINE.match(results[-1]) else ("unrecognised RESULT line" if results else "no RESULT line")
        return "FAILED", f"live catalog evidence is not a match ({shown})"
    if not candidate_sha:
        return "STALE", "cannot bind evidence: candidate commit unknown"
    try:
        checker = _load_checker(root)
        evidence_error, check = checker.EvidenceError, checker.check
        if not (isinstance(evidence_error, type) and issubclass(evidence_error, Exception) and callable(check)):
            raise TypeError("checker API")
    except Exception as exc:
        return "STALE", f"binding checker unavailable ({type(exc).__name__})"
    try:
        note = check(candidate_sha, Path(path), script=root / CATALOG_SCRIPT,
                     max_age_hours=CATALOG_EVIDENCE_MAX_AGE_HOURS)
    except evidence_error as exc:
        return "STALE", f"evidence not bound to candidate: {exc}"
    except Exception as exc:
        return "STALE", f"binding check error ({type(exc).__name__})"
    if not isinstance(note, str) or not note:
        return "STALE", "binding check error (no note)"
    return "PASSED", note


def check_live_catalog_evidence(path, root: Path = ROOT, candidate_sha: str | None = None, unbound: str = "",
                                snapshot: tuple | None = None) -> list[str]:
    changed = "evidence or sidecar changed during the test suite (pre-suite snapshot mismatch)"
    try:
        if snapshot is not None and _evidence_snapshot(path) != snapshot:
            return [changed]
        outcome, reason = catalog_evidence_outcome(path, root, candidate_sha)
        if snapshot is not None and _evidence_snapshot(path) != snapshot:  # bracket the checker's own reads
            return [changed]
    except Exception as exc:  # fail closed with a fixed reason, never a traceback
        return [f"binding check error ({type(exc).__name__})"]
    if outcome == "PASSED":
        print(f"      bound: {_printable(reason)}")
        return []
    return [f"{reason} ({unbound})" if unbound and outcome == "STALE" and not candidate_sha else reason]


def check_frozen(root: Path = ROOT) -> list[str]:
    errs: list[str] = []
    try:
        scope_rec = json.loads((root / "config/scope_router.json").read_text(encoding="utf-8"))["scope_set"]["sha256"]
        trace_rec = json.loads((root / "eval_results/trace_eval_v1.json").read_text(encoding="utf-8"))["questions_sha256"]
    except (OSError, KeyError, ValueError) as exc:
        return [f"cannot read recorded hash: {type(exc).__name__}: {exc}"]
    for rel, rec in (("config/scope_set_v1.json", scope_rec), ("config/trace_questions_v1.json", trace_rec)):
        try:
            actual = sha256_file(root / rel)
        except OSError as exc:
            errs.append(f"{rel}: unreadable ({exc})")
            continue
        if actual != rec:
            errs.append(f"{rel}: sha256 {actual[:12]} != recorded {rec[:12]} (file changed after freeze)")
    return errs


def _is_env_expr(node: ast.AST, tainted: set[str]) -> bool:
    for n in ast.walk(node):
        if isinstance(n, ast.Attribute) and n.attr in ENV_ATTRS:
            return True
        if isinstance(n, ast.Name) and (n.id in tainted or n.id in ENV_ATTRS):
            return True
    return False


def check_no_env_in_widgets(root: Path = ROOT) -> list[str]:
    errs: list[str] = []
    for f in _ui_files(root):
        rel = f.relative_to(root)
        try:
            tree = ast.parse(f.read_text(encoding="utf-8"))
        except SyntaxError as exc:
            errs.append(f"{rel}: cannot parse ({exc})")
            continue
        tainted: set[str] = set()
        for _ in range(3):  # propagate through simple assignments
            for n in ast.walk(tree):
                if isinstance(n, ast.Assign) and _is_env_expr(n.value, tainted):
                    tainted |= {t.id for t in n.targets if isinstance(t, ast.Name)}
        for n in ast.walk(tree):
            if isinstance(n, ast.Call):
                for kw in n.keywords:
                    if kw.arg in ("value", "placeholder", "help", "default") and _is_env_expr(kw.value, tainted):
                        errs.append(f"{rel}:{n.lineno}: environment-derived value passed to widget argument '{kw.arg}'")
    return errs


def check_no_unsafe_html(root: Path = ROOT) -> list[str]:
    errs: list[str] = []
    for f in _ui_files(root):
        rel = str(f.relative_to(root))
        for i, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            if "unsafe_allow_html" in line and not line.lstrip().startswith("#"):
                if not (rel in UNSAFE_HTML_ALLOWED and UNSAFE_HTML_ALLOWED[rel] in line):
                    errs.append(f"{rel}:{i}: unsafe_allow_html not on an allowlisted escaped line")
    return errs


def check_public_default(root: Path = ROOT) -> list[str]:
    sys.path.insert(0, str(root))
    try:
        from vera import public_mode as pm
        if not pm.is_public_mode({}):
            return ["public mode is NOT the default when VERA_PUBLIC_MODE is unset"]
        return []
    except Exception as exc:  # fail verbosely
        return [f"cannot evaluate public-mode default: {type(exc).__name__}: {exc}"]


def check_live_scope(root: Path = ROOT) -> list[str]:
    """Layer 1 only (mode none: no model call, no network, zero cost), using scope_measure's own helpers.

    Runs in-process and writes nothing, so the recorded measurement under tmp/scope_eval/ is never overwritten.
    """
    import importlib.util
    sys.path.insert(0, str(root))
    try:
        spec = importlib.util.spec_from_file_location("scope_measure_gate", root / "scripts" / "scope_measure.py")
        sm = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sm)
        from vera import scope_router as sr
        raw = json.loads(sr.CONFIG_PATH.read_text(encoding="utf-8"))
        doc, _sha = sm.load_set(raw)  # exits (SystemExit) if hash or status is wrong
        cfg = sr.parse_config(raw, layer2_mode="none")
        rows = []
        for item in doc["items"]:
            d = sr.route(item["text"], "none", None, cfg=cfg, chat=None)
            rows.append({"id": item["id"], "label": item["label"], "accepted": d.accepted, "layer": d.layer})
        m = sm.summarise(rows)
    except SystemExit as exc:
        return [f"scope_measure helper exited ({exc.code}); see stderr"]
    except Exception as exc:
        return [f"layer-1 run failed: {type(exc).__name__}: {exc}"]
    print(f"      layer 1 only [directional, small n]: false accepts {m['false_accepts']}/{m['n_out_of_scope']}, "
          f"false refusals {m['false_refusals']}/{m['n_in_scope']} (ids {m['false_refusal_ids']})")
    if m["false_accepts"] > 0:
        return [f"false accepts {m['false_accepts']} > 0 on frozen set (ids {m['false_accept_ids']})"]
    return []


CHILD_OK_RC = 3  # a --child-check run never exits 0, so it can never be mistaken for a gate PASS by exit code
CHILD_TIMEOUT_S = 300
_CHILD_CHECKS = {"c3": "check_public_default", "d": "check_live_scope"}
_CHILD_TAG = "VERA-GATE-CHILD-RESULT"


def _run_child_check(root: Path, name: str) -> list[str]:
    """G-1: run a check that executes candidate-tree code (c3 imports vera.public_mode; d runs scope_measure) in a
    child `python -I` in its own session, killed as a group afterwards. Success needs exit CHILD_OK_RC AND exactly one
    result line carrying this run's nonce (sent on stdin, never in argv or env), so a candidate SystemExit(0) or
    os._exit(0) cannot pass. A deliberately forging child (same trust domain as the candidate) is residual R2."""
    import secrets
    nonce = secrets.token_hex(16)
    tag = f"{_CHILD_TAG} nonce={nonce} errors="
    cmd = [sys.executable, "-I", str(Path(root) / "scripts" / "release_gate.py"), "--child-check", name]
    try:
        p = _run(cmd, cwd=root, timeout=CHILD_TIMEOUT_S, env=_child_env(dict(os.environ)), input=nonce + "\n")
    except subprocess.TimeoutExpired:
        return [f"{name} check exceeded {CHILD_TIMEOUT_S}s"]
    except Exception as exc:
        return [f"{name} check could not start ({type(exc).__name__})"]
    lines = (p.stdout or "").splitlines()
    hits = [ln for ln in lines if ln.startswith(tag)]
    for ln in lines:
        if not ln.startswith(_CHILD_TAG):
            print(f"      child| {_printable(ln)}")  # N-2: never at column 0, never mistakable for RESULT
    if p.returncode != CHILD_OK_RC or len(hits) != 1:
        return [f"{name} check did not prove completion (exit {p.returncode}, {len(hits)} result line(s))"]
    try:
        errs = json.loads(hits[0][len(tag):])
    except ValueError:
        errs = None
    if not isinstance(errs, list) or not all(isinstance(e, str) for e in errs):
        return [f"{name} check returned a malformed result"]
    return [_printable(e) for e in errs]


def run_offline_checks(root: Path = ROOT, run_tests: bool = True, catalog_evidence=None,
                       expect_candidate_sha: str | None = None,
                       isolate_candidate_code: bool | None = None) -> dict[str, list[str]]:
    """main() always isolates candidate-code checks; library callers default to isolating only when run_tests."""
    isolate = run_tests if isolate_candidate_code is None else isolate_candidate_code
    results: dict[str, list[str]] = {}
    if run_tests:
        snapshot = _evidence_snapshot(catalog_evidence)  # G-3: bytes the suite must not change
        pre = _candidate_state(root)
        results["(a) test suite"] = check_tests(root)
        post = _candidate_state(root)
        candidate, unbound = _bound_candidate(pre, post, expect_candidate_sha)
        results["(a2) live catalog evidence (required, owner-run)"] = check_live_catalog_evidence(
            catalog_evidence, root, candidate, unbound, snapshot)
    results["(b) frozen-file integrity"] = check_frozen(root)
    results["(c1) no env key in widgets"] = check_no_env_in_widgets(root)
    results["(c2) no unsafe_allow_html on model text"] = check_no_unsafe_html(root)
    results["(c3) public mode default"] = _run_child_check(root, "c3") if isolate else check_public_default(root)
    return results


def _isolated() -> bool:
    return bool(sys.flags.isolated)


def _main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--live", action="store_true", help="also re-run scope router layer 1 only (zero cost; never default)")
    ap.add_argument("--catalog-evidence", default=None,
                    help="path to the saved output of scripts/live_catalog_check.py --i-confirm-owner-run (owner-run, read-only)")
    ap.add_argument("--skip-tests", action="store_true", help="skip the test suite (diagnostics only; gate is not a pass)")
    ap.add_argument("--expect-candidate-sha", default=None,
                    help="full 40-hex commit the owner intends to gate; (a2) fails unless HEAD equals it")
    ap.add_argument("--child-check", choices=sorted(_CHILD_CHECKS), default=None, help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    if a.child_check:  # internal: one candidate-code check in this child; never prints a RESULT line, never exits 0
        nonce = sys.stdin.readline().strip()
        if not re.fullmatch(r"[0-9a-f]{32}", nonce):
            return 2
        errs = globals()[_CHILD_CHECKS[a.child_check]](ROOT)
        print(f"{_CHILD_TAG} nonce={nonce} errors={json.dumps(list(errs))}", flush=True)
        return CHILD_OK_RC
    print("VERA-DAR release gate")
    if not _isolated():
        print("  FAIL  gate must run as `python -I scripts/release_gate.py` (no script-dir or user-site imports)")
        print("RESULT: FAIL (gate not run isolated)")
        return 1
    if a.skip_tests:
        print("  WARNING: --skip-tests: suite not run; result cannot be a full pass")
    results = run_offline_checks(ROOT, run_tests=not a.skip_tests, catalog_evidence=a.catalog_evidence,
                                 expect_candidate_sha=a.expect_candidate_sha, isolate_candidate_code=True)
    if a.live:
        results["(d) live scope layer 1"] = _run_child_check(ROOT, "d")
    failed = 0
    for name, errs in results.items():
        print(f"  {'PASS' if not errs else 'FAIL'}  {name}")
        for e in errs:
            print(f"      - {_printable(e)}")
        failed += bool(errs)
    failed += bool(a.skip_tests)
    print(f"RESULT: {'FAIL' if failed else 'PASS'} ({len(results)} checks, {failed} failed)")
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    """G-1: no exception, SystemExit or KeyboardInterrupt from any check (candidate code included) can end the gate
    with exit 0 or without a RESULT line. OD-7 link 2 must still require the exact last line
    `RESULT: PASS (N checks, 0 failed)`, because os._exit() in this process cannot be caught."""
    try:
        return _main(argv)
    except BaseException as exc:  # noqa: B036 - deliberate: fail closed on everything
        try:
            print(f"RESULT: FAIL (gate aborted: {type(exc).__name__})", flush=True)
        except BaseException:
            pass
        return 1


if __name__ == "__main__":
    try:  # INT-1 M-6: G-1 also covers the RB-4 launch self-check (fail closed with a RESULT line)
        _refused = launch_refusals(os.environ, sys.flags)
    except BaseException as _exc:  # noqa: B036
        _refused = [f"launch self-check error ({type(_exc).__name__})"]
    if _refused:
        print("VERA-DAR release gate")
        for _r in _refused:
            print(f"  REFUSED  {_r}")
        print("RESULT: FAIL (gate refused to start)")
        sys.exit(1)
    sys.exit(main())

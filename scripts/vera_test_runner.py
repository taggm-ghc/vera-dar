"""VERA-DAR sanctioned test runner (K18W INT2, F-1/F-4/F-6). Stdlib only; run it as

    <venv>/bin/python -I scripts/vera_test_runner.py -- tests/test_x.py [-q] [-k EXPR] ...

It is the ONLY supported way to run the test suite outside the release gate (tests/vera_test_boundary.py refuses a
session that was not launched by this runner, the gate or the documented offline-only escape). Adapted from the K17W
W6 launcher design (scripts/vera_launcher.py r2c, design 5-r2.B: fixed absolute binaries B-1, tmpfs masks B-3,
credential masks B-4, allowlisted env B-5, refusals B-9) and from the K18W cleared sandbox (v8c_sandbox.sh). Unlike
W6 r2c it EXECUTES the launch; the attestation side of W6 (A-1..A-5) is not wired here (it stays a W7/W8 item).

What the kernel enforces for the pytest process and every child (user + net + mount + ipc + pid namespaces):
  * network: a new network namespace with only `lo` (no route, no DNS);
  * host sockets: tmpfs over /tmp /run /dev/shm /mnt /var/tmp (and /var/crash) and over a real (non-symlink)
    /var/run/postgresql, so no host Unix socket (a live Postgres at /var/run/postgresql/.s.PGSQL.5432, container
    runtimes, the user bus, WSL interop) is reachable; /dev/vsock is bound over with /dev/null;
  * credentials: tmpfs over /home (every ~/.pgpass, ~/.ssh, the VERA-DAR and AI-Internship .env files), with only the
    checkout (rw), its git directory (rw) and the venv (ro) bound back; credential-type files inside the checkout
    (.env*, *.env, *.pem, *.key, .pgpass ...; never *.example) are bound over with /dev/null;
  * environment: env -i with a fixed allowlist (PATH, HOME, LANG, TMPDIR, the runner marker, a validated gate nonce);
    /proc/self/environ of the test process therefore holds no credential;
  * argv: `python -I -B -X pycache_prefix=<tmpfs> -m pytest -c <root>/pytest.ini --rootdir <root>
    -p no:cacheprovider <validated args>`; a planted ini/conftest-disabling flag/config file is refused (F-4) and no
    bytecode from the checkout's __pycache__ directories is ever read.
Honest limits: same-uid code can still edit this file or the checkout before launch (accident-class control, not
authentication); the binaries are checked for root ownership (B-1) but not hashed; seccomp is installed by the
boundary inside the session, not by this runner.
"""
from __future__ import annotations

import fnmatch
import os
import re
import shlex
import stat
import sys

FIXED_PATH = "/usr/bin:/bin"
UNSHARE, SH, MOUNT, IP, MKDIR, ENV = "/usr/bin/unshare", "/bin/sh", "/usr/bin/mount", "/usr/bin/ip", "/usr/bin/mkdir", \
    "/usr/bin/env"
FIND = "/usr/bin/find"                     # INT3 MED-B: the in-namespace socket sweep (GNU find: -readable, -quit)
LAUNCH_BINARIES = (UNSHARE, SH, MOUNT, IP, MKDIR, ENV, FIND)
EXIT_REFUSED, EXIT_ISOLATION = 64, 78
RUNNER_MARKER = "VERA_TEST_RUNNER"
TMPFS_TARGETS = ("/tmp", "/run", "/dev/shm", "/var/tmp", "/var/crash")  # nosec B108 - tmpfs mount targets inside the runner's private mount namespace, not host /tmp
DB_SOCKET_DIRS = ("/var/run/postgresql", "/run/postgresql")
STAGE = "/mnt/.vera-stage"
# WSL (orchestrator item, INT2): Windows drives (/mnt/c ...) and other host-shared filesystems (9p, drvfs, virtiofs,
# e.g. /usr/lib/wsl/drivers or a 9p mount under /home) hold Windows executables; an .exe started through the kernel's
# WSLInterop binfmt handler would run on the Windows host OUTSIDE the network namespace. The runner hides /mnt and
# /home (tmpfs) and every other foreign mount, hides the binfmt_misc entry if it is visible, clears WSL_* names
# (env -i) and hides /run (interop sockets /run/WSL). The binfmt handler itself is kernel-global and cannot be
# disabled without privilege in the initial namespace: stated residual. No Windows binary is ever executed here.
FOREIGN_FSTYPES = ("9p", "drvfs", "virtiofs", "fuse.drvfs")
BINFMT_INTEROP = "/proc/sys/fs/binfmt_misc/WSLInterop"
SANDBOX_HOME = "/tmp/vera-home"  # nosec B108 - on the in-namespace tmpfs /tmp
PYCACHE = "/tmp/vera-pycache"  # nosec B108 - on the in-namespace tmpfs /tmp
CRED_MASK_GLOBS = (".env", ".env.*", "*.env", "*.env.*", "*-prv.env", "*secret*", "*.pem", "*.key", ".pgpass",
                   ".pg_service.conf", ".env.db-accounts", ".env-*", "*.env-*")
# E1 (PLAN-ENV v2.1): ".env-*" / "*.env-*" cover a misplaced .env-vera / .env-intern. Mirrored by
# tests/vera_test_boundary.py CRED_FILE_GLOBS (E1b/C10, equality asserted in tests/test_env_guard.py).
CRED_MASK_EXCLUDE = ("*.example",)
# E1 R-G1 preflight: the env files live ONLY in R/config/env (R = the checkout's parent). A non-example .env* file, or a
# config/env directory, inside R/vera or R/course-service is refused before launch (names only, never content).
RG1_TREES = ("course-service",)                  # besides the checkout itself (R/vera)
RG1_FILE_GLOB = ".env*"
# INT4 (WS-8h LOW-MED): no directory NAME is skipped below the checkout root any more (a planted conftest or credential
# file under tests/node_modules, tests/venv, tests/.venv or __pycache__ was invisible to both scans). Skipped: the
# root .git directory, the explicitly pinned venv (sys.prefix, by real path) and, for the planted-config scan only,
# venv/node_modules directories directly under the root (outside tests/, so never on a collection path: every
# path argument must resolve under tests/).
ROOT_ONLY_SKIP = (".git",)
ROOT_ONLY_PLANTED_SKIP = (".git", ".venv", "venv", "node_modules")
PYTEST_CONFIG_NAMES = ("pytest.ini", ".pytest.ini", "tox.ini", "setup.cfg", "pyproject.toml")
STARTUP_HOOKS = ("sitecustomize.py", "usercustomize.py")
REFUSED_ENV_PREFIXES = ("LD_",)
# INT4 (WS-8h MED): pytest reads these from the environment (extra argv / plugin modules to import). env -i already
# keeps them out of the sandbox; the runner refuses them outright so a caller cannot believe they took effect.
REFUSED_ENV_NAMES = ("PYTEST_ADDOPTS", "PYTEST_PLUGINS")
# ALLOWLIST of caller options (INT4, WS-8h MED). Only these exact spellings are accepted; every other argument that
# starts with '-' is refused, so no pytest 9.1.1 option can import a module or change the configuration:
# -p, -c/--config-file, --rootdir, --confcutdir, --noconftest, --pyargs, -o/--override-ini, -W/--pythonwarnings
# (an `action::module.Category` filter imports `module`), --import-mode, --debug, --trace-config, --run-live-stages,
# --basetemp outside /tmp, argparse abbreviations (`--override`, `--conf`) and clustered short options (`-qp x`).
# INT3 MED-C: every argument starting with '@' is refused (argparse fromfile_prefix_chars, WS-8g GE3), and a path
# argument or --ignore/--deselect value that passes through a symlink is refused. Each value is checked too: it may
# not start with '-' or '@', and --tb/--maxfail/--durations/-r values must match their pattern.
FLAG_OPTS = {"-q", "-qq", "-v", "-vv", "-x", "--exitfirst", "-l", "--strict-markers", "--co", "--collect-only"}
VALUE_OPTS = {"-k", "-m", "--deselect", "--basetemp", "--maxfail", "--tb", "--durations", "-r", "--ignore"}
ATTACHED_SHORT = ("-r", "-k", "-m")                 # the short value options that may carry an attached value (-rfE)
VALUE_PATTERNS = {"--tb": re.compile(r"auto|long|short|line|native|no"), "--maxfail": re.compile(r"[0-9]{1,6}"),
                  "--durations": re.compile(r"[0-9]{1,6}"), "-r": re.compile(r"[fEsxXpPaAwN]{1,16}")}
# Named here only to give a clear message (they would be refused as "not allowed" anyway).
NAMED_REFUSALS = {"-c": "the runner pins it", "--config-file": "the runner pins it", "--rootdir": "the runner pins it",
                  "-p": "the runner pins it", "--noconftest": "the runner pins it", "--confcutdir": "the runner pins it",
                  "-W": "a warning filter can import a module", "--pythonwarnings": "a warning filter can import a module",
                  "-o": "it overrides the pinned ini", "--override-ini": "it overrides the pinned ini"}
_HEX32 = re.compile(r"[0-9a-f]{32}")


def _under(p, base):
    p, base = os.path.normpath(p), os.path.normpath(base)
    return p == base or p.startswith(base.rstrip("/") + "/")


# ---- B-1 (from W6): trusted binaries --------------------------------------------------------------------------
def _root_unmapped():
    """True inside a user namespace whose uid_map does not map uid 0 (then root-owned files show as the overflow
    uid). Unreadable map: False (strict)."""
    try:
        with open("/proc/self/uid_map", encoding="ascii") as fh:
            rows = [ln.split() for ln in fh if ln.strip()]
        return not any(int(r[0]) <= 0 < int(r[0]) + int(r[2]) for r in rows if len(r) == 3)
    except (OSError, ValueError):
        return False


def _overflow_uid():
    try:
        with open("/proc/sys/kernel/overflowuid", encoding="ascii") as fh:
            return int(fh.read().strip())
    except (OSError, ValueError):
        return None


def _root_like(uid):
    """Root, or (only when root is unmapped in this user namespace, e.g. a nested sandbox) the overflow uid."""
    return uid == 0 or (uid == _overflow_uid() and _root_unmapped())


def trusted_binary(path, lstat=os.lstat, realpath=os.path.realpath):
    """Problems (empty = trusted): absolute; resolved target a regular file; target and every parent root-owned and
    not group/world-writable."""
    if not os.path.isabs(path):
        return [f"{path}: not an absolute path"]
    problems, p = [], realpath(path)
    first = True
    while True:
        try:
            st = lstat(p)
        except OSError as e:
            return problems + [f"{p}: cannot stat ({type(e).__name__})"]
        if not _root_like(st.st_uid):
            problems.append(f"{p}: not root-owned")
        if st.st_mode & (stat.S_IWGRP | stat.S_IWOTH) and not stat.S_ISLNK(st.st_mode):
            problems.append(f"{p}: group/world-writable")
        if first and not stat.S_ISREG(st.st_mode):
            problems.append(f"{p}: not a regular file")
        first = False
        parent = os.path.dirname(p)
        if parent == p:
            return problems
        p = parent


# ---- F-4: argv and planted configuration ----------------------------------------------------------------------
def _through_symlink(root, rel):
    """True when <root>/<rel> (normalised, not resolved) differs from its realpath: some component is a symlink."""
    p = os.path.normpath(os.path.join(root, rel))
    return os.path.realpath(p) != p


def validate_args(args, root):
    """(pytest args to pass, refusals). Paths must resolve under <root>/tests without a symlink; options from the
    allowlist only; no '@' argument file (INT3 MED-C)."""
    root = os.path.realpath(root)
    tests = os.path.join(root, "tests")
    out, bad, paths, i = [], [], [], 0
    at = [a for a in args if isinstance(a, str) and a.startswith("@")]
    if at:
        bad.append("argument files ('@...') are not allowed")
    while i < len(args):
        a = args[i]
        name, eq, val = a.partition("=") if a.startswith("--") else (a, "", "")
        if a.startswith("@"):
            i += 1
            continue                                     # refused above (once)
        short = a[:2] if a.startswith("-") and not a.startswith("--") else None
        named = name if name in NAMED_REFUSALS else (short if short in NAMED_REFUSALS else None)
        if named:
            bad.append(f"option not allowed: {named} ({NAMED_REFUSALS[named]})")
            i += 1
            continue
        if a in FLAG_OPTS:
            out.append(a)
            i += 1
            continue
        attached = short in ATTACHED_SHORT and len(a) > 2
        if (name in VALUE_OPTS and (eq or a == name)) or attached:
            if attached:
                opt, v, step = short, a[2:], 1
            elif eq:
                opt, v, step = name, val, 1
            elif i + 1 < len(args):
                opt, v, step = name, args[i + 1], 2
            else:
                bad.append(f"{a}: missing value")
                break
            if v.startswith("@"):
                bad.append(f"{opt}: a value starting with '@' is not allowed")
            elif v.startswith("-") or not v or "\0" in v or "\n" in v:
                bad.append(f"{opt}: value not allowed")
            elif opt in VALUE_PATTERNS and not VALUE_PATTERNS[opt].fullmatch(v):
                bad.append(f"{opt}: value not allowed")
            elif opt in ("--ignore", "--deselect") and (
                    not _under(os.path.realpath(os.path.join(root, v.split("::")[0])), tests)
                    or _through_symlink(root, v.split("::")[0])):
                bad.append(f"{opt}: outside tests/ or through a symlink")
            elif opt == "--basetemp" and not (os.path.isabs(v) and _under(v, "/tmp")):  # nosec B108 - in-namespace tmpfs
                bad.append("--basetemp must be under /tmp (the sandbox tmpfs)")
            else:
                out += [opt, v] if step == 2 or opt.startswith("--") else [opt + v]
            i += step
            continue
        if a.startswith("-"):
            bad.append(f"option not allowed: {name}")
            i += 1
            continue
        rel = a.split("::", 1)[0]
        target = os.path.realpath(os.path.join(root, rel))
        if not _under(target, tests):
            bad.append(f"path outside tests/: {a}")
        elif _through_symlink(root, rel):
            bad.append(f"path through a symlink: {a}")
        out.append(a)
        paths.append(a)
        i += 1
    if not paths:
        bad.append("no test path given (pass tests or tests/<file>)")
    return out, bad


def _pinned_venv():
    """Real path of the running venv (sys.prefix) or None outside a venv: the ONLY directory skipped by name-free scans."""
    return os.path.realpath(sys.prefix) if os.path.realpath(sys.prefix) != os.path.realpath(sys.base_prefix) else None


def _keep_dirs(d, dirs, root, venv, root_skip):
    """INT4: the subdirectories of <d> to walk. Only <root>/<name in root_skip> and the pinned venv are skipped."""
    return [x for x in dirs if not ((d == root and x in root_skip)
                                    or (venv and os.path.realpath(os.path.join(d, x)) == venv))]


def planted_configs(root, walk=os.walk, venv=None):
    """Paths (relative) of pytest config files, startup hooks or .pth files that could change collection. Only the
    tracked root pytest.ini is expected; anything else under the root is refused. Skipped (INT4): the root .git, the
    root-level venv/node_modules dirs (outside tests/) and the pinned venv; nothing inside tests/ is skipped."""
    root = os.path.realpath(root)
    venv = os.path.realpath(venv) if venv else None
    found = []
    for d, dirs, files in walk(root):
        dirs[:] = _keep_dirs(d, dirs, root, venv, ROOT_ONLY_PLANTED_SKIP)
        rel = os.path.relpath(d, root)
        for x in dirs:                                   # INT3 MED-C: os.walk does not descend symlinked dirs
            if os.path.islink(os.path.join(d, x)):
                found.append(("symlink: " + x) if rel == "." else "symlink: " + os.path.join(rel, x))
        for f in files:
            if os.path.islink(os.path.join(d, f)):
                found.append("symlink: " + (f if rel == "." else os.path.join(rel, f)))
                continue
            r = f if rel == "." else os.path.join(rel, f)
            if f in PYTEST_CONFIG_NAMES and r != "pytest.ini" and (rel != "." or f in ("pytest.ini", ".pytest.ini", "tox.ini")):
                found.append(r)
            elif f in STARTUP_HOOKS or f.endswith(".pth"):
                found.append(r)
            elif f == "conftest.py" and r not in ("conftest.py", "tests/conftest.py", "db/conftest.py",
                                                    "scripts/conftest.py"):
                found.append(r)
    if not os.path.isfile(os.path.join(root, "pytest.ini")):
        found.append("pytest.ini (tracked root config) missing")
    pp = os.path.join(root, "pyproject.toml")
    try:
        with open(pp, encoding="utf-8") as fh:
            if "[tool.pytest" in fh.read():
                found.append("pyproject.toml [tool.pytest*]")
    except OSError:
        pass
    return sorted(found)


def credential_files(root, walk=os.walk, venv=None):
    """Credential-type files to mask. INT4: every directory is walked except the root .git and the pinned venv."""
    out, root = [], os.path.realpath(root)
    venv = os.path.realpath(venv) if venv else None
    for d, dirs, files in walk(root):
        dirs[:] = _keep_dirs(d, dirs, root, venv, ROOT_ONLY_SKIP)
        for f in files:
            if any(fnmatch.fnmatchcase(f, x) for x in CRED_MASK_EXCLUDE):
                continue
            if any(fnmatch.fnmatchcase(f, g) for g in CRED_MASK_GLOBS):
                out.append(os.path.join(d, f))
    return sorted(out)


def rg1_findings(root, walk=os.walk, venv=None):
    """E1 R-G1: refusal reasons (relative paths, never content) for a non-example `.env*` file or a `config/env`
    directory inside the checkout (root = R/vera) or R/course-service. Only names are inspected (os.walk); no file is
    opened. The checkout's root .git and the pinned venv are skipped, as in credential_files()."""
    root = os.path.realpath(root)
    base = os.path.dirname(root)
    venv = os.path.realpath(venv) if venv else None
    out = []
    for tree in [root] + [os.path.join(base, t) for t in RG1_TREES]:
        if not os.path.isdir(tree):
            continue
        label = os.path.relpath(tree, base)
        for d, dirs, files in walk(tree):
            dirs[:] = _keep_dirs(d, dirs, tree, venv, ROOT_ONLY_SKIP)
            rel = os.path.relpath(d, tree)
            if "env" in dirs and os.path.basename(d) == "config":
                out.append(f"R-G1: a config/env directory inside {label}: {os.path.join(label, rel, 'env')}")
            for f in files:
                if fnmatch.fnmatchcase(f, RG1_FILE_GLOB) and not any(fnmatch.fnmatchcase(f, x)
                                                                   for x in CRED_MASK_EXCLUDE):
                    out.append(f"R-G1: an env file inside {label} (it belongs in R/config/env): "
                               f"{os.path.normpath(os.path.join(label, rel, f))}")
    return sorted(out)


# ---- B-5: environment -----------------------------------------------------------------------------------------
def sandbox_env(environ):
    env = {"PATH": FIXED_PATH, "HOME": SANDBOX_HOME, "LANG": "C.UTF-8", "TMPDIR": "/tmp",  # nosec B108 - in-namespace tmpfs
           "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1", "PYTHONDONTWRITEBYTECODE": "1",
           "PYTHONPYCACHEPREFIX": PYCACHE, RUNNER_MARKER: "1"}
    nonce = environ.get("VERA_GATE_NONCE") or ""
    if _HEX32.fullmatch(nonce):
        env["VERA_GATE_NONCE"] = nonce
    if environ.get("USER") and re.fullmatch(r"[A-Za-z0-9._-]{1,64}", environ["USER"]):
        env["USER"] = environ["USER"]
    return env


def launch_refusals(environ, isolated):
    out = []
    if not isolated:
        out.append("the runner must run under python -I")
    bad = sorted(k for k in environ if k.startswith(REFUSED_ENV_PREFIXES) or k in REFUSED_ENV_NAMES)
    if bad:
        out.append("refused variable names: " + ", ".join(bad))
    return out


# ---- git directories the boundary writes to (child bootstrap) or reads ------------------------------------------
def git_dirs(root):
    """[(path, rw)] for a checkout whose .git is a FILE (linked worktree): its private git dir (rw) and the common
    dir (ro). A plain .git directory is inside the root bind already: []."""
    g = os.path.join(root, ".git")
    if not os.path.isfile(g):
        return []
    try:
        with open(g, encoding="utf-8") as fh:
            line = fh.readline().strip()
    except OSError:
        return []
    if not line.startswith("gitdir:"):
        return []
    gd = os.path.realpath(os.path.join(root, line[7:].strip()))
    out = [(gd, True)]
    try:
        with open(os.path.join(gd, "commondir"), encoding="utf-8") as fh:
            common = os.path.realpath(os.path.join(gd, fh.readline().strip()))
        if not _under(gd, common):
            out.append((common, False))
        else:
            out.insert(0, (common, False))
    except OSError:
        pass
    return out


# ---- B-3/B-4: the namespace script ------------------------------------------------------------------------------
def sandbox_script(root, venv, gitdirs, cred_files, uid, gid, py, pytest_args, env, path_exists=os.path.exists,
                   realpath=os.path.realpath, foreign=()):
    """Shell text for `/bin/sh -c` (every path shlex-quoted, every tool absolute). Stages the binds on a tmpfs /mnt
    BEFORE /home is hidden, masks, moves the binds back (shallowest first), then drops to the real uid in a nested
    user namespace and execs pytest with env -i."""
    if not (isinstance(uid, int) and uid > 0 and isinstance(gid, int) and gid >= 0):
        raise ValueError("uid/gid: run as a non-root user")
    q = shlex.quote
    binds = [(root, True)] + list(gitdirs) + [(venv, False)]
    binds = sorted(dict(binds).items(), key=lambda kv: kv[0].count("/"))
    for p, _rw in binds:
        if not os.path.isabs(p) or "\n" in p or any(_under(p, t) for t in TMPFS_TARGETS + ("/mnt",)):
            raise ValueError(f"bind source not usable: {p}")
    L = ['fail() { echo "test isolation: runner sandbox setup failed: $1" >&2; exit %d; }' % EXIT_ISOLATION,
         "PATH=" + q(FIXED_PATH) + "; export PATH",
         f"{IP} link set lo up || fail lo",
         f"{MOUNT} -n --make-rprivate / || fail private",
         f"{MOUNT} -n -t tmpfs -o mode=0700 tmpfs /mnt || fail mnt-stage"]
    for n, (src, _rw) in enumerate(binds):
        L += [f"{MKDIR} -p {q(STAGE + str(n))} || fail mkdir-stage",
              f"{MOUNT} -n --bind {q(src)} {q(STAGE + str(n))} || fail bind-stage"]
    L.append(f"{MOUNT} -n -t tmpfs -o mode=0755 tmpfs /home || fail home")
    for n, (dst, rw) in enumerate(binds):
        L += [f"{MKDIR} -p {q(dst)} || fail mkdir-target",
              f"{MOUNT} -n --move {q(STAGE + str(n))} {q(dst)} || fail move"]
        if not rw:
            L.append(f"{MOUNT} -n -o remount,bind,ro {q(dst)} || fail ro")
    L.append(f"{MOUNT} -n -t tmpfs -o mode=0755 tmpfs /mnt || fail mnt")
    for t in TMPFS_TARGETS:
        if path_exists(t):
            L.append(f"{MOUNT} -n -t tmpfs -o mode=1777 tmpfs {q(t)} || fail {q('tmpfs ' + t)}")
    for d in DB_SOCKET_DIRS:                  # masked too unless it resolves under a tmpfs target (/var/run -> /run)
        if path_exists(d) and not any(_under(realpath(d), t) for t in TMPFS_TARGETS):
            L.append(f"{MOUNT} -n -t tmpfs -o mode=0755 tmpfs {q(d)} || fail db-socket-dir")
    for m in foreign:                         # WSL: host-shared mounts outside /home and /mnt
        L.append(f"{MOUNT} -n -t tmpfs -o mode=0755 tmpfs {q(m)} || fail foreign-mount")
    L.append(f"if [ -e {BINFMT_INTEROP} ]; then {MOUNT} -n --bind /dev/null {BINFMT_INTEROP} || fail binfmt; fi")
    if path_exists("/dev/vsock"):
        L.append(f"{MOUNT} -n --bind /dev/null /dev/vsock || fail vsock")
    for f in cred_files:
        L.append(f"if [ -e {q(f)} ]; then {MOUNT} -n --bind /dev/null {q(f)} || fail cred-mask; fi")
    # INT3 MED-B: after the fixed masks, sweep the WHOLE visible tree (not only the fixed dirs or the .s.PGSQL.* name;
    # WS-8g GC2 reached a socket in the checkout). (1) tmpfs over every directory this uid cannot list or traverse
    # (a socket under it could still be reachable by a known name, e.g. /var/lib/postgresql); (2) /dev/null over
    # every Unix socket file of any name, then tmpfs over the parent dir of any survivor; (3) re-sweep and fail if any
    # unlistable directory or socket remains.
    # /proc and /sys hold no socket files and are pruned. find never opens or connects to a socket (lstat only).
    prune = f"{FIND} / \\( -path /proc -o -path /sys \\) -prune -o"
    L += [f"{prune} -type d \\( ! -readable -o ! -executable \\) -prune "
          f"-exec {MOUNT} -n -t tmpfs -o mode=0755 tmpfs {{}} \\; 2>/dev/null",
          # each socket: /dev/null over it; if the kernel refuses that bind, tmpfs over its parent directory instead
          f"{prune} -type s -exec {SH} -c '{MOUNT} -n --bind /dev/null \"$1\" || {{ echo \"vera_test_runner: "
          f"socket-sweep: bind refused, parent directory masked\" >&2; {MOUNT} -n -t tmpfs -o mode=0755 tmpfs "
          f"\"${{1%/*}}\"; }}' sh {{}} \\;",
          # INT3 run 2: a socket can still show as S_ISSOCK after a successful /dev/null bind (observed, root cause
          # not established); a second pass hides the parent directory of every survivor with tmpfs (unambiguous)
          f"n_s=$({prune} -type s -print 2>/dev/null | wc -l)",
          f'[ "$n_s" = 0 ] || echo "vera_test_runner: socket-sweep: $n_s socket(s) survived the bind pass; parent '
          f'directories masked" >&2',
          f"{prune} -type s -exec {SH} -c '{MOUNT} -n -t tmpfs -o mode=0755 tmpfs \"${{1%/*}}\"' sh {{}} \\;",
          f"left_s=$({prune} -type s -print 2>/dev/null | wc -l)",
          f"left_d=$({prune} -type d \\( ! -readable -o ! -executable \\) -print 2>/dev/null | wc -l)",
          '[ "$left_s" = 0 ] && [ "$left_d" = 0 ] || fail "socket-sweep (left: $left_s socket(s), $left_d unlistable dir(s))"']
    L += [f"{MKDIR} -p {SANDBOX_HOME} {PYCACHE} || fail home-dir",
          f"cd {q(root)} || fail cd",
          "unset WSL_INTEROP WSL_DISTRO_NAME WSLENV"]
    envs = " ".join(q(f"{k}={v}") for k, v in sorted(env.items()))
    cmd = [py, "-I", "-B", "-X", f"pycache_prefix={PYCACHE}", "-m", "pytest", "-c", os.path.join(root, "pytest.ini"),
           "--rootdir", root, "-p", "no:cacheprovider", *pytest_args]
    L.append(f"exec {UNSHARE} --user --map-user={uid} --map-group={gid} -- {ENV} -i {envs} "
             + " ".join(q(c) for c in cmd) + " || fail exec")
    return "\n".join(L) + "\n"


def _unescape(s):
    return re.sub(r"\\([0-7]{3})", lambda m: chr(int(m.group(1), 8)), s)


def foreign_mounts(mountinfo_text):
    """Mountpoints whose filesystem type is host-shared (FOREIGN_FSTYPES), from /proc/self/mountinfo text."""
    out = []
    for line in (mountinfo_text or "").splitlines():
        f = line.split()
        if "-" not in f[6:]:
            continue
        i = f.index("-", 6)
        if len(f) > i + 1 and f[i + 1] in FOREIGN_FSTYPES:
            out.append(_unescape(f[4]))
    return sorted(set(out))


def foreign_masks(foreign, binds):
    """(masks, refusals): foreign mountpoints the script must hide (those not already under /home, /mnt or a tmpfs
    target), and a refusal for each bind source that lives on a foreign mount (it would be hidden or is Windows-side)."""
    hidden = TMPFS_TARGETS + ("/home", "/mnt")
    masks, bad = [], []
    for m in foreign:
        if any(_under(src, m) for src in binds):
            bad.append(f"bind source on a host-shared (9p/drvfs) mount: {m}")
        elif not any(_under(m, t) for t in hidden):
            masks.append(m)
    return masks, bad


def unshare_argv(script):
    return [UNSHARE, "--user", "--map-root-user", "--net", "--mount", "--ipc", "--pid", "--fork", "--kill-child",
            "--mount-proc", "--", SH, "-c", script, "sh"]


def plan(argv, environ, root=None, isolated=None, uid=None, gid=None):
    """Pure-ish: ('run', argv_for_execve) or ('refuse', [reasons]). Reads the checkout (configs, credential names)."""
    root = os.path.realpath(root or os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    isolated = bool(sys.flags.isolated) if isolated is None else isolated
    args = list(argv)
    if args[:1] == ["--"]:
        args = args[1:]
    reasons = launch_refusals(environ, isolated)
    pyargs, bad = validate_args(args, root)
    reasons += bad
    reasons += [f"planted or unexpected config: {p}" for p in planted_configs(root, venv=_pinned_venv())]
    reasons += rg1_findings(root, venv=_pinned_venv())
    for b in LAUNCH_BINARIES:
        reasons += trusted_binary(b)
    if reasons:
        return "refuse", reasons
    venv = os.path.realpath(sys.prefix)
    if venv == os.path.realpath(sys.base_prefix):
        return "refuse", ["run the runner with the project venv's python (sys.prefix is not a venv)"]
    py = os.path.join(venv, "bin", "python")
    gd = git_dirs(root)
    try:
        with open("/proc/self/mountinfo", encoding="utf-8") as fh:
            masks, bad = foreign_masks(foreign_mounts(fh.read()), [root, venv] + [p for p, _rw in gd])
    except OSError:
        return "refuse", ["cannot read /proc/self/mountinfo (foreign-mount check)"]
    if bad:
        return "refuse", bad
    script = sandbox_script(root, venv, gd, credential_files(root, venv=venv), os.getuid() if uid is None else uid,
                            os.getgid() if gid is None else gid, py, pyargs, sandbox_env(environ), foreign=masks)
    return "run", unshare_argv(script)


def main(argv=None):
    kind, val = plan(sys.argv[1:] if argv is None else argv, os.environ)
    if kind == "refuse":
        for r in val:
            os.write(2, ("vera_test_runner: refused: " + r + "\n").encode())
        return EXIT_REFUSED
    try:
        os.execve(UNSHARE, val, {"PATH": FIXED_PATH})            # env -i: nothing of the caller's env crosses
    except OSError as e:
        os.write(2, f"vera_test_runner: cannot start the sandbox ({type(e).__name__})\n".encode())
        return EXIT_ISOLATION


if __name__ == "__main__":
    sys.exit(main())

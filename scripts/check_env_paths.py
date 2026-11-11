#!/usr/bin/env python3
"""ENV-PATHS-1110: deterministic lint for env-sourced paths that reach a file sink without vera.safe_path.

This is a lint, sound only for the patterns listed below; vera/safe_path.py is the runtime control. The output always prints the
UNANALYSABLE and UNKNOWN counts and never says "clean".

Verdicts per site (sorted `file:line VERDICT kind env`):
  GUARDED       the sink argument is the safe_path() return value (data flow), or a safe_path() call receives a tainted value.
  UNGUARDED     a value derived from an env read / config loader reaches a file sink without passing through safe_path().
  UNANALYSABLE  getattr/setattr/vars/eval/exec/__import__ fed by a tainted value, os.environ passed whole, shell eval or
                indirect expansion. Fails the check unless exempted.
Exit status is non-zero on any non-exempt UNGUARDED or UNANALYSABLE site, or any exemption problem (format, expiry, stale, hash).

Known limits (false-negative classes, plan section D):
  1 aliasing: assignment propagation within one function/module scope, flow-insensitive, fail closed (merge = tainted).
  2 dict/config lookups: tainted only from the pinned CONFIG_LOADERS; other lookups feeding a sink are counted as UNKNOWN.
  3 f-strings, +, %, .format, os.path.join: taint propagates through every expression kind.
  4 getattr/**kwargs/setattr/vars/eval/exec: UNANALYSABLE when fed by taint (eval and exec included).
  5 cross-function: a function whose return value is tainted is a source for its callers (matched by name, iterated to a
    fixpoint). A tainted value passed INTO a function parameter is NOT followed (limit).
  6 whole os.environ: UNANALYSABLE only when handed to a subprocess-family call (os.environ, dict(os.environ), os.environ.copy(),
    {**os.environ}). Callees that take a Mapping are covered only when their parameter is named `environ` or defaults to
    os.environ (it is then treated as the environment). Any other whole-environ hand-off (other parameter names, containers,
    attribute stores) is NOT flagged.
  7 sinks via aliases: import aliases and module-level `name = module.attr` aliases are resolved; the sink list is SINKS and
    SINK_ATTRS below. A DSN-style call with a string constant holding sslrootcert/sslcert/sslkey/passfile is UNGUARDED.
  8 shell: grep for $VAR / ${VAR} as a path operand (redirect, cat/cp/mv/rm/tee/source/./mkdir/touch/ln); eval and ${!x}
    are UNANALYSABLE. Variables derived from other variables are missed.
Not followed at all: whole-environ hand-offs other than class 6 above, attribute stores (self.x = tainted), tainted data passed into callee parameters, tests/ (by design),
files whose name starts with .env and any directory named env (never opened; skipped by name).
"""
from __future__ import annotations

import argparse
import ast
import datetime as dt
import hashlib
import json
import re
import sys
from pathlib import Path

CONFIG_LOADERS = ("vera.env_precedence.load_config", "env_precedence.load_config")  # pinned (resolved dotted names): taint dict lookups
ENVIRON_PARAMS = {"environ"}  # a parameter with this name, or whose default is os.environ, is treated as os.environ
SUBPROCESS_CALLS = ("subprocess.", "os.system", "os.exec", "os.spawn", "os.popen")
SINKS = {
    "open", "io.open", "codecs.open", "os.open", "os.scandir", "os.listdir", "os.remove", "os.unlink", "os.rename", "os.replace",
    "os.makedirs", "os.mkdir", "os.rmdir", "os.walk", "os.chdir", "glob.glob", "glob.iglob", "shutil.copy", "shutil.copy2",
    "shutil.copyfile", "shutil.copytree", "shutil.move", "shutil.rmtree", "importlib.import_module",
    "importlib.util.spec_from_file_location", "sqlite3.connect", "ssl.create_default_context", "runpy.run_path",
    "os.system", "subprocess.run", "subprocess.Popen", "subprocess.call", "subprocess.check_call", "subprocess.check_output",
    "vera.safe_path.append_text",
}
SINK_ATTRS = {"read_text", "write_text", "read_bytes", "write_bytes", "open", "mkdir", "unlink", "touch", "iterdir", "rglob",
              "rename", "rmdir"}  # str.replace / str.glob-like names are left out on purpose: they collide with string methods
TWO_PATH_SINKS = {"shutil.copy", "shutil.copy2", "shutil.copyfile", "shutil.copytree", "shutil.move", "os.rename", "os.replace"}
PATH_KW = {"path", "file", "filename", "fname", "src", "dst", "cafile", "capath", "name", "dir", "directory"}
DSN_CALLS = {"create_engine", "sqlalchemy.create_engine", "psycopg.connect", "psycopg2.connect", "connect"}
DSN_FILE_PARAM = re.compile(r"(sslrootcert|sslcert|sslkey|passfile|file://)", re.I)
DYNAMIC = {"getattr", "setattr", "vars", "eval", "exec", "__import__", "compile"}
GUARD_NAMES = {"safe_path", "vera.safe_path.safe_path"}
REASONS = {"selector", "denied-name", "db-url-not-file", "test-only-runner-set"}
PY_ROOTS = ("vera", "scripts", "main.py", "pages", "streamlit_app.py")
SH_FILES = ("start.sh", "run.sh", "run_demo.sh")
SKIP_DIRS = {"__pycache__", "tests", "env", ".git", "node_modules"}
T, G, N = 1, 2, 0  # tainted dominates guarded dominates none


def _skip(p: Path) -> bool:
    return any(part in SKIP_DIRS or part.startswith(".env") for part in p.parts) or p.name.startswith(".env")


def _merge(a, b):
    (la, ea), (lb, eb) = a, b
    if T in (la, lb):
        return T, ea | eb
    if G in (la, lb):
        return G, ea | eb
    return N, ea | eb


NONE = (N, frozenset())


class Scope:
    """Analyses one function body (or module body) with flow-insensitive assignment propagation."""

    def __init__(self, an, node, body, fname):
        self.an, self.node, self.body, self.fname = an, node, body, fname
        self.env: dict[str, tuple] = {}
        self.loader_vars: set[str] = set()
        a = getattr(node, "args", None)
        self.local_environ = set()
        if a:
            pos = a.posonlyargs + a.args
            pairs = list(zip(pos[len(pos) - len(a.defaults):], a.defaults)) + list(zip(a.kwonlyargs, a.kw_defaults))
            self.local_environ = {x.arg for x in pos + a.kwonlyargs} & ENVIRON_PARAMS
            self.local_environ |= {x.arg for x, dflt in pairs if dflt is not None and self.dotted(dflt) == "os.environ"}

    def is_loader(self, call):
        return isinstance(call, ast.Call) and self.dotted(call.func) in CONFIG_LOADERS

    def whole_environ(self, e):
        """True when `e` hands over the entire environment (os.environ, dict(os.environ), os.environ.copy(), {**os.environ})."""
        if self.is_environ(e):
            return True
        if isinstance(e, ast.Call):
            if isinstance(e.func, ast.Attribute) and e.func.attr == "copy" and self.is_environ(e.func.value):
                return True
            return any(self.whole_environ(a) for a in e.args)
        if isinstance(e, ast.Dict):
            return any(k is None and self.whole_environ(v) for k, v in zip(e.keys, e.values))
        return False

    def nodes(self):
        stack = [b for b in self.body if not isinstance(b, (ast.FunctionDef, ast.AsyncFunctionDef))]
        while stack:
            n = stack.pop()
            yield n
            for c in ast.iter_child_nodes(n):
                if isinstance(c, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                    continue
                stack.append(c)

    def dotted(self, e):
        parts = []
        while isinstance(e, ast.Attribute):
            parts.append(e.attr)
            e = e.value
        if isinstance(e, ast.Name):
            parts.append(self.an.aliases.get(e.id, e.id))
            return ".".join(reversed(parts))
        return None

    def is_environ(self, e):
        d = self.dotted(e)
        return d in ("os.environ",) or (isinstance(e, ast.Name) and (e.id in self.an.environ_names or e.id in self.local_environ))

    @staticmethod
    def env_name(call_or_sub):
        a = None
        if isinstance(call_or_sub, ast.Call) and call_or_sub.args:
            a = call_or_sub.args[0]
        elif isinstance(call_or_sub, ast.Subscript):
            a = call_or_sub.slice
        return a.value if isinstance(a, ast.Constant) and isinstance(a.value, str) else "?"

    def ev(self, e):
        """Return (level, env names) of an expression."""
        if e is None or isinstance(e, ast.Constant):
            return NONE
        if isinstance(e, ast.Name):
            return self.env.get(e.id, NONE)
        if isinstance(e, ast.Subscript):
            if self.is_environ(e.value):
                return T, frozenset({self.env_name(e)})
            if isinstance(e.value, ast.Name) and e.value.id in self.loader_vars:
                return T, frozenset({"config"})
            return self.ev(e.value)
        if isinstance(e, ast.Call):
            d = self.dotted(e.func)
            if d in ("os.getenv", "os.environ.get") or (isinstance(e.func, ast.Attribute) and e.func.attr == "get"
                                                          and self.is_environ(e.func.value)):
                return T, frozenset({self.env_name(e)})
            if d in GUARD_NAMES or (isinstance(e.func, ast.Name) and e.func.id == "safe_path"):
                return G, frozenset()
            fn = e.func.attr if isinstance(e.func, ast.Attribute) else (e.func.id if isinstance(e.func, ast.Name) else None)
            if self.is_loader(e):
                return T, frozenset({"config"})
            if d and d.startswith(SUBPROCESS_CALLS):
                return NONE  # a command's output is not the env value that was passed to it
            if isinstance(e.func, ast.Name) and fn in self.an.returns_tainted:  # plain-name calls only: methods collide by name
                return T, frozenset({"via:" + fn})
            if isinstance(e.func, ast.Attribute) and e.func.attr == "get" and isinstance(e.func.value, ast.Name) \
                    and e.func.value.id in self.loader_vars:
                return T, frozenset({"config"})
            acc = self.ev(e.func.value) if isinstance(e.func, ast.Attribute) else NONE
            for a in list(e.args) + [k.value for k in e.keywords]:
                acc = _merge(acc, self.ev(a.value if isinstance(a, ast.Starred) else a))
            return acc
        acc = NONE
        for c in ast.iter_child_nodes(e):
            if isinstance(c, (ast.expr_context, ast.operator, ast.boolop, ast.unaryop, ast.cmpop)):
                continue
            if isinstance(c, ast.comprehension):
                acc = _merge(acc, self.ev(c.iter))
                continue
            if isinstance(c, ast.expr):
                acc = _merge(acc, self.ev(c))
        return acc

    def targets(self, t):
        if isinstance(t, ast.Name):
            yield t.id
        elif isinstance(t, (ast.Tuple, ast.List)):
            for x in t.elts:
                yield from self.targets(x)
        elif isinstance(t, ast.Starred):
            yield from self.targets(t.value)

    def propagate(self):
        for _ in range(6):
            before = dict(self.env)
            for n in self.nodes():
                pairs = []
                if isinstance(n, ast.Assign):
                    pairs = [(t, n.value) for t in n.targets]
                elif isinstance(n, (ast.AnnAssign, ast.AugAssign)) and n.value is not None:
                    pairs = [(n.target, n.value)]
                elif isinstance(n, ast.NamedExpr):
                    pairs = [(n.target, n.value)]
                elif isinstance(n, (ast.For, ast.AsyncFor)):
                    pairs = [(n.target, n.iter)]
                elif isinstance(n, ast.withitem) and n.optional_vars is not None:
                    pairs = [(n.optional_vars, n.context_expr)]
                for tgt, val in pairs:
                    if self.is_loader(val):
                        for nm in self.targets(tgt):
                            self.loader_vars.add(nm)
                    lv = self.ev(val)
                    for nm in self.targets(tgt):
                        self.env[nm] = _merge(self.env.get(nm, NONE), lv)
            if self.env == before:
                break

    def returns_taint(self):
        return any(isinstance(n, ast.Return) and n.value is not None and self.ev(n.value)[0] == T for n in self.nodes())

    def sites(self, relfile, src):
        out, unknown = [], 0
        fsha = hashlib.sha256(
            (ast.get_source_segment(src, self.node) or "" if self.node is not None else src).encode()).hexdigest()
        fn = getattr(self.node, "name", "<module>")

        def add(n, verdict, kind, envs):
            out.append(dict(file=relfile, line=n.lineno, verdict=verdict, kind=kind, function=fn,
                            env=",".join(sorted(envs)) or "-", func_sha256=fsha))

        for n in self.nodes():
            if not isinstance(n, ast.Call):
                continue
            d = self.dotted(n.func)
            fnm = n.func.attr if isinstance(n.func, ast.Attribute) else getattr(n.func, "id", None)
            args = list(n.args) + [k.value for k in n.keywords]
            argv = [a.value if isinstance(a, ast.Starred) else a for a in args]
            lv = NONE
            for a in argv:
                lv = _merge(lv, self.ev(a))
            # whole os.environ passed to a call (class 6)
            if d and d.startswith(SUBPROCESS_CALLS):
                for a in argv:
                    if self.whole_environ(a):
                        add(n, "UNANALYSABLE", "whole-environ", {"os.environ"})
            if (d in GUARD_NAMES or fnm == "safe_path") and argv:
                first = self.ev(argv[0])
                if first[0] == T:
                    add(n, "GUARDED", "safe_path", first[1])
                continue
            if fnm in DYNAMIC or d in DYNAMIC:
                dl = NONE  # getattr/setattr/hasattr: only the attribute name (2nd argument) makes the access dynamic
                for a in (n.args[1:2] if fnm in ("getattr", "setattr", "hasattr") else argv):
                    dl = _merge(dl, self.ev(a.value if isinstance(a, ast.Starred) else a))
                if dl[0] == T:
                    add(n, "UNANALYSABLE", "dynamic:" + str(fnm), dl[1])
                continue
            if d in DSN_CALLS or fnm in ("create_engine",):
                consts = [c.value for a in argv for c in ast.walk(a) if isinstance(c, ast.Constant) and isinstance(c.value, str)]
                if any(DSN_FILE_PARAM.search(c) for c in consts):
                    add(n, "UNGUARDED", "dsn_file_param", lv[1])
                continue  # a tainted DSN without file params is a database URL, not a file path
            # Only the path operands count: the receiver of a Path method, or the leading positional / path-named keyword
            # arguments of a function sink. Content arguments (write_text(data)) and env= of subprocess are not paths.
            lvl = NONE
            is_sink = False
            if isinstance(n.func, ast.Attribute) and n.func.attr in SINK_ATTRS and d not in SINKS:
                is_sink = True
                lvl = self.ev(n.func.value)
            elif d in SINKS or (fnm == "open" and isinstance(n.func, ast.Name)):
                is_sink = True
                npos = 2 if d in TWO_PATH_SINKS else 1
                ops = list(n.args[:npos]) + [k.value for k in n.keywords if k.arg in PATH_KW]
                for a in ops:
                    lvl = _merge(lvl, self.ev(a.value if isinstance(a, ast.Starred) else a))
            if not is_sink:
                continue
            if lvl[0] == T:
                add(n, "UNGUARDED", fnm or d or "sink", lvl[1])
            elif lvl[0] == G:
                add(n, "GUARDED", fnm or d or "sink", lvl[1])
            elif any(isinstance(c, ast.Subscript) or (isinstance(c, ast.Call) and getattr(c.func, "attr", "") == "get")
                     for a in argv for c in ast.walk(a)):
                unknown += 1
        return out, unknown


class Analyzer:
    def __init__(self):
        self.returns_tainted: set[str] = set()
        self.aliases: dict[str, str] = {}
        self.environ_names: set[str] = set()

    def load_aliases(self, tree):
        self.aliases, self.environ_names = {}, set()
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                for a in n.names:
                    self.aliases[a.asname or a.name.split(".")[0]] = a.name if a.asname else a.name.split(".")[0]
            elif isinstance(n, ast.ImportFrom) and n.module:
                for a in n.names:
                    full = f"{n.module}.{a.name}"
                    self.aliases[a.asname or a.name] = full
                    if full == "os.environ":
                        self.environ_names.add(a.asname or a.name)
        for n in tree.body:  # module-level `name = module.attr` aliases (open = io.open, e = os.environ)
            if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
                parts, v = [], n.value
                while isinstance(v, ast.Attribute):
                    parts.append(v.attr)
                    v = v.value
                if isinstance(v, ast.Name) and parts:
                    full = ".".join([self.aliases.get(v.id, v.id)] + list(reversed(parts)))
                    self.aliases[n.targets[0].id] = full
                    if full == "os.environ":
                        self.environ_names.add(n.targets[0].id)
                elif isinstance(v, ast.Name) and not parts:
                    self.aliases[n.targets[0].id] = self.aliases.get(v.id, v.id)

    def scopes(self, tree):
        yield None, tree.body
        for n in ast.walk(tree):
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                yield n, n.body

    def run_file(self, relfile, text):
        try:
            tree = ast.parse(text)
        except SyntaxError:
            return [dict(file=relfile, line=1, verdict="UNANALYSABLE", kind="syntax-error", function="<module>", env="-",
                         func_sha256="")], 0
        self.load_aliases(tree)
        sites, unknown = [], 0
        for node, body in self.scopes(tree):
            sc = Scope(self, node, body, getattr(node, "name", "<module>"))
            sc.propagate()
            if node is not None and sc.returns_taint():
                self.returns_tainted.add(node.name)
            s, u = sc.sites(relfile, text)
            sites += s
            unknown += u
        return sites, unknown


SH_CMD = re.compile(r"(?:^|[;&|(]\s*|\s)(cat|cp|mv|rm|tee|source|\.|mkdir|touch|ln|chmod)\s+[^#]*?\$\{?[A-Za-z_]")
SH_REDIR = re.compile(r">>?\s*\"?\$\{?[A-Za-z_]")
SH_VAR = re.compile(r"\$\{?([A-Za-z_][A-Za-z0-9_]*)")


def scan_shell(relfile, text):
    out = []
    for i, line in enumerate(text.splitlines(), 1):
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        fsha = hashlib.sha256(text.encode()).hexdigest()
        base = dict(file=relfile, line=i, function="<script>", func_sha256=fsha)
        if re.search(r"(^|[;&|\s])eval\s", s) or "${!" in s:
            out.append(dict(base, verdict="UNANALYSABLE", kind="shell-eval-indirect", env="-"))
        elif SH_CMD.search(s) or SH_REDIR.search(s):
            names = sorted({m for m in SH_VAR.findall(s)})
            out.append(dict(base, verdict="UNGUARDED", kind="shell-path-operand", env=",".join(names) or "-"))
    return out


def sha256_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_exemptions(path, today):
    """Return (entries, problems)."""
    if path is None or not Path(path).is_file():
        return [], []
    probs, ents = [], []
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        raw = data["exemptions"]
    except (OSError, ValueError, KeyError, TypeError):
        return [], ["exemptions file unreadable or malformed"]
    need = ("file", "function", "env", "sink", "func_sha256", "reason", "reviewer", "review_by")
    for i, e in enumerate(raw):
        miss = [k for k in need if not isinstance(e, dict) or not str(e.get(k, "")).strip()]
        if miss:
            probs.append(f"exemption #{i}: missing {','.join(miss)}")
            continue
        if e["reason"] not in REASONS:
            probs.append(f"exemption #{i}: reason not in enumerated set")
        if str(e["reviewer"]).strip().lower() in ("implementer", "self", "me"):
            probs.append(f"exemption #{i}: reviewer must be independent")
        try:
            if dt.date.fromisoformat(e["review_by"]) < today:
                probs.append(f"exemption #{i}: expired {e['review_by']}")
        except ValueError:
            probs.append(f"exemption #{i}: bad review_by date")
        ents.append(e)
    return ents, probs


def scan(base, work_dirs=(), exemptions_path=None, today=None):
    base, today = Path(base), today or dt.date.today()
    files = []
    for r in PY_ROOTS:
        p = base / r
        files += [p] if p.is_file() else sorted(p.rglob("*.py")) if p.is_dir() else []
    shells = [base / s for s in SH_FILES if (base / s).is_file()]
    for w in work_dirs:
        files += sorted(Path(w).rglob("*.py"))
        shells += sorted(Path(w).rglob("*.sh"))
    files = [f for f in files if f.is_file() and not f.is_symlink() and not _skip(f.relative_to(base) if f.is_relative_to(base) else Path(f.name))]
    shells = [f for f in shells if f.is_file() and not f.is_symlink() and not _skip(f.relative_to(base) if f.is_relative_to(base) else Path(f.name))]

    def rel(f):
        return str(f.relative_to(base)) if f.is_relative_to(base) else str(f)

    texts = {f: f.read_text(encoding="utf-8", errors="replace") for f in files}
    an = Analyzer()
    for _ in range(4):  # fixpoint for cross-function return taint
        before = set(an.returns_tainted)
        for f in files:
            an.run_file(rel(f), texts[f])
        if an.returns_tainted == before:
            break
    sites, unknown = [], 0
    for f in files:
        s, u = an.run_file(rel(f), texts[f])
        sites += s
        unknown += u
    for f in shells:
        sites += scan_shell(rel(f), f.read_text(encoding="utf-8", errors="replace"))
    ents, problems = load_exemptions(exemptions_path, today)
    used = set()
    for s in sites:
        s["exempt"] = False
        if s["verdict"] == "GUARDED":
            continue
        for i, e in enumerate(ents):
            if (e["file"], e["function"], e["env"], e["sink"], e["func_sha256"]) == (
                    s["file"], s["function"], s["env"], s["kind"], s["func_sha256"]):
                s["exempt"] = True
                used.add(i)
    for i, e in enumerate(ents):
        if i not in used:
            problems.append(f"stale exemption #{i} ({e['file']}:{e['function']}): matches no site or function changed")
    sites.sort(key=lambda s: (s["file"], s["line"], s["verdict"], s["kind"]))
    unreviewed = sum(1 for e in ents if str(e["reviewer"]).upper().startswith("PENDING"))
    return dict(sites=sites, unknown=unknown, problems=problems, unreviewed=unreviewed)


def failures(res):
    bad = [s for s in res["sites"] if s["verdict"] in ("UNGUARDED", "UNANALYSABLE") and not s["exempt"]]
    return bad, res["problems"]


def format_result(res):
    lines = [f"{s['file']}:{s['line']} {s['verdict']}{' (exempt)' if s['exempt'] else ''} {s['kind']} {s['env']}"
             for s in res["sites"]]
    c = lambda v, ex=None: sum(1 for s in res["sites"] if s["verdict"] == v and (ex is None or s["exempt"] == ex))  # noqa: E731
    lines.append(f"SUMMARY guarded={c('GUARDED')} unguarded={c('UNGUARDED')} (exempt={c('UNGUARDED', True)}) "
                 f"unanalysable={c('UNANALYSABLE')} (exempt={c('UNANALYSABLE', True)}) UNKNOWN={res['unknown']} "
                 f"exemption_problems={len(res['problems'])} UNREVIEWED_EXEMPTIONS={res.get('unreviewed', 0)}")
    lines += [f"PROBLEM {p}" for p in res["problems"]]
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base", default=str(Path(__file__).resolve().parent.parent))
    ap.add_argument("--work", action="append", default=[], help="extra directory of .py/.sh files to scan")
    ap.add_argument("--exemptions", default=None)
    ap.add_argument("--emit", action="store_true", help="print exemption skeletons (reason/reviewer left empty) for open sites")
    a = ap.parse_args(argv)
    ex = a.exemptions or str(Path(a.base) / "config" / "env_paths_exemptions.json")
    res = scan(a.base, a.work, ex)
    if a.emit:
        seen = {}
        for s in res["sites"]:
            if s["verdict"] in ("UNGUARDED", "UNANALYSABLE") and not s["exempt"]:
                seen[(s["file"], s["function"], s["env"], s["kind"])] = dict(
                    file=s["file"], function=s["function"], env=s["env"], sink=s["kind"], func_sha256=s["func_sha256"],
                    reason="", reviewer="", review_by="")
        print(json.dumps({"exemptions": list(seen.values())}, indent=1))
        return 0
    print(format_result(res))
    bad, probs = failures(res)
    return 1 if bad or probs else 0


if __name__ == "__main__":
    sys.exit(main())

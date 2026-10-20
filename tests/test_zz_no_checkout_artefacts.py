"""Regression (PLAN-DEF1, hardened per VET-DEF1 R3): a guarded run must leave no bytecode, pytest cache or tmp/
artefact in the checkout.
Named test_zz_* so it is collected last (files run alphabetically); it sees everything earlier tests and their children
wrote. A BASELINE of artefact paths (each with the newest mtime in its subtree) is taken when this module is imported;
the run fails on any NEW artefact path or on any baseline path whose subtree became newer. tmp/ is scanned recursively,
so a new mkdtemp under an existing tmp/release-gate is caught. A stale tree therefore cannot hide new writes.
Limits (documented, not fixed here): writes made at session finish are not seen, and another process touching the
checkout during the run (an editor, an operator using ./tmp, a parallel run) can cause a false positive.
Never deletes anything."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_SKIP_TOP = {".git", ".venv", "venv", "node_modules"}
_NAMES = {"__pycache__", ".pytest_cache"}


def _mtime(p: str) -> float | None:
    try:
        return os.lstat(p).st_mtime
    except OSError:
        return None


def _subtree_newest(top: str) -> float | None:
    newest = _mtime(top)
    if newest is None:
        return None
    for d, dirs, files in os.walk(top):
        for x in dirs + files:
            m = _mtime(os.path.join(d, x))
            if m is not None and m > newest:
                newest = m
    return newest


def artefact_map(root: Path) -> dict[str, float]:
    """{relative artefact path: newest mtime in its subtree}. Every path under top-level tmp/ is its own entry."""
    root = str(root)
    found: dict[str, float] = {}
    for d, dirs, _files in os.walk(root):
        rel = os.path.relpath(d, root)
        if rel == ".":
            dirs[:] = [x for x in dirs if x not in _SKIP_TOP]
            if "tmp" in dirs:
                dirs.remove("tmp")
                tmp = os.path.join(root, "tmp")
                m = _mtime(tmp)
                if m is not None:
                    found["tmp"] = m
                for td, tdirs, tfiles in os.walk(tmp):
                    for x in tdirs + tfiles:
                        p = os.path.join(td, x)
                        m = _mtime(p)
                        if m is not None:
                            found[os.path.relpath(p, root)] = m
        for x in list(dirs):
            if x in _NAMES:
                p = os.path.join(d, x)
                m = _subtree_newest(p)
                if m is not None:
                    found[os.path.normpath(os.path.join(rel, x))] = m
                dirs.remove(x)
    return found


def new_artefacts(baseline: dict[str, float], current: dict[str, float]) -> list[str]:
    return sorted(p for p, m in current.items() if p not in baseline or m > baseline[p])


_BASELINE = artefact_map(ROOT)


def test_scan_detects_planted_artefacts(tmp_path):
    (tmp_path / "pkg" / "__pycache__").mkdir(parents=True)
    (tmp_path / "tmp" / "release-gate").mkdir(parents=True)
    (tmp_path / ".git" / "__pycache__").mkdir(parents=True)
    for d in ("pkg/__pycache__", "tmp", "tmp/release-gate"):               # deterministic: baseline is old
        os.utime(tmp_path / d, (1_000_000, 1_000_000))
    base = artefact_map(tmp_path)
    assert set(base) == {"pkg/__pycache__", "tmp", "tmp/release-gate"}
    assert new_artefacts(base, artefact_map(tmp_path)) == []
    (tmp_path / ".pytest_cache").mkdir()                                   # a new path
    (tmp_path / "tmp" / "release-gate" / "run-x").mkdir()                  # new nested tmp/ entry
    (tmp_path / ".git" / "__pycache__" / "y.pyc").write_bytes(b"")         # ignored: under .git
    pyc = tmp_path / "pkg" / "__pycache__" / "m.pyc"                       # newer subtree of a baseline path
    pyc.write_bytes(b"")
    os.utime(pyc, (2_000_000, 2_000_000))
    os.utime(tmp_path / "pkg" / "__pycache__", (1_000_000, 1_000_000))     # only the subtree is newer
    assert new_artefacts(base, artefact_map(tmp_path)) == [
        ".pytest_cache", "pkg/__pycache__", "tmp/release-gate", "tmp/release-gate/run-x"]


def test_run_left_no_artefacts_in_checkout():
    assert new_artefacts(_BASELINE, artefact_map(ROOT)) == [], \
        "guarded run wrote into the checkout (see PLAN-DEF1 root causes)"

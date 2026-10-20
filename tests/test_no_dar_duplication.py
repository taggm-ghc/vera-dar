"""Rename lint (owner instruction 2026-10-19): the product name is VERA-DAR; it must never be doubled.

Fails when a text file in the checkout holds the product name twice in a row, joined by a hyphen, space or
underscore, or with no separator. The scan skips frozen historical evidence and tool output directories.
The needle is built from pieces so that this file does not trip its own scan.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_DAR = "DA" + "R"
_VERA = "VE" + "RA"
_DOUBLE = re.compile(_DAR + r"[-_ ]?" + _DAR, re.IGNORECASE)
_SKIP_DIRS = {".git", "venv", ".venv", "__pycache__", "tmp", "eval_results", ".pytest_cache", "node_modules"}
_SKIP_PREFIXES = ("docs/trace_eval/",)


def _text_files():
    for p in ROOT.rglob("*"):
        if not p.is_file() or _SKIP_DIRS.intersection(p.relative_to(ROOT).parts):
            continue
        rel = p.relative_to(ROOT).as_posix()
        if rel.startswith(_SKIP_PREFIXES) or p.suffix in {".pyc", ".png", ".jpg", ".gz", ".zip", ".fossil"}:
            continue
        try:
            yield rel, p.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue


def test_no_doubled_product_name():
    hits = [f"{rel}:{i}" for rel, txt in _text_files() for i, ln in enumerate(txt.split("\n"), 1) if _DOUBLE.search(ln)]
    assert not hits, "doubled product name: " + ", ".join(hits[:20])


def test_scan_pattern_catches_doubling():
    assert _DOUBLE.search(_VERA + "-" + _DAR + "-" + _DAR)
    assert _DOUBLE.search(_DAR + " " + _DAR)
    assert not _DOUBLE.search(_VERA + "-" + _DAR + " and " + _DAR + "INT")

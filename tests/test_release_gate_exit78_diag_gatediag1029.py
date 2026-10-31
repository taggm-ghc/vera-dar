"""GATEDIAG-1029: the release gate prints a bounded, redacted child-stderr excerpt on exit 78 only.

Synthetic data only; no process, DB or network. The suite child is faked by monkeypatching release_gate._run.
"""
from __future__ import annotations

import importlib.util
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("release_gate_gatediag", ROOT / "scripts" / "release_gate.py")
g = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(g)

PREFIX = "exit 78 detail (test-boundary or sandbox refusal); child stderr (sanitized):"
ENV = {"VERA_API_KEY": "zzSyntheticSecretValue", "HOME": "/home/x", "SOME_DSN": "pg-synth-dsn-9"}


def _s(text, env=ENV):
    return "\n".join(g.sanitize_child_stderr(text, env))


@pytest.mark.parametrize("raw,leak", [
    ("connect https://db.example.com:5432/x?sslmode=require failed", "example.com"),
    ("postgres://alice:hunter2@10.1.2.3/db", "hunter2"),
    ("login alice:hunter2@somehost", "hunter2"),
    ("token=abc123def password: s3cret API_KEY=qwerty", "s3cret"),
    ("peer 192.168.10.20 refused", "192.168.10.20"),
    ("peer fe80::1ff:fe23:4567:890a refused", "fe80"),
    ("blob " + "a1b2c3d4" * 4 + " end", "a1b2c3d4a1b2"),
    ("blob QUJDREVGR0hJSktMTU5PUFFSU1RVVldYWVo= end", "QUJDREVGR0hJ"),
    ("cannot reach api.vendor.io now", "vendor.io"),
    ("env leaked zzSyntheticSecretValue here and pg-synth-dsn-9", "zzSyntheticSecretValue"),
])
def test_secrets_urls_ips_hosts_redacted(raw, leak):
    out = _s(raw)
    assert leak not in out
    assert "[REDACTED]" in out


def test_env_dsn_value_redacted_and_benign_text_kept():
    out = _s("unshare: unshare failed: Operation not permitted\nuses pg-synth-dsn-9")
    assert "pg-synth-dsn-9" not in out
    assert "unshare failed: Operation not permitted" in out


def test_length_bounded_lines_and_bytes():
    many = "\n".join(f"line {i} unshare: no" for i in range(500))
    out = g.sanitize_child_stderr(many, {})
    assert len(out) <= g.STDERR_MAX_LINES and out[-1].startswith("line 499")
    wide = "\n".join("x " * 200 for _ in range(20))
    assert len("\n".join(g.sanitize_child_stderr(wide, {}))) <= g.STDERR_MAX_BYTES + 3


def test_control_chars_neutralised_and_never_raises():
    assert "\x1b" not in _s("\x1b[31mRESULT: PASS\x1b[0m")
    assert g.sanitize_child_stderr(None, None) is not None


def _check(monkeypatch, tmp_path, rc, stderr):
    monkeypatch.setattr(g, "inherited_pytest_configs", lambda root: [])
    monkeypatch.setattr(g, "_run", lambda cmd, **kw: types.SimpleNamespace(
        returncode=rc, stdout="", stderr=stderr))
    monkeypatch.delenv("PYTEST_ADDOPTS", raising=False)
    return g.check_tests(tmp_path)


def test_exit_78_prints_prefix_and_sanitized_excerpt(monkeypatch, tmp_path):
    errs = _check(monkeypatch, tmp_path, 78, "unshare: failed\nconnect https://secret.example.com/x\n")
    assert errs[0].startswith("test isolation was not established")
    assert errs[1] == PREFIX
    assert any("unshare: failed" in e for e in errs[2:])
    assert not any("example.com" in e for e in errs)


def test_non_78_exits_print_no_sanitized_excerpt(monkeypatch, tmp_path):
    for rc in (64, 1):
        errs = _check(monkeypatch, tmp_path, rc, "boom\n")
        assert errs and not any(PREFIX in e for e in errs)
    assert _check(monkeypatch, tmp_path, 124, "") != []


def test_f1_huge_lines_bounded_and_fast():
    import time
    t = time.monotonic()
    one = g.sanitize_child_stderr("a:" * 500_000, ENV)
    many = g.sanitize_child_stderr("\n".join(("x:" * 5000) for _ in range(2000)), ENV)
    assert time.monotonic() - t < 1.0
    assert len("\n".join(one)) <= g.STDERR_MAX_BYTES + 3
    assert len("\n".join(many)) <= g.STDERR_MAX_BYTES + 3


def test_f3_control_char_cannot_split_keyword():
    out = _s("pass\x1bword=hunter2")
    assert "hunter2" not in out


def test_f3_format_chars_cannot_split_keyword():
    for sep in ("​", "­", "⁠", "﻿"):
        out = _s(f"pass{sep}word=hunter2")
        assert "hunter2" not in out, sep


def _frags(secret, n=4):
    return [secret[i:i + n] for i in range(len(secret) - n + 1)]


def test_n1_line_cut_straddling_secret_leaves_no_fragment():
    secret = "Zq9!xR#2mT$7vB&4"
    env = {"VERA_API_KEY": secret}
    for pad in range(g.STDERR_LINE_MAX - len(secret) - 2, g.STDERR_LINE_MAX + 2):
        out = _s(("w " * 400)[:pad] + "v=" + secret + " tail", env)
        assert not any(f in out for f in _frags(secret)), pad
    # a cut that is not caused by a secret is marked and the partial last token is not shown
    out = _s("ab " * 300 + "partialtoken" * 20, env)
    assert out.endswith(g.STDERR_CUT_MARK) and "partialtoken" not in out


def test_n1_raw_cut_straddling_secret_leaves_no_fragment():
    secret = "Zq9!xR#2mT$7vB&4"
    env = {"VERA_API_KEY": secret}
    base = ("k " * 100 + "\n") * 60                       # long enough to force the raw cut
    for off in range(0, 40):
        text = ("z" * off + secret + "\n") + base
        out = _s(text, env)
        assert not any(f in out for f in _frags(secret)), off
    # secret sits exactly across the raw-cut boundary
    for k in range(1, len(secret)):
        text = "x" * 50 + secret[:k] + secret[k:] + "\n" + ("q " * 100 + "\n") * 0
        tail = "\n" + "y" * (g.STDERR_RAW_MAX - len(secret) + k - 1)
        out = _s(text + tail + "\nlast line", env)
        assert not any(f in out for f in _frags(secret)), k


@pytest.mark.parametrize("sep", ["\x1b", "​", "\t", "\x00"])
def test_n4_secret_split_by_removed_char_still_redacted(sep):
    secret = "Zq9!xR#2mT$7vB&4"
    out = _s("err " + secret[:8] + sep + secret[8:] + " end", {"VERA_API_KEY": secret})
    assert not any(f in out for f in _frags(secret)), repr(sep)


def test_n5_single_long_line_gets_marker_not_silence():
    out = g.sanitize_child_stderr("x" * (g.STDERR_RAW_MAX + 100), ENV)
    assert out == ["(stderr tail truncated: no complete line in last 8 KB)"]


def test_n5_cut_on_line_start_keeps_first_line():
    first = "L" * 7 + " keepme"
    text = "junk line\n" + first + "\n" + "z" * (g.STDERR_RAW_MAX - len(first) - 2) + "\n"
    assert len(text) == g.STDERR_RAW_MAX + 10 and text[-g.STDERR_RAW_MAX - 1] == "\n"
    out = g.sanitize_child_stderr(text, {})
    assert out[0] == first


def test_n2_long_userinfo_password_redacted():
    out = _s("alice:" + "!a#" * 100 + "@somehost")
    assert "!a#!a#" not in out


def test_n2_worst_case_timing_bounded():
    import time
    t = time.monotonic()
    for unit in ("-key", "key-", ".key", "a:", "x@"):
        g.sanitize_child_stderr("\n".join((unit * 600)[:512] for _ in range(20)), {})
    assert time.monotonic() - t < 5.0

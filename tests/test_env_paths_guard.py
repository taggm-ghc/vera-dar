"""ENV-PATHS-1110: scripts/check_env_paths.py (deterministic lint) on the repo and on planted fixtures. Offline."""
import datetime as dt
import functools
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("check_env_paths", REPO / "scripts" / "check_env_paths.py")
cep = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cep)

# Pins (ENV-PATHS-1110): changing the checker or the exemption list must be a visible, reviewed change.
PIN_EXEMPTIONS_SHA256 = "b97629705d785a12cb62495b4f294c30797d02ef9c25b6ace2fdfbbc841a5a8d"
PIN_CHECKER_SHA256 = "fd78b639d23ce895b0bb1ff5407cdc321b7fc57192835971006ac14e991db420"
PIN_UNREVIEWED = 0  # exemptions still carrying a PENDING reviewer (none: the fulldoc site is wired)
TODAY = dt.date(2026, 11, 10)


def _scan(tmp_path, files, exemptions=None):
    for rel, text in files.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    ex = None
    if exemptions is not None:
        ex = tmp_path / "ex.json"
        ex.write_text(json.dumps({"exemptions": exemptions}))
    return cep.scan(tmp_path, exemptions_path=ex, today=TODAY)


def _verdicts(res):
    return sorted((s["verdict"], s["kind"]) for s in res["sites"])


def _has(res, verdict):
    return any(s["verdict"] == verdict for s in res["sites"])


# ---- the repository itself ---------------------------------------------------------------------------------------
@functools.lru_cache(maxsize=1)
def _repo_scan():
    return cep.scan(REPO, exemptions_path=REPO / "config" / "env_paths_exemptions.json", today=TODAY)


def test_repo_has_no_unexempt_unguarded_or_unanalysable_site():
    res = _repo_scan()
    bad, probs = cep.failures(res)
    assert not bad, "\n".join(f"{s['file']}:{s['line']} {s['verdict']} {s['kind']} {s['env']}" for s in bad)
    assert not probs, probs
    guarded = {(s["file"], s["env"]) for s in res["sites"] if s["verdict"] == "GUARDED"}
    assert ("vera/m2_runner.py", "VERA_COST_LOG") in guarded and ("vera/trace_viz.py", "VERA_TRACE_EVAL_RESULTS") in guarded
    assert any(f == "vera/trace_eval/ui.py" for f, _ in guarded)
    assert any(f == "vera/fulldoc_corpus.py" for f, _ in guarded)
    assert res["unreviewed"] == PIN_UNREVIEWED


def test_repo_summary_prints_unanalysable_and_unknown_never_clean():
    out = cep.format_result(_repo_scan())
    assert "unanalysable=" in out and "UNKNOWN=" in out and "clean" not in out.lower()


def test_output_is_deterministic():
    a = cep.format_result(_repo_scan())
    b = cep.format_result(_repo_scan())
    assert a == b


def test_pins():
    assert hashlib.sha256((REPO / "config" / "env_paths_exemptions.json").read_bytes()).hexdigest() == PIN_EXEMPTIONS_SHA256
    assert hashlib.sha256((REPO / "scripts" / "check_env_paths.py").read_bytes()).hexdigest() == PIN_CHECKER_SHA256


# ---- false-negative classes (plan section D) ---------------------------------------------------------------------
def test_c0_direct_flow_is_unguarded(tmp_path):
    r = _scan(tmp_path, {"vera/a.py": "import os\ndef f():\n    p = os.environ.get('X')\n    return open(p).read()\n"})
    assert ("UNGUARDED", "open") in _verdicts(r)


def test_c1_aliasing(tmp_path):
    r = _scan(tmp_path, {"vera/a.py": "import os\ndef f():\n    p = os.getenv('X')\n    q = p\n    z = q\n    open(z)\n"})
    assert _has(r, "UNGUARDED")


def test_c2_dict_config_lookup(tmp_path):
    src = ("from vera import env_precedence\nfrom pathlib import Path\ndef f():\n    cfg = env_precedence.load_config()\n"
           "    return Path(cfg['x']).read_text()\n")
    assert _has(_scan(tmp_path, {"vera/a.py": src}), "UNGUARDED")
    other = "from pathlib import Path\ndef f(cfg):\n    return Path(cfg['x']).read_text()\n"
    r = _scan(tmp_path / "o", {"vera/a.py": other})
    assert not _has(r, "UNGUARDED") and r["unknown"] == 0  # not a pinned loader; Path(...) receiver not a counted sink arg


def test_c2_unknown_lookup_into_open_is_counted(tmp_path):
    r = _scan(tmp_path, {"vera/a.py": "def f(cfg):\n    return open(cfg['x'])\n"})
    assert r["unknown"] == 1 and not _has(r, "UNGUARDED")


def test_c3_fstring_concat_format_join(tmp_path):
    body = ("import os\ndef f():\n    p = os.environ.get('X')\n    open(f'{p}/a')\n    open(p + '/a')\n    open('%s/a' % p)\n"
            "    open('{}/a'.format(p))\n    open(os.path.join(p, 'a'))\n")
    r = _scan(tmp_path, {"vera/a.py": body})
    assert sum(1 for s in r["sites"] if s["verdict"] == "UNGUARDED") == 5


def test_c4_getattr_setattr_vars_eval(tmp_path):
    body = ("import os\ndef f(o):\n    n = os.environ.get('X')\n    getattr(o, n)\n    setattr(o, n, 1)\n    eval(n)\n"
            "    exec(n)\n    vars(n)\n")
    r = _scan(tmp_path, {"vera/a.py": body})
    assert sum(1 for s in r["sites"] if s["verdict"] == "UNANALYSABLE") == 5
    assert not _has(_scan(tmp_path / "o", {"vera/a.py": "def f(o):\n    return getattr(o, 'x')\n"}), "UNANALYSABLE")


def test_c5_cross_function_return(tmp_path):
    body = ("import os\ndef helper():\n    return os.environ.get('X')\ndef f():\n    open(helper())\n")
    assert _has(_scan(tmp_path, {"vera/a.py": body}), "UNGUARDED")
    cross = {"vera/h.py": "import os\ndef helper():\n    return os.environ.get('X')\n",
             "vera/b.py": "from vera.h import helper\ndef f():\n    open(helper())\n"}
    assert _has(_scan(tmp_path / "x", cross), "UNGUARDED")


def test_c6_whole_environ_to_subprocess(tmp_path):
    body = ("import os, subprocess\ndef f():\n    subprocess.run(['x'], env=os.environ)\n    subprocess.run(['x'], env=dict(os.environ))\n"
            "    subprocess.run(['x'], env=os.environ.copy())\n    subprocess.run(['x'], env={**os.environ})\n")
    r = _scan(tmp_path, {"vera/a.py": body})
    assert sum(1 for s in r["sites"] if s["kind"] == "whole-environ") == 4
    mapping = "import os\ndef f(environ):\n    open(environ.get('X'))\n"
    assert _has(_scan(tmp_path / "m", {"vera/a.py": mapping}), "UNGUARDED")  # Mapping parameter named environ
    dflt = "import os\ndef f(env=os.environ):\n    open(env['X'])\n"
    assert _has(_scan(tmp_path / "d", {"vera/a.py": dflt}), "UNGUARDED")


def test_c7_sink_aliases(tmp_path):
    body = ("import os, io, shutil, glob\nfrom pathlib import Path as P\nfrom os import environ as E\nopen2 = io.open\n"
            "def f():\n    p = E['X']\n    P(p).read_text()\n    open2(p)\n    shutil.copy(p, '/x')\n    glob.glob(p)\n"
            "    os.scandir(p)\n    P(p).write_text('a')\n")
    r = _scan(tmp_path, {"vera/a.py": body})
    kinds = sorted(s["kind"] for s in r["sites"] if s["verdict"] == "UNGUARDED")
    assert kinds == ["copy", "glob", "open2", "read_text", "scandir", "write_text"] or len(kinds) == 6, kinds


def test_c7_dsn_file_params(tmp_path):
    flagged = "import os\nfrom sqlalchemy import create_engine\ndef f():\n    create_engine(os.environ['U'] + '?sslrootcert=/x/ca.pem')\n"
    assert ("UNGUARDED", "dsn_file_param") in _verdicts(_scan(tmp_path, {"vera/a.py": flagged}))
    plain = "import os\nfrom sqlalchemy import create_engine\ndef f():\n    create_engine(os.environ['U'])\n"
    assert not _has(_scan(tmp_path / "p", {"vera/a.py": plain}), "UNGUARDED")  # a DB URL is not a file path


def test_c8_shell(tmp_path):
    sh = ("#!/bin/sh\nset -e\ncat \"$CFG\"\nrm -f ${OUT}\necho hi > \"$LOG\"\neval \"$X\"\nv=NAME\necho ${!v}\n# cat $IGNORED\n"
          "echo \"$UNRELATED\"\n")
    r = _scan(tmp_path, {"start.sh": sh})
    got = [(s["line"], s["verdict"]) for s in r["sites"]]
    assert got == [(3, "UNGUARDED"), (4, "UNGUARDED"), (5, "UNGUARDED"), (6, "UNANALYSABLE"), (8, "UNANALYSABLE")], got


def test_selector_not_a_path_is_not_a_site(tmp_path):
    body = "import os\ndef f():\n    if os.environ.get('SEL') == '1':\n        return open('/fixed/file')\n"
    assert not _has(_scan(tmp_path, {"vera/a.py": body}), "UNGUARDED")


# ---- GUARDED requires data flow -----------------------------------------------------------------------------------
def test_guarded_requires_the_safe_path_result_to_reach_the_sink(tmp_path):
    ok = ("import os\nfrom vera.safe_path import safe_path\ndef f(R):\n    raw = os.environ.get('X')\n"
          "    p = safe_path(raw, [R], key='X')\n    return open(p)\n")
    r = _scan(tmp_path, {"vera/a.py": ok})
    assert _has(r, "GUARDED") and not _has(r, "UNGUARDED")
    through = ("import os\nfrom vera.safe_path import safe_path\ndef f(R):\n    raw = os.environ.get('X')\n"
               "    safe_path(raw, [R], key='X')\n    return open(raw)\n")
    r2 = _scan(tmp_path / "t", {"vera/a.py": through})
    assert _has(r2, "UNGUARDED")  # merely passing the raw value through safe_path() does not guard the later open
    mixed = ("import os\nfrom vera.safe_path import safe_path\ndef f(R):\n    raw = os.environ.get('X')\n"
             "    p = safe_path(raw, [R], key='X')\n    return open(p + raw)\n")
    assert _has(_scan(tmp_path / "m", {"vera/a.py": mixed}), "UNGUARDED")


# ---- skipped by name ----------------------------------------------------------------------------------------------
def test_env_files_and_tests_are_skipped_by_name(tmp_path):
    bad = "import os\ndef f():\n    open(os.environ['X'])\n"
    r = _scan(tmp_path, {"vera/.env-planted.py": bad, "vera/env/x.py": bad, "tests/t.py": bad, "scripts/.env-x.sh": "cat $A\n"})
    assert r["sites"] == []


# ---- exemption list ------------------------------------------------------------------------------------------------
SRC = "import os\ndef f():\n    open(os.environ['X'])\n"


def _entry(r, **over):
    s = r["sites"][0]
    e = dict(file=s["file"], function=s["function"], env=s["env"], sink=s["kind"], func_sha256=s["func_sha256"],
             reason="selector", reviewer="verifier-1", review_by="2026-12-01")
    e.update(over)
    return e


def test_exemption_applies_and_exit_is_clean(tmp_path):
    first = _scan(tmp_path, {"vera/a.py": SRC})
    r = _scan(tmp_path, {}, [_entry(first)])
    bad, probs = cep.failures(r)
    assert not bad and not probs and r["sites"][0]["exempt"]


@pytest.mark.parametrize("over,needle", [({"reason": "because"}, "reason"), ({"reviewer": "implementer"}, "reviewer"),
                                         ({"review_by": "2026-11-01"}, "expired"), ({"review_by": "soon"}, "date"),
                                         ({"reviewer": ""}, "missing")])
def test_exemption_format_rejections(tmp_path, over, needle):
    first = _scan(tmp_path, {"vera/a.py": SRC})
    _, probs = cep.failures(_scan(tmp_path, {}, [_entry(first, **over)]))
    assert any(needle in p for p in probs), probs


def test_stale_exemption_fails(tmp_path):
    first = _scan(tmp_path, {"vera/a.py": SRC})
    e = _entry(first, function="gone")
    _, probs = cep.failures(_scan(tmp_path, {}, [e]))
    assert any("stale" in p for p in probs)


def test_changed_function_invalidates_its_exemption(tmp_path):
    first = _scan(tmp_path, {"vera/a.py": SRC})
    e = _entry(first)
    r = _scan(tmp_path, {"vera/a.py": SRC + "    pass\n"}, [e])
    bad, probs = cep.failures(r)
    assert bad and any("stale" in p for p in probs)


def test_pending_reviewer_is_counted_not_hidden(tmp_path):
    first = _scan(tmp_path, {"vera/a.py": SRC})
    r = _scan(tmp_path, {}, [_entry(first, reviewer="PENDING-VERIFIER", reason="pending-decision")])
    assert r["unreviewed"] == 1 and "UNREVIEWED_EXEMPTIONS=1" in cep.format_result(r)


def test_malformed_exemption_file_fails_closed(tmp_path):
    (tmp_path / "vera").mkdir()
    (tmp_path / "ex.json").write_text("{not json")
    r = cep.scan(tmp_path, exemptions_path=tmp_path / "ex.json", today=TODAY)
    assert r["problems"]


def test_cli_exit_codes(tmp_path):
    (tmp_path / "vera").mkdir()
    (tmp_path / "vera" / "a.py").write_text(SRC)
    assert cep.main(["--base", str(tmp_path), "--exemptions", str(tmp_path / "none.json")]) == 1
    (tmp_path / "vera" / "a.py").write_text("x = 1\n")
    assert cep.main(["--base", str(tmp_path), "--exemptions", str(tmp_path / "none.json")]) == 0

"""Offline tests for scripts/run_with_env.py. Uses only temp files; never reads the real .env."""
import importlib.util
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "run_with_env.py"
_spec = importlib.util.spec_from_file_location("run_with_env", SCRIPT)
rwe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rwe)

FAKE = (
    "# comment\n"
    "GROQ_API_KEY=fake-groq\n"
    "export OPENAI_API_KEY='fake-openai'\n"
    'TAVILY_API_KEY="fake-tavily"\n'
    "VERA_DB_URL_RW=postgresql://u:fakepw@host/db\n"
    "ANTHROPIC_API_KEY=must-not-be-copied\n"
    "OTHER=ignored\n"
)


def test_parse_copies_only_allow_listed_names_and_strips_quotes():
    got = rwe.parse_env_text(FAKE)
    assert set(got) == set(rwe.ALLOWED_NAMES)
    assert got["OPENAI_API_KEY"] == "fake-openai"
    assert "TAVILY_API_KEY" not in got
    assert "ANTHROPIC_API_KEY" not in got and "OTHER" not in got


def test_empty_value_counts_as_missing():
    assert "GROQ_API_KEY" not in rwe.parse_env_text("GROQ_API_KEY=\n")


def test_file_is_parsed_not_executed(tmp_path):
    marker = tmp_path / "ran"
    f = tmp_path / "e.env"
    f.write_text(f"GROQ_API_KEY=$(touch {marker})\n")
    rwe.parse_env_text(f.read_text())
    assert not marker.exists()


def _run(tmp_path, text, *extra, code="import os;print(os.environ.get('GROQ_API_KEY'))"):
    f = tmp_path / "e.env"
    f.write_text(text)
    return subprocess.run([sys.executable, "-B", str(SCRIPT), "--env-file", str(f), *extra, "--",
                           sys.executable, "-B", "-c", code], capture_output=True, text=True,
                          env={"PATH": "/usr/bin:/bin"})


def test_child_sees_value_but_wrapper_never_prints_it(tmp_path):
    r = _run(tmp_path, FAKE)
    assert r.returncode == 0 and r.stdout.strip() == "fake-groq"
    assert "fake-groq" not in r.stderr and "fakepw" not in r.stderr
    assert "missing=" in r.stderr


def test_missing_name_refuses_without_flag(tmp_path):
    r = _run(tmp_path, "GROQ_API_KEY=fake-groq\n")
    assert r.returncode == rwe.EXIT_USAGE and "refusing" in r.stderr
    assert "fake-groq" not in r.stderr


def test_allow_missing_runs(tmp_path):
    r = _run(tmp_path, "GROQ_API_KEY=fake-groq\n", "--allow-missing")
    assert r.returncode == 0 and r.stdout.strip() == "fake-groq"


def test_unreadable_file_and_no_command(tmp_path):
    assert rwe.main(["--env-file", str(tmp_path / "nope.env"), "--", "true"]) == rwe.EXIT_USAGE
    f = tmp_path / "e.env"
    f.write_text(FAKE)
    assert rwe.main(["--env-file", str(f)]) == rwe.EXIT_USAGE

"""Static checks of the deploy files (item #73): Dockerfile, .dockerignore, start.sh. Offline; no Docker needed."""
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _lines(name):
    return [l.strip() for l in (ROOT / name).read_text().splitlines() if l.strip() and not l.strip().startswith("#")]


def test_dockerignore_keeps_secrets_and_private_files_out():
    ignore = set(_lines(".dockerignore"))
    for pattern in (".env", ".env.*", "*-prv", "*-prv.*", "*-prv/", ".git/", ".venv/", "requirements-dev.txt"):
        assert pattern in ignore, pattern


def test_dockerignore_does_not_drop_runtime_files():
    ignore = set(_lines(".dockerignore"))
    for needed in ("config/", "vera/", "pages/", "eval_results/", "main.py", "streamlit_app.py", "start.sh",
                   "requirements.txt"):
        assert needed not in ignore and needed.rstrip("/") not in ignore, needed


def test_dockerfile_runtime_only_non_root_no_secrets():
    text = (ROOT / "Dockerfile").read_text()
    body = "\n".join(_lines("Dockerfile"))
    assert "pip install --no-cache-dir -r requirements.txt" in body
    assert "requirements-dev" not in body
    assert "\nUSER " in "\n" + body
    assert 'CMD ["./start.sh", "api"]' in body
    for marker in ("OPENAI_API_KEY=", "VERA_API_KEY=", "DB_URL=", "sk-"):
        assert marker not in text, marker


def test_start_script_is_executable_single_worker_and_honours_port():
    script = ROOT / "start.sh"
    assert os.access(script, os.X_OK)
    text = script.read_text()
    assert 'PORT="${PORT:-8000}"' in text
    assert "--workers 1" in text  # caps are in-memory per process
    assert text.count("exec ") == 2
    assert "forwarded-allow-ips" not in text  # client identity is vera/public_mode.py's job


def test_start_script_rejects_unknown_mode():
    result = subprocess.run([str(ROOT / "start.sh"), "nope"], capture_output=True, text=True)
    assert result.returncode == 2
    assert "usage" in result.stderr

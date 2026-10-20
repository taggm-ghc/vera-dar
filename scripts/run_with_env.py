"""Run one command with a few named variables from an env file placed in its environment.

Mirrors production, where the host swivels the live .env* values into the process
environment: the runner reads only os.environ and never loads a file itself. Locally this
wrapper does the swivel for one command and nothing more.

- Only the allow-listed names are copied; every other line of the file is ignored.
- The file is parsed as NAME=value text, never executed or sourced.
- Values are never printed. Only names (set or missing) go to stderr.
- A missing allow-listed name is an error unless --allow-missing is given.

Usage:
    venv/bin/python scripts/run_with_env.py [--env-file PATH] [--allow-missing] -- CMD [ARGS...]
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vera import env_files  # noqa: E402  (PLAN-ENV: the only place the env location is defined)

# The three values the runner needs from its environment: the two keys R1 approved plus the rw DB URL.
# Why no search key: M2 now uses a keyless search (p3m3/vera-plan-m2-keyless-search.md).
ALLOWED_NAMES: tuple[str, ...] = ("GROQ_API_KEY", "OPENAI_API_KEY", "VERA_DB_URL_RW")
# Why: R/config/env/.env-vera is the single local file; callers can point elsewhere with --env-file.
DEFAULT_ENV_FILE = env_files.VERA_ENV
EXIT_USAGE = 2


def parse_env_text(text: str, names: tuple[str, ...] = ALLOWED_NAMES) -> dict[str, str]:
    """Return {name: value} for allow-listed names only; the last assignment wins."""
    found: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        name, _, value = line.partition("=")
        name = name.strip()
        if name not in names:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if value:
            found[name] = value
    return found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("cmd", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    cmd = args.cmd[1:] if args.cmd[:1] == ["--"] else args.cmd
    if not cmd:
        print("run_with_env: no command given (put it after --)", file=sys.stderr)
        return EXIT_USAGE
    try:
        if args.env_file == DEFAULT_ENV_FILE:   # the default location gets the full E3-E5 preflight
            env_files.preflight_dir()
            env_files.preflight_file(args.env_file)
    except env_files.EnvFilesError as e:
        print(f"run_with_env: refused ({e.code}) for {args.env_file.name}", file=sys.stderr)
        return EXIT_USAGE
    try:
        text = args.env_file.read_text(encoding="utf-8")
    except OSError as e:
        print(f"run_with_env: cannot read env file {args.env_file.name}: {type(e).__name__}", file=sys.stderr)
        return EXIT_USAGE
    values = parse_env_text(text)
    missing = [n for n in ALLOWED_NAMES if n not in values]
    print("run_with_env: set=" + ",".join(n for n in ALLOWED_NAMES if n in values) +
          " missing=" + ",".join(missing), file=sys.stderr)
    if missing and not args.allow_missing:
        print("run_with_env: refusing to run with missing names (use --allow-missing to override)",
              file=sys.stderr)
        return EXIT_USAGE
    env = dict(os.environ)
    env.update(values)
    os.execvpe(cmd[0], cmd, env)  # replaces this process; nothing after this line runs
    return 0  # pragma: no cover


if __name__ == "__main__":
    sys.exit(main())

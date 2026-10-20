"""PLAN-ENV s1.1: the one sanctioned entry point for loading VERA-DAR's env file (R/config/env/.env-vera).

Paths come from this file's own location only: no cwd, no $HOME, no environment override. This is the only module
allowed to climb to the R root (parents[2]). Contexts that load no file (production, test, agent) never stat ENV_DIR,
so a container with no R/config (Render) boots normally. Contexts that load a file (local_dev, agent_explicit) are
preflighted, then handed to env_precedence.load_env(root=ENV_DIR).

EnvFilesError carries a code (E1-E6) and a names-only message, raised `from None`: never a value, a length or a hash.
  E1 ENV_DIR missing, a symlink, not owned by the euid, or group/other-writable
  E2 R/vera is not a VERA-DAR checkout
  E3 the env file is missing (required)
  E4 the env file is a symlink or not a regular file
  E5 the env file mode is not exactly 0600
  E6 admin-floor names are present in the file
"""
from __future__ import annotations

import fnmatch
import os
import stat
from pathlib import Path
from typing import MutableMapping

from vera import env_precedence

REPO = Path(__file__).resolve().parents[1]      # R/vera
R_ROOT = Path(__file__).resolve().parents[2]    # R: the only sanctioned parents[2] climb
ENV_DIR = R_ROOT / "config" / "env"
VERA_ENV = ENV_DIR / ".env-vera"
INTERN_ENV = ENV_DIR / ".env-intern"            # fingerprinted only; VERA-DAR never loads it (D-ENV-4)

ADMIN_FLOOR_PATTERNS = ("DB_ADMIN_*", "*_DBADMIN*", "VERA_DB_URL_ADMIN", "VERA_MIGRATE_PRODUCTION")
FILE_MODE = 0o600


class EnvFilesError(RuntimeError):
    """Fail-closed refusal. `code` is E1..E6; the message names only the code and a file/dir basename."""

    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code


def _fail(code: str, message: str):
    raise EnvFilesError(code, message) from None


def preflight_dir(env_dir: Path | None = None) -> None:
    """E1 (dir) and E2 (checkout) checks. Looks at ENV_DIR only when called."""
    d = Path(ENV_DIR if env_dir is None else env_dir)
    if not (REPO / "main.py").is_file() or not (REPO / "vera" / "env_precedence.py").is_file():
        _fail("E2", "R/vera is not a VERA-DAR checkout")
    try:
        st = os.lstat(d)
    except OSError:
        _fail("E1", f"{d.name} is missing or unreadable")
    if stat.S_ISLNK(st.st_mode) or not stat.S_ISDIR(st.st_mode):
        _fail("E1", f"{d.name} is a symlink or not a directory")
    if st.st_uid != os.geteuid():
        _fail("E1", f"{d.name} is not owned by the current user")
    if st.st_mode & 0o022:
        _fail("E1", f"{d.name} is group- or other-writable")


def preflight_file(path: Path | None = None) -> None:
    """E3, E4, E5 checks on one env file (lstat only; contents untouched)."""
    p = Path(VERA_ENV if path is None else path)
    try:
        st = os.lstat(p)
    except FileNotFoundError:
        _fail("E3", f"{p.name} is missing")
    except OSError:
        _fail("E3", f"{p.name} cannot be examined")
    if stat.S_ISLNK(st.st_mode) or not stat.S_ISREG(st.st_mode):
        _fail("E4", f"{p.name} is a symlink or not a regular file")
    if stat.S_IMODE(st.st_mode) != FILE_MODE:
        _fail("E5", f"{p.name} mode is not 0600")


def _check_admin_floor(env_dir: Path, name: str, cfg: dict) -> None:
    """E6: names only. Reuses env_precedence's fd-level reader and parser; nothing is applied."""
    text = env_precedence._read_listed_file(env_dir, name, cfg["file_limits"])
    values, notes = env_precedence._parse(name, text or "", "warn")
    names = set(values) | set(notes)
    bad = sorted(n for n in names if any(fnmatch.fnmatchcase(n.upper(), p) for p in ADMIN_FLOOR_PATTERNS))
    if bad:
        _fail("E6", f"{name} holds admin-floor names {bad}")


def loads_a_file(cfg: dict, context: str, selected_by: str) -> bool:
    files = cfg["contexts"][context]["files"]
    if selected_by == "agent_explicit":
        files = [f for f in files if f["path"] in cfg["agent_explicit"]["files"]]
    return bool(files)


def load_vera_env(environ: MutableMapping[str, str] | None = None, *, log: bool = True, modules=None, cfg: dict | None = None):
    """Select the context, preflight when it loads a file, then env_precedence.load_env(root=ENV_DIR)."""
    env = os.environ if environ is None else environ
    cfg = env_precedence.load_config() if cfg is None else cfg
    context, how = env_precedence.select_context(cfg, env, modules)
    if loads_a_file(cfg, context, how):
        preflight_dir()
        preflight_file(VERA_ENV)
        _check_admin_floor(ENV_DIR, VERA_ENV.name, cfg)
    return env_precedence.load_env(environ, cfg=cfg, root=ENV_DIR, modules=modules, log=log)

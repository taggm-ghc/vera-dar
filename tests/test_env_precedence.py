"""PRB-DARINT-002 tests for vera/env_precedence.py. Synthetic files under tmp_path only; never the real .env files.
Every environment here is a plain dict built by the test, so os.environ is never read or written."""
from __future__ import annotations

import copy
import json
import logging
import os

import pytest

from vera import env_precedence as ep

SENT = "sentinel-value-7f3a"          # must never appear in an audit or a log line
DB_SENT = "postgresql://u:" + SENT + "@sentinel.invalid/db"


@pytest.fixture(autouse=True)
def _private_umask():
    old = os.umask(0o077)       # synthetic files get 0600, as a real .env should
    yield
    os.umask(old)


@pytest.fixture
def cfg():
    return ep.load_config()


def L(env, cfg, root, **kw):
    return ep.load_env(env, cfg=cfg, root=root, modules={}, log=False, **kw)


EXPL = {"VERA_ENV_CONTEXT": "local_dev"}


def _repo(tmp_path, text=None, name=".env-vera"):
    root = tmp_path / "parent" / "repo"
    root.mkdir(parents=True)
    if text is not None:
        (root / name).write_text(text)
    return root


# --- shipped config ------------------------------------------------------------------------------------------
def test_shipped_config_shape(cfg):
    assert {c: [f["path"] for f in v["files"]] for c, v in cfg["contexts"].items()} == {
        "production": [], "test": [], "agent": [], "local_dev": [".env-vera"]}
    assert cfg["contexts"]["local_dev"]["files"][0]["required"] is True          # D-ENV-1
    assert cfg["agent_explicit"]["files"] == [".env-vera"]
    assert list(cfg["restricted_keys"]["allowed"]["local_dev"]) == [".env-vera"]
    assert cfg["precedence"]["interpolate"] is False and cfg["default_context"] == "local_dev"


# --- branch 1: context ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("env,expected", [
    ({"RENDER": "true"}, ("production", "render")),
    ({"RENDER": "true", "VERA_ENV_CONTEXT": "production"}, ("production", "render")),
    ({"VERA_TEST_RUNNER": ""}, ("test", "test_marker")),
    ({"VERA_TEST_RUNNER": "1"}, ("test", "test_marker")),
    ({"VERA_TEST_BOUNDARY_CHILD": "1", "VERA_ENV_CONTEXT": "test"}, ("test", "test_marker")),
    ({"VERA_ENV_CONTEXT": "local_dev"}, ("local_dev", "explicit")),
    ({"VERA_ENV_CONTEXT": "agent", "CLAUDECODE": "1"}, ("agent", "explicit")),
    ({"VERA_ENV_CONTEXT": "production", "AI_AGENT": "1"}, ("production", "explicit")),
    ({"CLAUDECODE": "1", "VERA_ENV_CONTEXT": "local_dev"}, ("local_dev", "agent_explicit")),
    ({"AI_AGENT": "1", "VERA_ENV_CONTEXT": "local_dev"}, ("local_dev", "agent_explicit")),
    ({"VERA_ENV_CONTEXT": "agent"}, ("agent", "explicit")),
    ({"CLAUDECODE": "1"}, ("agent", "agent_marker")),
    ({"AI_AGENT": ""}, ("agent", "agent_marker")),
    ({}, ("local_dev", "default")),
])
def test_select_context(cfg, env, expected):
    assert ep.select_context(cfg, env, {}) == expected


@pytest.mark.parametrize("env,code", [
    ({"RENDER": "true", "VERA_ENV_CONTEXT": "local_dev"}, "F1"),
    ({"VERA_TEST_RUNNER": "1", "VERA_ENV_CONTEXT": "local_dev"}, "F1"),
    ({"VERA_ENV_CONTEXT": "staging"}, "F2"),
    ({"VERA_ENV_CONTEXT": ""}, "F2"),
    ({"VERA_ENV_CONTEXT": "Local_Dev"}, "F2"),
    ({"VERA_ENV_CONTEXT": " local_dev"}, "F2"),
    ({"RENDER": "TRUE"}, "F1"),
    ({"RENDER": "false"}, "F1"),
    ({"RENDER": ""}, "F1"),
    ({"CLAUDECODE": "1", "VERA_ENV_CONTEXT": "staging"}, "F2"),
    ({"RENDER": "true", "VERA_ENV_CONTEXT": "nope"}, "F2"),
])
def test_context_fail_closed(cfg, env, code):
    with pytest.raises(ep.EnvPrecedenceError) as e:
        ep.select_context(cfg, env, {})
    assert e.value.code == code


# --- branch 2: files -----------------------------------------------------------------------------------------
def test_no_upward_search(cfg, tmp_path):
    # ENV-A2: .env-vera is required (D-ENV-1), so a missing file is F5 instead of status "missing".
    root = _repo(tmp_path)
    (tmp_path / "parent" / ".env-vera").write_text(f"PARENT_ONLY={SENT}\n")
    (tmp_path / ".env-vera").write_text(f"HOME_ONLY={SENT}\n")
    env = {}
    with pytest.raises(ep.EnvPrecedenceError) as e:
        L(env, cfg, root)
    assert e.value.code == "F5" and env == {} and SENT not in str(e.value)


@pytest.mark.parametrize("ctx_env", [{"RENDER": "true"}, {"VERA_TEST_RUNNER": "1"}, {"CLAUDECODE": "1"},
                                     {"VERA_ENV_CONTEXT": "agent"}, {"VERA_ENV_CONTEXT": "production"}])
def test_fileless_contexts_never_open_files(cfg, tmp_path, ctx_env, monkeypatch):
    root = _repo(tmp_path, f"FROM_FILE={SENT}\nINTERNAL_DB_URL={DB_SENT}\n")
    opened = []
    monkeypatch.setattr(ep, "_read_listed_file", lambda r, rel, lim: opened.append(rel))
    env = dict(ctx_env)
    audit = L(env, cfg, root)
    assert env == ctx_env and audit.files == [] and opened == []


def test_test_boundary_module_selects_test(cfg):
    assert ep.select_context(cfg, {}, {"tests.vera_test_boundary": object()}) == ("test", "test_marker")
    with pytest.raises(ep.EnvPrecedenceError):
        ep.select_context(cfg, EXPL, {"tests.vera_test_boundary": object()})


def test_in_session_default_is_test_context(cfg, tmp_path):
    root = _repo(tmp_path, f"A={SENT}\n")
    env = {}
    audit = ep.load_env(env, cfg=cfg, root=root, log=False)       # real sys.modules: the boundary is loaded
    assert audit.context == "test" and env == {}
    with pytest.raises(ep.EnvPrecedenceError) as e:               # explicit local_dev cannot widen a test session
        ep.load_env(dict(EXPL), cfg=cfg, root=root, log=False)
    assert e.value.code == "F1"


def test_default_context_skips_db_urls_by_name(cfg, tmp_path):
    root = _repo(tmp_path, f"OPENAI_API_KEY={SENT}\nVERA_DB_URL_RO={DB_SENT}\n")
    env = {}
    audit = L(env, cfg, root)
    assert env == {"OPENAI_API_KEY": SENT} and list(audit.skipped) == ["VERA_DB_URL_RO"]


def test_local_dev_loads_repo_root_env(cfg, tmp_path):
    root = _repo(tmp_path, f"# c\nOPENAI_API_KEY={SENT}\nexport VERA_DB_URL_RO='{DB_SENT}'\n")
    env = dict(EXPL)
    audit = L(env, cfg, root)
    assert env == {**EXPL, "OPENAI_API_KEY": SENT, "VERA_DB_URL_RO": DB_SENT}
    assert audit.sources == {"VERA_ENV_CONTEXT": "process", "OPENAI_API_KEY": ".env-vera",
                             "VERA_DB_URL_RO": ".env-vera"}
    assert sorted(audit.applied) == ["OPENAI_API_KEY", "VERA_DB_URL_RO"]


# --- branch 3: precedence ------------------------------------------------------------------------------------
def test_file_wins_over_inherited(cfg, tmp_path, caplog):
    # ENV-A2 (C3/OR-3): was test_process_env_wins. The file now wins; the override is logged by NAME only.
    root = _repo(tmp_path, f"OPENAI_API_KEY={SENT}\nGROQ_API_KEY={SENT}\n")
    env = {"OPENAI_API_KEY": "from-process-" + SENT[::-1]}
    with caplog.at_level(logging.WARNING, logger="vera.env_precedence"):
        audit = L(env, cfg, root)
    assert env["OPENAI_API_KEY"] == SENT and env["GROQ_API_KEY"] == SENT
    assert audit.sources["OPENAI_API_KEY"] == ".env-vera" and audit.overridden == ["OPENAI_API_KEY"]
    assert audit.shadowed == {}
    warns = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert warns == ["env: file .env-vera overrides inherited OPENAI_API_KEY"]
    assert SENT not in caplog.text and SENT[::-1] not in caplog.text


def test_empty_inherited_value_is_overridden_by_file(cfg, tmp_path):
    # ENV-A2 (C3): was test_empty_process_value_still_wins.
    root = _repo(tmp_path, f"VERA_API_KEY={SENT}\n")
    env = {"VERA_API_KEY": ""}
    audit = L(env, cfg, root)
    assert env["VERA_API_KEY"] == SENT and audit.overridden == ["VERA_API_KEY"]


def test_same_inherited_value_is_not_an_override(cfg, tmp_path, caplog):
    root = _repo(tmp_path, f"OPENAI_API_KEY={SENT}\n")
    env = {"OPENAI_API_KEY": SENT}
    with caplog.at_level(logging.WARNING, logger="vera.env_precedence"):
        audit = L(env, cfg, root)
    assert env["OPENAI_API_KEY"] == SENT and audit.overridden == [] and "overrides inherited" not in caplog.text


def _two_file_cfg(cfg, mode):
    c = copy.deepcopy(cfg)
    c["contexts"]["local_dev"]["files"] = [{"path": ".env-vera", "required": False},
                                          {"path": ".env.second", "required": False}]
    c["precedence"]["cross_file_conflict"] = mode
    return c


def test_first_listed_file_wins(cfg, tmp_path):
    root = _repo(tmp_path, "A=first\n")
    (root / ".env.second").write_text("A=second\nB=second\n")
    env = {}
    audit = L(env, _two_file_cfg(cfg, "first_wins"), root)
    assert env == {"A": "first", "B": "second"}
    assert audit.shadowed["A"] == [{"source": ".env.second", "by": ".env-vera"}]


def test_cross_file_conflict_error_mode(cfg, tmp_path):
    root = _repo(tmp_path, "A=first\n")
    (root / ".env.second").write_text("A=second\n")
    env = {}
    with pytest.raises(ep.EnvPrecedenceError) as e:
        L(env, _two_file_cfg(cfg, "error"), root)
    assert e.value.code == "F6" and env == {}


def test_values_are_literal_no_interpolation(cfg, tmp_path):
    root = _repo(tmp_path, "A=${HOME}/x\n")
    env = {"HOME": "/h"}
    audit = L(env, cfg, root)
    assert env["A"] == "${HOME}/x" and audit.notes["A"] == ["interpolation_not_applied@.env-vera"]


def test_bare_name_without_value_is_ignored(cfg, tmp_path):
    root = _repo(tmp_path, "BARE\nA=1\n")
    env = {}
    audit = L(env, cfg, root)
    assert env == {"A": "1"} and audit.notes["BARE"] == ["no_value@.env-vera"]


# --- branch 4: fail closed -----------------------------------------------------------------------------------
@pytest.mark.parametrize("text,code", [
    ("A=1\nA=2\n", "F3"),                                  # in-file duplicate
    ("A=1\nBARE\nBARE=2\n", "F3"),
    ("A=1\nnot a valid line\n", "F3"),                      # unparseable
    (f"INTERNAL_DB_URL={DB_SENT}\n", "F4"),                 # Render-only DB URL from a file
    (f"OK=1\nPGPASSWORD={SENT}\n", "F7"),                  # ENV-A2: PG* is now file-denied (C3), was F4
    (f"DATABASE_URL={DB_SENT}\n", "F4"),
    ("MY_DB_URL_X\n", "F4"),                                # restricted even without a value
])
def test_fail_closed_is_atomic(cfg, tmp_path, text, code):
    root = _repo(tmp_path, text)
    env = {"KEEP": "1"}
    with pytest.raises(ep.EnvPrecedenceError) as e:
        L(env, cfg, root)
    assert e.value.code == code and env == {"KEEP": "1"}
    assert SENT not in str(e.value)


def test_restricted_key_not_allowed_from_other_listed_file(cfg, tmp_path):
    root = _repo(tmp_path, "A=1\n")
    (root / ".env.second").write_text(f"VERA_DB_URL_RO={DB_SENT}\n")
    with pytest.raises(ep.EnvPrecedenceError) as e:
        L({}, _two_file_cfg(cfg, "first_wins"), root)
    assert e.value.code == "F4"


def test_symlink_refused(cfg, tmp_path):
    root = _repo(tmp_path)
    outside = tmp_path / "outside.env"
    outside.write_text(f"A={SENT}\n")
    (root / ".env-vera").symlink_to(outside)
    with pytest.raises(ep.EnvPrecedenceError) as e:
        L({}, cfg, root)
    assert e.value.code == "F3"


def test_dangling_symlink_refused_not_missing(cfg, tmp_path):
    root = _repo(tmp_path)
    (root / ".env-vera").symlink_to(tmp_path / "nowhere")
    with pytest.raises(ep.EnvPrecedenceError):
        L({}, cfg, root)


def test_symlinked_directory_component_refused(cfg, tmp_path):
    root = _repo(tmp_path)
    real = tmp_path / "realsub"
    real.mkdir()
    (real / "x.env").write_text("A=1\n")
    (root / "sub").symlink_to(real)
    c = copy.deepcopy(cfg)
    c["contexts"]["local_dev"]["files"] = [{"path": "sub/x.env", "required": False}]
    c["restricted_keys"]["allowed"]["local_dev"] = {}
    c["agent_explicit"]["files"] = ["sub/x.env"]
    with pytest.raises(ep.EnvPrecedenceError) as e:
        L({}, c, root)
    assert e.value.code == "F3"


def test_symlinked_repo_root_itself_is_fine(cfg, tmp_path):
    root = _repo(tmp_path, "A=1\n")
    link = tmp_path / "rootlink"
    link.symlink_to(root)
    env = {}
    L(env, cfg, link)
    assert env == {"A": "1"}


def test_directory_in_place_of_file_refused(cfg, tmp_path):
    root = _repo(tmp_path)
    (root / ".env-vera").mkdir()
    with pytest.raises(ep.EnvPrecedenceError) as e:
        L({}, cfg, root)
    assert e.value.code == "F3"


def test_non_utf8_refused(cfg, tmp_path):
    root = _repo(tmp_path)
    (root / ".env-vera").write_bytes(b"A=\xff\xfe\n")
    with pytest.raises(ep.EnvPrecedenceError) as e:
        L({}, cfg, root)
    assert e.value.code == "F3"


def test_required_missing_fails_closed(cfg, tmp_path):
    root = _repo(tmp_path)
    c = copy.deepcopy(cfg)
    c["contexts"]["local_dev"]["files"][0]["required"] = True
    with pytest.raises(ep.EnvPrecedenceError) as e:
        L({}, c, root)
    assert e.value.code == "F5"


@pytest.mark.parametrize("edit", [
    lambda c: c["contexts"]["local_dev"]["files"].append({"path": "/home/x/.env", "required": False}),
    lambda c: c["contexts"]["local_dev"]["files"].append({"path": "../.env", "required": False}),
    lambda c: c["contexts"]["local_dev"]["files"].append({"path": "~/.env", "required": False}),
    lambda c: c["contexts"]["local_dev"]["files"].append({"path": ".env-vera", "required": False}),
    lambda c: c["contexts"]["production"]["files"].append({"path": ".env", "required": False}),
    lambda c: c["contexts"]["test"]["files"].append({"path": ".env", "required": False}),
    lambda c: c["precedence"].__setitem__("interpolate", True),
    lambda c: c["precedence"].__setitem__("cross_file_conflict", "last_wins"),
    lambda c: c.__setitem__("override", True),
    lambda c: c.pop("restricted_keys"),
    lambda c: c.__setitem__("default_context", "nope"),
    lambda c: c["restricted_keys"]["allowed"]["local_dev"].__setitem__(".env.unlisted", ["X"]),
])
def test_bad_config_fails_closed(cfg, edit):
    c = copy.deepcopy(cfg)
    edit(c)
    with pytest.raises(ep.EnvPrecedenceError) as e:
        ep.validate_config(c)
    assert e.value.code == "F0"


def test_load_config_unreadable(tmp_path):
    with pytest.raises(ep.EnvPrecedenceError) as e:
        ep.load_config(tmp_path / "absent.json")
    assert e.value.code == "F0"
    (tmp_path / "bad.json").write_text("{not json")
    with pytest.raises(ep.EnvPrecedenceError):
        ep.load_config(tmp_path / "bad.json")


# --- branch 5: audit, names only -----------------------------------------------------------------------------
def test_audit_and_log_hold_names_never_values(cfg, tmp_path, caplog):
    root = _repo(tmp_path, f"OPENAI_API_KEY={SENT}\nVERA_DB_URL_WO={DB_SENT}\nA=${{X}}{SENT}\n")
    env = {**EXPL, "VERA_API_KEY": SENT + "-proc"}
    with caplog.at_level(logging.INFO, logger="vera.env_precedence"):
        audit = ep.load_env(env, cfg=cfg, root=root, modules={})
    blob = json.dumps(audit.as_dict()) + caplog.text + repr(audit)
    assert SENT not in blob and "sentinel.invalid" not in blob
    assert "OPENAI_API_KEY" in caplog.text and "context=local_dev" in caplog.text
    assert "VERA_API_KEY" not in caplog.text          # process-only names are not logged


def test_resolve_does_not_mutate(cfg, tmp_path):
    root = _repo(tmp_path, "A=1\n")
    env = {"B": "2"}
    to_set, audit = ep.resolve(env, cfg=cfg, root=root, modules={})
    assert to_set == {"A": "1"} and env == {"B": "2"} and audit.applied == []


def test_default_root_is_repo_root_not_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert ep.REPO_ROOT == ep.CONFIG_PATH.parent.parent and (ep.REPO_ROOT / "vera" / "env_precedence.py").is_file()
    assert os.getcwd() != str(ep.REPO_ROOT)


# --- vetting round (PRB-DARINT-002-VET) ----------------------------------------------------------------------
@pytest.mark.parametrize("text", ["RENDER=true\n", "VERA_ENV_CONTEXT=production\n", "render_x=1\n",
                                  "VERA_TEST_RUNNER=1\n", "CLAUDECODE=0\n", "PATH=/x\n", "PYTHONPATH=/x\n",
                                  "LD_PRELOAD=/x\n", "HOME=/x\n"])
def test_control_names_from_file_fail_closed(cfg, tmp_path, text):
    root = _repo(tmp_path, text)
    env = {}
    with pytest.raises(ep.EnvPrecedenceError) as e:
        L(env, cfg, root)
    assert e.value.code == "F7" and env == {}


def test_restricted_match_is_case_insensitive(cfg, tmp_path):
    root = _repo(tmp_path, f"vera_db_url_ro={DB_SENT}\n")
    with pytest.raises(ep.EnvPrecedenceError) as e:
        L(dict(EXPL), cfg, root)
    assert e.value.code == "F4"


@pytest.mark.parametrize("text", [f"A.B={SENT}\n", f"\u00e9={SENT}\n", f"1A={SENT}\n"])
def test_invalid_names_fail_closed_without_value_in_message(cfg, tmp_path, text):
    root = _repo(tmp_path, text)
    with pytest.raises(ep.EnvPrecedenceError) as e:
        L({}, cfg, root)
    assert e.value.code == "F3" and SENT not in str(e.value)


def test_error_messages_never_carry_the_raw_line(cfg, tmp_path):
    root = _repo(tmp_path, f"A={SENT}\nA={SENT}2\n")
    with pytest.raises(ep.EnvPrecedenceError) as e:
        L({}, cfg, root)
    assert SENT not in str(e.value) and "line 2" in str(e.value)


def test_bom_size_and_mode_refused(cfg, tmp_path):
    root = _repo(tmp_path)
    f = root / ".env-vera"
    f.write_bytes(b"\xef\xbb\xbfA=1\n")
    with pytest.raises(ep.EnvPrecedenceError, match="byte-order"):
        L({}, cfg, root)
    f.write_text("A=" + "x" * (cfg["file_limits"]["max_bytes"] + 1) + "\n")
    with pytest.raises(ep.EnvPrecedenceError, match="larger than"):
        L({}, cfg, root)
    f.write_text("A=1\n")
    f.chmod(0o664)
    with pytest.raises(ep.EnvPrecedenceError, match="writable"):
        L({}, cfg, root)
    f.chmod(0o600)
    env = {}
    L(env, cfg, root)
    assert env == {"A": "1"}


def test_fifo_refused_without_blocking(cfg, tmp_path):
    root = _repo(tmp_path)
    os.mkfifo(root / ".env-vera")
    with pytest.raises(ep.EnvPrecedenceError) as e:
        L({}, cfg, root)
    assert e.value.code == "F3"


def test_duplicate_warn_transition_mode_is_last_wins(cfg, tmp_path):
    c = copy.deepcopy(cfg)
    c["precedence"]["duplicate_key_in_file"] = "warn"
    root = _repo(tmp_path, "A=1\nA=2\n")
    env = {}
    audit = L(env, c, root)
    assert env == {"A": "2"} and audit.notes["A"] == ["duplicate_last_wins@.env-vera"]


def test_parser_version_pinned(cfg, tmp_path):
    import dotenv.version
    assert dotenv.version.__version__ == cfg["parser_version"]
    c = copy.deepcopy(cfg)
    c["parser_version"] = "0.0.0"
    root = _repo(tmp_path, "A=1\n")
    with pytest.raises(ep.EnvPrecedenceError) as e:
        L({}, c, root)
    assert e.value.code == "F0"
    assert L({"RENDER": "true"}, c, root).context == "production"   # file-less contexts never touch the parser


def test_parser_contract(cfg, tmp_path):
    root = _repo(tmp_path, "export A='q w'\nB=\"x\\ny\"\nC= z # c\nD=\n")
    env = {}
    L(env, cfg, root)
    assert env == {"A": "q w", "B": "x\ny", "C": "z", "D": ""}


def test_process_latch_returns_first_audit(monkeypatch):
    first = ep.EnvAudit(context="local_dev", selected_by="default")
    monkeypatch.setattr(ep, "_PROCESS_AUDIT", [first])
    monkeypatch.setattr(ep, "resolve", lambda *a, **k: (_ for _ in ()).throw(AssertionError("re-resolved")))
    assert ep.load_env() is first and ep.load_env(os.environ) is first


def test_check_files_reports_names_only(cfg, tmp_path):
    root = _repo(tmp_path, f"A={SENT}\nA=2\nB=${{X}}\nINTERNAL_DB_URL={DB_SENT}\nRENDER=true\n")
    rep = ep.check_files(cfg=cfg, root=root)
    assert rep == {".env-vera": {"status": "readable", "duplicates": ["A"], "interpolation": ["B"],
                            "control_names": ["RENDER"], "restricted_not_allowed": ["INTERNAL_DB_URL"]}}
    assert SENT not in json.dumps(rep)


# --- verifier round (PRB-002 REVISE: H1, D6, filename denylist, R1) ----------------------------------------------
def test_hardlink_to_home_env_refused(cfg, tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    (home / ".env").write_text(f"FORBIDDEN_HOME={SENT}\n")
    root = _repo(tmp_path)
    os.link(home / ".env", root / ".env-vera")
    env = {}
    with pytest.raises(ep.EnvPrecedenceError, match="hard links") as e:
        L(env, cfg, root)
    assert e.value.code == "F3" and env == {}


def test_hardlink_to_prv_env_refused(cfg, tmp_path):
    root = _repo(tmp_path, f"FORBIDDEN_PRV={SENT}\n", name="-prv.env")
    os.link(root / "-prv.env", root / ".env-vera")
    with pytest.raises(ep.EnvPrecedenceError) as e:
        L({}, cfg, root)
    assert e.value.code == "F3"


def test_home_env_inode_refused_when_root_is_home(cfg, tmp_path, monkeypatch):
    # ENV-A2: the shipped file is .env-vera, so a config listing ".env" is built to keep exercising the ~/.env check.
    root = _repo(tmp_path, f"A={SENT}\n", name=".env")
    c = copy.deepcopy(cfg)
    c["contexts"]["local_dev"]["files"] = [{"path": ".env", "required": True}]
    c["restricted_keys"]["allowed"]["local_dev"] = {}
    c["agent_explicit"]["files"] = [".env"]
    monkeypatch.setenv("HOME", str(root))
    with pytest.raises(ep.EnvPrecedenceError, match="~/.env") as e:
        L({}, c, root)
    assert e.value.code == "F3"


@pytest.mark.parametrize("bad", [".", "./.env", "sub/./.env", "-prv.env", "x-prv.env", "sub/a-prv.env",
                                 "a-prv.env.bak", ".env.render.api", ".ENV.RENDER.ui", ".env.example",
                                 ".env.db-accounts", ".env-intern", ".ENV-INTERN"])
def test_dot_and_forbidden_filenames_rejected_by_config(cfg, bad):
    c = copy.deepcopy(cfg)
    c["contexts"]["local_dev"]["files"] = [{"path": bad, "required": False}]
    c["restricted_keys"]["allowed"]["local_dev"] = {}
    with pytest.raises(ep.EnvPrecedenceError) as e:
        ep.validate_config(c)
    assert e.value.code == "F0"


def test_cli_main_exit_codes_and_names_only(cfg, tmp_path, capsys):
    root = _repo(tmp_path, "A=1\n")
    assert ep.main(["--check", "--root", str(root)]) == 0
    (root / ".env-vera").write_text(f"A={SENT}\nA=2\nPGHOST={SENT}\n")
    assert ep.main(["--check", "--root", str(root)]) == 1
    (root / ".env-vera").chmod(0o666)
    assert ep.main(["--check", "--root", str(root)]) == 1          # refused (writable) is not clean
    (root / ".env-vera").unlink()
    assert ep.main(["--check", "--root", str(root)]) == 0          # missing is clean
    assert ep.main([]) == 2
    assert SENT not in capsys.readouterr().out


# --- OD-DARINT-AGENTDEV: agent-driven local development (agent_explicit) ------------------------------------------
AGENT_DEV = {"CLAUDECODE": "1", "VERA_ENV_CONTEXT": "local_dev"}
DEV_FILE = (f"OPENAI_API_KEY={SENT}\nGROQ_API_KEY={SENT}\nVERA_API_KEY={SENT}\nVERA_OWNER_KEY={SENT}\n"
            f"EXTERNAL_DB_URL={DB_SENT}\nVERA_DB_URL_RO={DB_SENT}\nVERA_DB_URL_WO={DB_SENT}\nVERA_DB_URL_RW={DB_SENT}\n")
DB_NAMES = {"EXTERNAL_DB_URL", "VERA_DB_URL_RO", "VERA_DB_URL_WO", "VERA_DB_URL_RW"}


def test_agent_explicit_loads_ordinary_dev_keys_only(cfg, tmp_path, caplog):
    root = _repo(tmp_path, DEV_FILE)
    env = dict(AGENT_DEV)
    with caplog.at_level(logging.INFO, logger="vera.env_precedence"):
        audit = ep.load_env(env, cfg=cfg, root=root, modules={})
    assert audit.context == "local_dev" and audit.selected_by == "agent_explicit"
    assert set(env) - set(AGENT_DEV) == {"OPENAI_API_KEY", "GROQ_API_KEY", "VERA_API_KEY"}
    assert set(audit.skipped) == {"VERA_OWNER_KEY"} | DB_NAMES
    assert all(r == "excluded_for_agent@.env-vera" for r in audit.skipped.values())
    assert "agent_explicit" in caplog.text and SENT not in caplog.text + json.dumps(audit.as_dict())


@pytest.mark.parametrize("marker", ["CLAUDECODE", "AI_AGENT"])
def test_agent_explicit_never_gets_owner_key_or_db_urls(cfg, tmp_path, marker):
    root = _repo(tmp_path, DEV_FILE + "PGHOST_X=1\n".replace("PGHOST_X=1", "OTHER=1"))
    env = {marker: "", "VERA_ENV_CONTEXT": "local_dev"}
    L(env, cfg, root)
    assert "VERA_OWNER_KEY" not in env and not (DB_NAMES & set(env)) and env["OTHER"] == "1"


def test_agent_explicit_still_fails_closed_on_disallowed_db_name(cfg, tmp_path):
    root = _repo(tmp_path, f"INTERNAL_DB_URL={DB_SENT}\n")
    with pytest.raises(ep.EnvPrecedenceError) as e:
        L(dict(AGENT_DEV), cfg, root)
    assert e.value.code == "F4"


def test_agent_explicit_loads_only_listed_agent_files(cfg, tmp_path):
    c = _two_file_cfg(cfg, "first_wins")
    root = _repo(tmp_path, "A=1\n")
    (root / ".env.second").write_text("B=2\n")
    env = dict(AGENT_DEV)
    audit = L(env, c, root)
    assert env.get("A") == "1" and "B" not in env and [f["path"] for f in audit.files] == [".env-vera"]


def test_human_explicit_path_unchanged(cfg, tmp_path):
    root = _repo(tmp_path, DEV_FILE)
    env = dict(EXPL)
    audit = L(env, cfg, root)
    assert audit.selected_by == "explicit" and audit.skipped == {}
    assert {"VERA_OWNER_KEY", "OPENAI_API_KEY"} | DB_NAMES <= set(env)


def test_default_path_unchanged(cfg, tmp_path):
    root = _repo(tmp_path, DEV_FILE)
    env = {}
    audit = L(env, cfg, root)
    assert audit.selected_by == "default" and set(audit.skipped) == DB_NAMES and "VERA_OWNER_KEY" in env


@pytest.mark.parametrize("edit", [
    lambda c: c["agent_explicit"]["excluded_names"].remove("VERA_OWNER_KEY"),
    lambda c: c["agent_explicit"].__setitem__("excluded_names", ["*DB_URL*", "DATABASE_URL", "PG*"]),
    lambda c: c["agent_explicit"]["excluded_names"].remove("PG*"),
    lambda c: c["agent_explicit"].__setitem__("files", [".env.other"]),
    lambda c: c.pop("agent_explicit"),
])
def test_config_cannot_hand_agent_the_owner_key_or_db_urls(cfg, edit):
    c = copy.deepcopy(cfg)
    edit(c)
    with pytest.raises(ep.EnvPrecedenceError) as e:
        ep.validate_config(c)
    assert e.value.code == "F0"


def test_owner_key_pattern_exclusion_still_counts(cfg):
    c = copy.deepcopy(cfg)
    c["agent_explicit"]["excluded_names"] = [n for n in c["agent_explicit"]["excluded_names"]
                                             if n != "VERA_OWNER_KEY"] + ["VERA_OWNER_*"]
    ep.validate_config(c)


def test_api_kickstart_child_allowlist_never_carries_owner_key_or_db_urls():
    ks = json.loads((ep.REPO_ROOT / "config" / "api_kickstart.json").read_text())
    allow = ks["env_allowlist"]
    assert "VERA_OWNER_KEY" not in allow
    assert not [n for n in allow if ep._match(n, ["*DB_URL*", "DATABASE_URL", "PG*", "*OWNER*"])]
    assert {"CLAUDECODE", "AI_AGENT", "VERA_ENV_CONTEXT"} <= set(allow)
    assert ks["autostart_allowed_contexts"] == ["local_dev", "agent"]


# --- D-DARINT-036: agent process env (owner key + non-RO/WO DB names stripped; RO/WO kept) ----------------------
INHERITED = {"VERA_OWNER_KEY": SENT, "VERA_DB_URL_RW": DB_SENT, "PGHOST": SENT, "PGPASSWORD": SENT,
             "EXTERNAL_DB_URL": DB_SENT, "INTERNAL_DB_URL": DB_SENT, "DATABASE_URL": DB_SENT,
             "VERA_DB_URL_RO": DB_SENT + "ro", "VERA_DB_URL_WO": DB_SENT + "wo", "OPENAI_API_KEY": SENT}
STRIPPED = {"VERA_OWNER_KEY", "VERA_DB_URL_RW", "PGHOST", "PGPASSWORD", "EXTERNAL_DB_URL", "INTERNAL_DB_URL",
            "DATABASE_URL"}


@pytest.mark.parametrize("ctx", [{"CLAUDECODE": "1"}, {"AI_AGENT": ""}, {"CLAUDECODE": "1", "VERA_ENV_CONTEXT": "agent"},
                                 {"CLAUDECODE": "1", "VERA_ENV_CONTEXT": "local_dev"},
                                 {"AI_AGENT": "1", "VERA_ENV_CONTEXT": "production"}])
def test_agent_shell_loses_owner_key_and_non_ro_wo_db_names(cfg, tmp_path, ctx, caplog):
    root = _repo(tmp_path, DEV_FILE)
    env = {**INHERITED, **ctx}
    with caplog.at_level(logging.INFO, logger="vera.env_precedence"):
        audit = ep.load_env(env, cfg=cfg, root=root, modules={})
    assert not (STRIPPED & set(env)) and sorted(audit.stripped) == sorted(STRIPPED)
    assert env["VERA_DB_URL_RO"] == DB_SENT + "ro" and env["VERA_DB_URL_WO"] == DB_SENT + "wo"
    assert env["OPENAI_API_KEY"] == SENT and "VERA_OWNER_KEY" not in audit.sources
    assert "stripped=" in caplog.text and "VERA_OWNER_KEY" in caplog.text
    assert SENT not in caplog.text + json.dumps(audit.as_dict())


HUMAN_STRIPPED = (STRIPPED - {"VERA_OWNER_KEY"}) | {"VERA_DB_URL_RO", "VERA_DB_URL_WO"}


def test_human_local_dev_strips_inherited_db_names(cfg, tmp_path):
    # ENV-A2 (C3): was test_human_path_keeps_inherited_everything. Human local_dev (default or explicit) now strips
    # every inherited STRIP name; non-STRIP names (owner key, API keys) are kept.
    root = _repo(tmp_path, "A=1\n")
    for extra in ({}, dict(EXPL)):
        env = {**INHERITED, **extra}
        audit = L(env, cfg, root)
        assert set(audit.stripped) == HUMAN_STRIPPED and not (HUMAN_STRIPPED & set(env))
        assert env["VERA_OWNER_KEY"] == SENT and env["OPENAI_API_KEY"] == SENT


def test_kickstart_child_never_gets_owner_key(cfg, tmp_path):
    from vera import api_kickstart as ks
    kcfg = ks.load_config()
    parent = {**INHERITED, "CLAUDECODE": "1", "VERA_ENV_CONTEXT": "local_dev", "PATH": "/usr/bin"}
    child = ks._child_env(kcfg, parent)
    assert "VERA_OWNER_KEY" not in child and not [n for n in child if ep._match(n, ["*DB_URL*", "DATABASE_URL", "PG*"])]
    root = _repo(tmp_path, DEV_FILE)
    audit = L(child, cfg, root)                     # the child's own load_env (agent_explicit via inherited marker)
    assert audit.selected_by == "agent_explicit" and "VERA_OWNER_KEY" not in child
    assert not (DB_NAMES & set(child))


@pytest.mark.parametrize("edit", [
    lambda c: c["agent_process_env"]["strip"].remove("VERA_OWNER_KEY"),
    lambda c: c["agent_process_env"]["strip"].remove("PG*"),
    lambda c: c["agent_process_env"]["strip"].remove("*DB_URL*"),
    lambda c: c["agent_process_env"]["strip"].remove("DATABASE_URL"),
    lambda c: c["agent_process_env"]["keep"].append("VERA_OWNER_KEY"),
    lambda c: c["agent_process_env"]["keep"].append("VERA_DB_URL_RW"),
    lambda c: c.pop("agent_process_env"),
    lambda c: c["restricted_keys"]["patterns"].remove("PG*"),
    lambda c: c["restricted_keys"]["patterns"].remove("*DB_URL*"),
    lambda c: c["restricted_keys"].__setitem__("patterns", ["*DB_URL*", "PG*"]),
    lambda c: c["agent_explicit"]["excluded_names"].remove("DATABASE_URL"),
])
def test_config_cannot_drop_floor_names(cfg, edit):
    c = copy.deepcopy(cfg)
    edit(c)
    with pytest.raises(ep.EnvPrecedenceError) as e:
        ep.validate_config(c)
    assert e.value.code == "F0"


# --- G4: agent-marker floor --------------------------------------------------------------------------------------
@pytest.mark.parametrize("markers", [[], ["CLAUDECODE"], ["AI_AGENT"], ["OTHER_AGENT"], ["claudecode", "ai_agent"]])
def test_config_cannot_drop_agent_marker_floor(cfg, markers):
    c = copy.deepcopy(cfg)
    c["agent_markers"] = markers
    with pytest.raises(ep.EnvPrecedenceError) as e:
        ep.validate_config(c)
    assert e.value.code == "F0"


def test_extra_agent_markers_allowed_and_detected(cfg):
    c = copy.deepcopy(cfg)
    c["agent_markers"] = ["CLAUDECODE", "AI_AGENT", "OTHER_AGENT"]
    ep.validate_config(c)
    assert ep.select_context(c, {"OTHER_AGENT": "1"}, {}) == ("agent", "agent_marker")


# --- ENV-A2: PLAN-ENV v2.1 1.2 / 1.3 (C3, OR-3) precedence-change tests ----------------------------------------------
STRIP_INHERITED = {"PGSERVICE": SENT, "KRB5CCNAME": SENT, "GSS_X": SENT, "X_DBADMIN_URL": SENT, "DB_ADMIN_USER": SENT,
                   "VERA_MIGRATE_PRODUCTION": "1", "SSL_CERT_FILE": SENT, "REQUESTS_CA_BUNDLE": SENT,
                   "CURL_CA_BUNDLE": SENT, "HTTP_PROXY": SENT, "https_proxy": SENT, "HTTP_FOO_PROXY": SENT,
                   "all_proxy": SENT, "NO_PROXY": SENT}


def test_default_root_is_env_dir():
    import inspect
    assert ep.ENV_DIR == ep.REPO_ROOT.parent / "config" / "env"
    for fn in (ep.load_env, ep.resolve, ep.check_files):
        assert inspect.signature(fn).parameters["root"].default == ep.ENV_DIR


def test_env_dir_matches_env_files_constant():
    env_files = pytest.importorskip("vera.env_files")
    assert ep.ENV_DIR == env_files.ENV_DIR


def test_explicit_local_dev_strips_then_file_supplies_db_urls(cfg, tmp_path, caplog):
    root = _repo(tmp_path, DEV_FILE)
    env = {**INHERITED, **STRIP_INHERITED, **EXPL}
    with caplog.at_level(logging.WARNING, logger="vera.env_precedence"):
        audit = L(env, cfg, root)
    assert set(audit.stripped) == HUMAN_STRIPPED | set(STRIP_INHERITED)
    assert not (set(STRIP_INHERITED) & set(env)) and "INTERNAL_DB_URL" not in env and "PGHOST" not in env
    assert {n: env[n] for n in DB_NAMES} == {n: DB_SENT for n in DB_NAMES}       # only the file supplies them
    assert {n: audit.sources[n] for n in DB_NAMES} == {n: ".env-vera" for n in DB_NAMES}
    assert sorted(audit.overridden) == ["VERA_DB_URL_RO", "VERA_DB_URL_WO"]      # inherited ...ro / ...wo differed
    assert "env: file .env-vera overrides inherited VERA_DB_URL_RO" in caplog.text
    assert SENT not in caplog.text + json.dumps(audit.as_dict())


def test_default_local_dev_has_no_db_url(cfg, tmp_path):
    root = _repo(tmp_path, DEV_FILE)
    env = {**INHERITED, **STRIP_INHERITED}
    audit = L(env, cfg, root)
    assert audit.selected_by == "default" and set(audit.skipped) == DB_NAMES
    assert not [n for n in env if ep._match(n, ep._INHERITED_STRIP)]
    assert env["VERA_OWNER_KEY"] == SENT and env["OPENAI_API_KEY"] == SENT


@pytest.mark.parametrize("name", ["SSL_CERT_FILE", "SSL_CERT_DIR", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE",
                                  "HTTP_PROXY", "https_proxy", "Http_Foo_Proxy", "ALL_PROXY", "no_proxy",
                                  "PGHOST", "PGPASSFILE", "pgservice", "KRB5CCNAME", "KRB5_CONFIG", "GSSAPI_X",
                                  "DB_ADMIN_URL", "VERA_DBADMIN_PW", "VERA_DB_URL_ADMIN", "VERA_MIGRATE_PRODUCTION"])
@pytest.mark.parametrize("ctx", [{}, EXPL, AGENT_DEV])
def test_tls_proxy_libpq_admin_names_refused_in_file(cfg, tmp_path, name, ctx):
    root = _repo(tmp_path, f"OK=1\n{name}={SENT}\n")
    env = dict(ctx)
    with pytest.raises(ep.EnvPrecedenceError) as e:
        L(env, cfg, root)
    assert e.value.code == "F7" and env == ctx and SENT not in str(e.value)


@pytest.mark.parametrize("ctx", [{"RENDER": "true"}, {"VERA_ENV_CONTEXT": "production"}])
def test_production_context_unchanged(cfg, tmp_path, ctx):
    root = _repo(tmp_path, DEV_FILE)
    env = {**INHERITED, **STRIP_INHERITED, **ctx}
    before = dict(env)
    audit = L(env, cfg, root)
    assert env == before and audit.stripped == [] and audit.files == [] and audit.overridden == []


@pytest.mark.parametrize("ctx", [{"CLAUDECODE": "1"}, {"AI_AGENT": "1", "VERA_ENV_CONTEXT": "agent"}])
def test_agent_context_unchanged(cfg, tmp_path, ctx):
    root = _repo(tmp_path, DEV_FILE)
    env = {**INHERITED, **STRIP_INHERITED, **ctx}
    audit = L(env, cfg, root)
    assert sorted(audit.stripped) == sorted(STRIPPED | {"PGSERVICE"})            # D-DARINT-036 set, as before
    assert set(STRIP_INHERITED) - {"PGSERVICE"} <= set(env) and audit.files == []


def test_agent_explicit_file_wins_for_ordinary_names(cfg, tmp_path, caplog):
    root = _repo(tmp_path, DEV_FILE)
    env = {**AGENT_DEV, "OPENAI_API_KEY": "inherited", "VERA_DB_URL_RO": "ro-inherited"}
    with caplog.at_level(logging.WARNING, logger="vera.env_precedence"):
        audit = L(env, cfg, root)
    assert env["OPENAI_API_KEY"] == SENT and audit.overridden == ["OPENAI_API_KEY"]
    assert env["VERA_DB_URL_RO"] == "ro-inherited" and "VERA_DB_URL_RO" in audit.skipped     # excluded, kept
    assert "VERA_OWNER_KEY" not in env and SENT not in caplog.text


@pytest.mark.parametrize("floor", list(ep._FILE_DENIED_FLOOR))
def test_config_cannot_drop_file_denied_floor(cfg, floor):
    c = copy.deepcopy(cfg)
    c["file_denied_names"].remove(floor)
    with pytest.raises(ep.EnvPrecedenceError) as e:
        ep.validate_config(c)
    assert e.value.code == "F0"

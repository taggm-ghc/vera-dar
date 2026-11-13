"""LLM-ALLOWLIST-1113 (P-1112-02): every LLM call site refuses a base_url whose host is not on the allowlist derived
from the provider entries of config/ask-provider-chain.json. Offline: recorder doubles for the SDK classes, dummy
key values, no network. Real record shapes are copied from the tracked config at test time; only base_url changes.
Hosts are derived from the tracked file, never written as literals; lookalikes use reserved .test names."""
from __future__ import annotations

import ast
import dataclasses
import json
import logging
import os
from pathlib import Path

import pytest
from openai import OpenAIError

from vera import llm_allowlist as la

ROOT = Path(__file__).resolve().parent.parent
CHAIN = ROOT / "config" / "ask-provider-chain.json"
DUMMY = "dummy-not-a-key"


def _chain():
    return json.loads(CHAIN.read_text(encoding="utf-8"))


def _real_host():
    return next(iter(la.allowed_hosts(CHAIN)))


def _real_provider():
    return next(p for p in _chain()["providers"] if p.get("base_url"))


def _real_url():
    return _real_provider()["base_url"]


def _swap(url):
    """a real tracked checker/provider record with only base_url changed"""
    return {**_real_provider(), "base_url": url}


BAD = "https://llm-lookalike.test/v1"


class Recorder:
    """stands in for the SDK client class: records constructions, makes no I/O"""
    made: list

    def __init__(self, **kw):
        type(self).made.append(kw)


def _recorder():
    return type("Rec", (Recorder,), {"made": []})


# ---------------------------------------------------------------- module: allowlist shape and lookalikes
def test_shipped_providers_pass_and_checkers_are_a_regression_check():
    env = _real_provider()["api_key_env"]
    assert la.require_allowed(_real_url(), env) == _real_url()
    checkers = _chain()["grounding"]["claim_check"]["checkers"]
    withurl = [c for c in checkers if c.get("base_url")]
    assert withurl
    for c in withurl:
        assert la.require_allowed(c["base_url"], c["api_key_env"]) == c["base_url"]


def test_default_path_is_the_tracked_chain_file():
    assert la.CONFIG_PATH.resolve() == CHAIN.resolve()


def _lookalikes():
    h = _real_host()
    return [
        ("https://" + h + ".evil.test/v1", la.HOST_NOT_ALLOWED),
        ("https://evil-" + h + "/v1", la.HOST_NOT_ALLOWED),
        ("https://x." + h + "/v1", la.HOST_NOT_ALLOWED),
        ("https://127.0.0.1/v1", la.HOST_NOT_ALLOWED),
        ("https://" + h + "@evil.test/", la.URL_INVALID),
        ("https://@" + h + "/v1", la.URL_INVALID),
        ("https://user:pw@" + h + "/v1", la.URL_INVALID),
        ("https://evil.test#@" + h, la.HOST_NOT_ALLOWED),
        ("https://" + h + "\\@evil.test/", la.URL_INVALID),
        ("https://" + h + ":8443/v1", la.URL_INVALID),
        ("https://" + h + ":abc/v1", la.URL_INVALID),
        ("http://" + h + "/v1", la.URL_INVALID),
        ("ftp://" + h + "/v1", la.URL_INVALID),
        ("https://" + h.replace(".", "\t.", 1) + "/v1", la.URL_INVALID),
        (" https://" + h + "/v1", la.URL_INVALID),
        ("https://" + h + "/v1\n", la.URL_INVALID),
        ("https://" + h + "./v1", la.URL_INVALID),
        ("https://" + h[:-1] + "\u0430/v1", la.URL_INVALID),
        ("https://%61" + h[1:] + "/v1", la.URL_INVALID),
        ("https://[::1]/v1", la.URL_INVALID),
        ("", la.URL_INVALID), (None, la.URL_INVALID), (5, la.URL_INVALID), (b"https://x", la.URL_INVALID),
        ("https:///v1", la.URL_INVALID), ("//" + h, la.URL_INVALID), (h, la.URL_INVALID),
    ]


@pytest.mark.parametrize("idx", range(len(_lookalikes())))
def test_lookalikes_are_refused_with_their_own_code(idx):
    url, code = _lookalikes()[idx]
    env = _real_provider()["api_key_env"]
    with pytest.raises(la.LlmHostRefused) as ei:
        la.require_allowed(url, env)
    assert ei.value.code == code
    assert isinstance(ei.value, OpenAIError)


def test_case_and_explicit_443_are_accepted_after_normalisation():
    h, env = _real_host(), _real_provider()["api_key_env"]
    assert la.require_allowed("HTTPS://" + h.upper() + ":443/v1", env)


def test_key_env_must_be_bound_to_the_host():
    url = _real_url()
    other = "OPENAI_API_KEY" if _real_provider()["api_key_env"] != "OPENAI_API_KEY" else "OTHER_API_KEY"
    for bad in (other, "GROQ_KEY", "database_url", "DATABASE_URL", "", None, 7):
        with pytest.raises(la.LlmHostRefused) as ei:
            la.require_allowed(url, bad)
        assert ei.value.code == la.KEY_NOT_BOUND


def _chain_file(tmp_path, providers, name="chain.json"):
    p = tmp_path / name
    p.write_text(json.dumps({"providers": providers}), encoding="utf-8")
    return p


def test_allowlist_unavailable_fails_closed_with_its_own_code(tmp_path):
    env = _real_provider()["api_key_env"]
    bad_files = [tmp_path / "absent.json", _chain_file(tmp_path, [], "empty.json"),
                 _chain_file(tmp_path, [{"base_url": "http://plain.test/v1", "api_key_env": "A_API_KEY"}], "http.json"),
                 _chain_file(tmp_path, [{"base_url": None, "api_key_env": "A_API_KEY"}], "none.json")]
    mal = tmp_path / "mal.json"
    mal.write_text("{not json", encoding="utf-8")
    bad_files.append(mal)
    for p in bad_files:
        with pytest.raises(la.LlmHostRefused) as ei:
            la.require_allowed(_real_url(), env, path=p)
        assert ei.value.code == la.ALLOWLIST_UNAVAILABLE


def test_custom_file_allows_only_its_own_hosts(tmp_path):
    p = _chain_file(tmp_path, [{"base_url": "https://llm.example.test/v1", "api_key_env": "EX_API_KEY"}])
    assert la.require_allowed("https://llm.example.test/v1", "EX_API_KEY", path=p)
    with pytest.raises(la.LlmHostRefused) as ei:
        la.require_allowed(_real_url(), _real_provider()["api_key_env"], path=p)
    assert ei.value.code == la.HOST_NOT_ALLOWED


def test_cache_follows_the_file_and_never_caches_empty(tmp_path):
    p = _chain_file(tmp_path, [{"base_url": "https://a.example.test/v1", "api_key_env": "A_API_KEY"}])
    assert la.require_allowed("https://a.example.test/v1", "A_API_KEY", path=p)
    st = p.stat()
    p.write_text(json.dumps({"providers": [{"base_url": "https://b.example.test/v1", "api_key_env": "B_API_KEY"}]}),
                 encoding="utf-8")
    os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000_000))  # force a new identity: not a vacuous pass
    assert la.require_allowed("https://b.example.test/v1", "B_API_KEY", path=p)
    with pytest.raises(la.LlmHostRefused):
        la.require_allowed("https://a.example.test/v1", "A_API_KEY", path=p)
    p.write_text("{", encoding="utf-8")
    os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns + 9_000_000_000))
    with pytest.raises(la.LlmHostRefused) as ei:
        la.require_allowed("https://b.example.test/v1", "B_API_KEY", path=p)
    assert ei.value.code == la.ALLOWLIST_UNAVAILABLE


def test_refusal_text_never_carries_the_url_or_host(caplog):
    h = _real_host()
    secret_url = "https://" + h + ":bad-port-marker/v1"
    with caplog.at_level(logging.WARNING, logger="vera"):
        with pytest.raises(la.LlmHostRefused) as ei:
            la.require_allowed(secret_url, "X_API_KEY")
    e = ei.value
    for blob in (str(e), repr(e), repr(e.args), caplog.text):
        assert h not in blob and "bad-port-marker" not in blob
    assert e.__cause__ is None and e.__suppress_context__ is True
    assert la.URL_INVALID in caplog.text


def test_four_distinct_codes_and_texts():
    codes = {la.HOST_NOT_ALLOWED, la.URL_INVALID, la.ALLOWLIST_UNAVAILABLE, la.KEY_NOT_BOUND}
    assert len(codes) == 4 and all(c.startswith("R-LLM-") for c in codes)
    assert len({la._TEXT[c] for c in codes}) == 4


# ---------------------------------------------------------------- sites: red against an unguarded site
def _refused_and_nothing_built(rec, exc):
    assert exc.value.code in (la.HOST_NOT_ALLOWED,)
    assert rec.made == []


def test_site_inference_chain_client(monkeypatch):
    from vera import inference_chain as ic
    monkeypatch.setenv(_real_provider()["api_key_env"], DUMMY)
    rec = _recorder()
    monkeypatch.setattr(ic, "OpenAI", rec)
    entries = ic.load_chain("m", None, path=CHAIN, env={_real_provider()["api_key_env"]: DUMMY}, openai_enabled=False)
    assert entries
    ok = entries[0].client()
    assert len(rec.made) == 1 and rec.made[0]["base_url"] == _real_url()
    rec.made.clear()
    with pytest.raises(la.LlmHostRefused) as ei:
        dataclasses.replace(entries[0], base_url=BAD).client()
    _refused_and_nothing_built(rec, ei)
    assert ok is not None


def test_site_chain_entry_key_not_bound(monkeypatch):
    from vera import inference_chain as ic
    monkeypatch.setenv("OPENAI_API_KEY", DUMMY)
    rec = _recorder()
    monkeypatch.setattr(ic, "OpenAI", rec)
    entries = ic.load_chain("m", None, path=CHAIN, env={_real_provider()["api_key_env"]: DUMMY}, openai_enabled=False)
    with pytest.raises(la.LlmHostRefused) as ei:
        dataclasses.replace(entries[0], api_key_env="OPENAI_API_KEY").client()
    assert ei.value.code == la.KEY_NOT_BOUND and rec.made == []


def test_site_chain_refusal_moves_to_next_entry_not_crash(monkeypatch):
    from vera import inference_chain as ic
    seen = []

    def fake(question, model, pricing, client=None, extra_body=None, **kw):
        seen.append(model)
        raise AssertionError("no answer call may happen for a refused entry")

    monkeypatch.setattr(ic, "answer_in_scope", fake)
    monkeypatch.setattr(ic, "OpenAI", _recorder())
    entries = ic.load_chain("m", None, path=CHAIN, env={_real_provider()["api_key_env"]: DUMMY}, openai_enabled=False)
    bad = [dataclasses.replace(e, base_url=BAD) for e in entries]
    with pytest.raises(OpenAIError) as ei:
        ic._chain_run("q", bad)
    assert isinstance(ei.value, la.LlmHostRefused) and seen == []


def test_site_claim_check_client(monkeypatch):
    from vera import claim_check as cc
    chk = next(c for c in _chain()["grounding"]["claim_check"]["checkers"] if c.get("base_url"))
    monkeypatch.setenv(chk["api_key_env"], DUMMY)
    rec = _recorder()
    monkeypatch.setattr(cc, "OpenAI", rec)
    cc._client(chk, 5.0)
    assert len(rec.made) == 1 and rec.made[0]["base_url"] == chk["base_url"]
    rec.made.clear()
    with pytest.raises(la.LlmHostRefused) as ei:
        cc._client({**chk, "base_url": BAD}, 5.0)
    _refused_and_nothing_built(rec, ei)


def test_site_claim_check_judge_skips_refused_checker_and_builds_nothing(monkeypatch):
    from vera import claim_check as cc
    chk = next(c for c in _chain()["grounding"]["claim_check"]["checkers"] if c.get("base_url"))
    monkeypatch.setenv(chk["api_key_env"], DUMMY)
    rec = _recorder()
    monkeypatch.setattr(cc, "OpenAI", rec)
    cfg = {"checkers": [{**chk, "base_url": BAD}], "timeout_s": 5}
    out = cc.judge([{"id": 0, "text": "c", "cites": [1]}], [{"n": 1, "title": "t", "snippet": "s"}], cfg)
    assert out == (None, None) and rec.made == []


def _stance_cfg():
    return {"stance_timeout_s": 5}


def test_site_claim_stance(monkeypatch):
    import openai
    from vera import claim_stance as cs
    chk = next(c for c in _chain()["grounding"]["claim_check"]["checkers"] if c.get("base_url"))
    monkeypatch.setenv(chk["api_key_env"], DUMMY)
    rec = _recorder()
    monkeypatch.setattr(openai, "OpenAI", rec)
    call = cs._real_call(_stance_cfg())
    with pytest.raises(la.LlmHostRefused) as ei:
        call({**chk, "base_url": BAD}, [])
    _refused_and_nothing_built(rec, ei)
    with pytest.raises(Exception):  # the recorder has no chat API: proves the allowed path constructed the client
        call(chk, [])
    assert len(rec.made) == 1 and rec.made[0]["base_url"] == chk["base_url"]


def test_site_draft_hypothesis(monkeypatch):
    import openai
    from vera import draft_hypothesis as dh
    chk = next(c for c in _chain()["grounding"]["claim_check"]["checkers"] if c.get("base_url"))
    monkeypatch.setenv(chk["api_key_env"], DUMMY)
    rec = _recorder()
    monkeypatch.setattr(openai, "OpenAI", rec)
    cfg = {"draft_timeout_s": 5}
    monkeypatch.setattr(dh, "_groq_checker", lambda: {**chk, "base_url": BAD})
    with pytest.raises(la.LlmHostRefused) as ei:
        dh._real_call(cfg)
    _refused_and_nothing_built(rec, ei)
    monkeypatch.setattr(dh, "_groq_checker", lambda: chk)
    assert dh._real_call(cfg) is not None
    assert len(rec.made) == 1 and rec.made[0]["base_url"] == chk["base_url"]
    monkeypatch.setattr(dh, "_groq_checker", lambda: {**chk, "base_url": BAD})
    assert dh.make("q", {"draft_timeout_s": 5, "draft_model": "m", "draft_max_chars": 10, "max_claims": 1}) is None


def test_site_generator_client_settings_url_and_client_kwargs(monkeypatch):
    from vera import generator_client as gc
    env = {_real_provider()["api_key_env"]: DUMMY}
    rec = _recorder()
    s_ok = gc.GeneratorSettings("m", _real_url(), _real_provider()["api_key_env"])
    gc.make_client(s_ok, env=env, client_cls=rec, timeout=3)
    assert len(rec.made) == 1 and rec.made[0]["base_url"] == _real_url()
    rec.made.clear()
    s_bad = gc.GeneratorSettings("m", BAD, _real_provider()["api_key_env"])
    with pytest.raises(la.LlmHostRefused):
        gc.make_client(s_bad, env=env, client_cls=rec)
    assert rec.made == []
    # the effective kwargs count, not only the settings: a caller-supplied base_url is checked too
    s_def = gc.GeneratorSettings("m", None, _real_provider()["api_key_env"])
    with pytest.raises(la.LlmHostRefused):
        gc.make_client(s_def, env=env, client_cls=rec, base_url=BAD)
    s_default = gc.GeneratorSettings("m")
    with pytest.raises(la.LlmHostRefused):
        gc.make_client(s_default, env={}, client_cls=rec, base_url=BAD)
    assert rec.made == []


def test_corpus_admission_does_not_fall_back_to_default_client(monkeypatch):
    from vera import corpus_admission as ca
    from vera import inference_chain as ic
    monkeypatch.setattr(ic, "OpenAI", _recorder())
    entries = ic.load_chain("m", None, path=CHAIN, env={_real_provider()["api_key_env"]: DUMMY}, openai_enabled=False)
    bad = dataclasses.replace(entries[0], base_url=BAD)
    import vera.ask_service as asv
    monkeypatch.setattr(asv, "_get_client", lambda: (_ for _ in ()).throw(AssertionError("default client used")))
    with pytest.raises(la.LlmHostRefused):
        ca.llm_call_for(bad)([{"role": "user", "content": "x"}])


# ---------------------------------------------------------------- recurrence: AST scan with a positive control
_CTORS = {"OpenAI", "AsyncOpenAI", "client_cls"}


def _unguarded(source: str, label: str) -> list:
    tree = ast.parse(source)
    out = []

    def calls_in(fn):
        return [n for n in ast.walk(fn) if isinstance(n, ast.Call)]

    def name_of(c):
        f = c.func
        return f.id if isinstance(f, ast.Name) else (f.attr if isinstance(f, ast.Attribute) else "")

    funcs = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    for fn in funcs:
        guarded = any(name_of(c) == "require_allowed" for c in calls_in(fn))
        for c in calls_in(fn):
            nm = name_of(c)
            kws = {k.arg for k in c.keywords}
            addressed = "base_url" in kws or None in kws
            if (nm in _CTORS or (nm in ("with_options", "copy") and "base_url" in kws)) and addressed and not guarded:
                out.append((label, fn.name, nm, c.lineno))
    return out


def test_ast_scan_positive_control_flags_an_unguarded_call():
    src = "def f(u):\n    return OpenAI(base_url=u, api_key='x')\n"
    assert _unguarded(src, "ctl") == [("ctl", "f", "OpenAI", 2)]
    assert _unguarded("def g(kw):\n    return client_cls(**kw)\n", "ctl") != []
    assert _unguarded("def h(c, u):\n    return c.with_options(base_url=u)\n", "ctl") != []
    ok = "def f(u):\n    u = require_allowed(u, 'A_API_KEY')\n    return OpenAI(base_url=u)\n"
    assert _unguarded(ok, "ctl") == []
    assert _unguarded("def d():\n    return OpenAI(max_retries=0)\n", "ctl") == []


def test_every_base_url_client_construction_in_the_tree_is_guarded():
    hits = []
    scanned = 0
    for base in ("vera", "scripts"):
        for p in sorted((ROOT / base).rglob("*.py")):
            scanned += 1
            hits += _unguarded(p.read_text(encoding="utf-8"), str(p.relative_to(ROOT)))
    assert scanned > 50
    # pinned exemptions: the default OpenAI clients pass no base_url (the ** is only the optional timeout dict)
    exempt = {("vera/ask_service.py", "_build_client"), ("vera/ask_service.py", "complete_after_exhaustion")}
    seen_exempt = {(h[0], h[1]) for h in hits} & exempt
    assert seen_exempt == exempt  # the exemptions are real, not stale
    hits = [h for h in hits if (h[0], h[1]) not in exempt]
    assert hits == [], hits
    # the guarded sites must really be present (non-vacuous): each named file calls require_allowed
    for rel in ("vera/inference_chain.py", "vera/claim_check.py", "vera/claim_stance.py",
                "vera/draft_hypothesis.py", "vera/generator_client.py"):
        assert "require_allowed(" in (ROOT / rel).read_text(encoding="utf-8"), rel

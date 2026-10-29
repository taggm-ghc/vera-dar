"""ACQ-1027 K3/K4/K5 (offline): URL scrubbing, redaction, key-like config values, sentinel key absent from logs."""
import json
import logging
from pathlib import Path

import pytest

from vera import acq_urls as U
from vera import search_providers as sp

ROOT = Path(__file__).resolve().parent.parent
SENT = "SENTINELkey0123456789abcdefXYZ"
SEARCH = json.loads((ROOT / "config" / "vera_eval_run.json").read_text())["search"]


def test_scrub_drops_key_params_and_loaded_key_values():
    u = f"https://api.example.org/w?search=a&api_key={SENT}&apikey=x&key=y&token=t&access_token=z&mailto=a@b.c&other={SENT}&n=1"
    out = U.scrub(u, [SENT])
    assert out == "https://api.example.org/w?search=a&n=1"
    assert SENT not in out


def test_scrub_keeps_clean_url_and_is_pure_str():
    u = "https://a.org/p?x=1#f"
    assert U.scrub(u) == u and isinstance(U.scrub(u), str)
    assert U.scrub("not a url") == "not a url"


def test_has_key_like_param():
    assert U.has_key_like_param("https://a.org/?API_KEY=1")
    assert not U.has_key_like_param("https://a.org/?q=1&monkey=2")


def test_redact_text():
    s = f"boom GET https://a.org/w?q=1&api_key={SENT} failed; {SENT}; token=abc123 done"
    out = U.redact_text(s, [SENT])
    assert SENT not in out and "abc123" not in out and "q=1" in out


@pytest.mark.parametrize("v", [
    "sk-abcdefghijklmnop1234", "ghp_abcdefghijklmnopqrstuv", "Bearer abcdefghijklmnop12345",
    "AbCdEf0123456789AbCdEf0123456789", SENT])
def test_key_like_positive(v):
    assert U.key_like_value(v)


@pytest.mark.parametrize("v", [
    "https://api.openalex.org/works", "VERA_DB_URL_WO", "VERA_AUTOACQ_DISABLED", "vera-dar-auto-acquire-v1",
    "a" * 64 if False else "0123456789abcdef" * 4, "VERA-DAR automatic acquisition (open-access full text; visitor question not stored)",
    "short", "", 5, True, None])
def test_key_like_negative(v):
    assert not U.key_like_value(v)


def test_shipped_auto_acquire_json_has_no_key_like_value():
    bad = []

    def walk(x, path=""):
        if isinstance(x, dict):
            for k, v in x.items():
                walk(v, f"{path}.{k}")
        elif isinstance(x, list):
            for i, v in enumerate(x):
                walk(v, f"{path}[{i}]")
        elif U.key_like_value(x):
            bad.append(path)
    walk(json.loads((ROOT / "config" / "auto_acquire.json").read_text()))
    assert bad == []


class _Err(Exception):
    pass


def _provider(tx):
    return sp.OpenAlexProvider(SEARCH["openalex"], transport=tx, query_terms=SEARCH["query_terms"])


def test_keyed_openalex_error_and_logs_never_carry_key(monkeypatch, caplog):
    monkeypatch.setenv("OPENALEX_API_KEY", SENT)
    seen = {}

    def tx(url, params, headers, timeout):
        seen["key"] = params.get("api_key")
        logging.getLogger("urllib3.connectionpool").debug("GET %s?api_key=%s", url, params["api_key"])  # must be muted
        raise _Err(f"HTTPError for url: {url}?search=x&api_key={params['api_key']}")
    caplog.set_level(logging.DEBUG)
    with pytest.raises(Exception) as ei:
        _provider(tx).call("boxing mma", 5, 5.0)
    assert seen["key"] == SENT  # still sent as api_key param
    assert SENT not in str(ei.value)
    assert SENT not in caplog.text
    assert logging.getLogger("urllib3.connectionpool").level == logging.NOTSET  # restored


def test_keyed_openalex_response_url_scrubbed(monkeypatch):
    monkeypatch.setenv("OPENALEX_API_KEY", SENT)

    class R:
        status_code, text, headers = 200, "{}", {}
        url = f"https://api.openalex.org/works?api_key={SENT}&x=1"
    r = _provider(lambda *a: R()).call("boxing mma", 5, 5.0)
    assert SENT not in r.url


def test_keyless_unchanged(monkeypatch):
    monkeypatch.delenv("OPENALEX_API_KEY", raising=False)
    calls = []
    _provider(lambda u, p, h, t: calls.append(p) or object()).call("boxing mma", 5, 5.0)
    assert "api_key" not in calls[0]


def test_load_config_refuses_a_key_like_value_and_names_only_the_path(tmp_path):
    import json
    from vera import acq_config
    good = {"enabled": True, "keys": {"openalex_api_key_env": "OPENALEX_API_KEY"}}
    bad = {"enabled": True, "keys": {"openalex_api_key": "sk-" + "a1B2c3D4e5F6g7H8i9J0k1L2"}}
    pg, pb = tmp_path / "g.json", tmp_path / "b.json"
    pg.write_text(json.dumps(good))
    pb.write_text(json.dumps(bad))
    assert acq_config.load_config(pg) == good
    assert acq_config.refuse_key_like_values(bad) == ["keys.openalex_api_key"]


def test_load_config_fails_loud_with_paths_only_and_absent_or_unreadable_stays_none(tmp_path, caplog):
    import json
    from vera import acq_config
    secret = "sk-" + "a1B2c3D4e5F6g7H8i9J0k1L2"
    pb = tmp_path / "b.json"
    pb.write_text(json.dumps({"enabled": True, "keys": {"openalex_api_key": secret}, "x": {"y": secret}}))
    with caplog.at_level("ERROR", logger="vera"), pytest.raises(acq_config.ConfigError) as ei:
        acq_config.load_config(pb)
    assert "keys.openalex_api_key" in str(ei.value) and "x.y" in str(ei.value) and secret not in str(ei.value)
    errs = [r for r in caplog.records if r.levelname == "ERROR"]
    assert errs and "keys.openalex_api_key" in errs[0].getMessage() and secret not in caplog.text
    assert acq_config.load_config(tmp_path / "missing.json") is None
    (tmp_path / "bad.json").write_text("{not json")
    assert acq_config.load_config(tmp_path / "bad.json") is None
    (tmp_path / "list.json").write_text("[1]")
    assert acq_config.load_config(tmp_path / "list.json") is None


def test_request_path_callers_degrade_to_off_on_config_error(monkeypatch):
    from vera import acq_config, catstep, disclosure

    def boom(*a, **k):
        raise acq_config.ConfigError("acquisition config refused: key-like value at path(s) x")
    monkeypatch.setattr(acq_config, "load_config", boom)
    assert disclosure.auto_acquire_enabled() is False and disclosure._layer2_answering_on() is True
    assert catstep._micro_cfg() == {}
    from vera import ask_service, auto_acquire, search_providers
    assert ask_service._acq_cfg_if_on() is None
    assert auto_acquire.enqueue("q", "decline") != "ACQ-QUEUED"
    assert search_providers.openalex_key_env_name({}) == "OPENALEX_API_KEY"  # falls back to the default name


_MAP_PATH = "micro_topic.registry_table.refused_state_map."


def test_refused_state_map_exemption_is_narrow_positive_and_negative():
    from vera import acq_urls, topic_registry
    assert acq_urls._REFUSED_STATE_NAMES == topic_registry.REFUSED_STATES
    # positive: exactly this path, and ONLY the values banned, prohibited, secrets (HARDEN-FIX2 E6)
    assert acq_urls.key_like_value("secrets", _MAP_PATH + "secret_leak") is False
    assert acq_urls.key_like_value("banned", _MAP_PATH + "harmful") is False and acq_urls.key_like_value("prohibited", _MAP_PATH + "advice") is False
    assert acq_urls.key_like_value("secrets", "micro_topic.registry_table.other.secret_leak") is True  # outside the exemption: still refused
    assert acq_urls.key_like_value("secrets", "registry_table.refused_state_map.secret_leak") is True
    assert acq_urls.key_like_value("secrets", _MAP_PATH + "secret_leak.deeper") is True
    for v in ("harmful", "advice", "rule_change", "secret_leak", "abc", "banned1", "Banned", "banned "):
        assert acq_urls._refused_map_value_ok(v, _MAP_PATH + "harmful") is False, v  # no category-code-shaped exemption any more
    for v in ("banned", "prohibited", "secrets"):
        assert acq_urls._refused_map_value_ok(v, _MAP_PATH + "harmful") is True
        assert acq_urls._refused_map_value_ok(v, "micro_topic.registry_table.other.harmful") is False  # only under that path
    # negative: a credential-like value under the exempt path is still refused
    for v in ("sk-" + "a1B2c3D4e5F6g7H8i9J0k1L2", "ghp_" + "a1B2c3D4e5F6g7H8i9J0k1L2m3N4", "A1b2C3d4E5f6G7h8I9j0K1l2M3n4", "banned1", "Banned"):
        assert acq_urls.key_like_value(v, _MAP_PATH + "harmful") is True, v
    from vera import acq_config
    ok = {"micro_topic": {"registry_table": {"refused_state_map": {"secret_leak": "secrets", "harmful": "banned"}}}}
    assert acq_config.refuse_key_like_values(ok) == []
    bad = {"micro_topic": {"registry_table": {"refused_state_map": {"harmful": "sk-" + "a1B2c3D4e5F6g7H8i9J0k1L2"}}}}
    assert acq_config.refuse_key_like_values(bad) == [_MAP_PATH + "harmful"]


# INC-1028-1: URL pass must mark (not drop) key-like params, with no literal key passed to redact_text.
@pytest.mark.parametrize("url,name", [
    ("https://a.org/w?q=1&api%5Fkey=SECRETVAL1", "api%5Fkey"),
    ("https://a.org/w?q=1&api%5fkey=SECRETVAL1", "api%5fkey"),
    ("https://a.org/w?q=1&access-token=SECRETVAL1", "access-token"),
    ("https://a.org/w?q=1&API_KEY=SECRETVAL1", "API_KEY"),
    ("https://a.org/w?q=1&Access-Token=SECRETVAL1", "Access-Token"),
    ("https://a.org/w?q=1&ApiKey=SECRETVAL1", "ApiKey"),
    ("https://a.org/w?q=1&MAILTO=SECRETVAL1", "MAILTO"),
    ("https://a.org/w?q=1#access_token=SECRETVAL1", "access_token"),
    ("https://a.org/w?q=1#state=x&access%2Dtoken=SECRETVAL1", "access%2Dtoken"),
])
def test_redact_text_url_key_param_forms_no_key_passed(url, name):
    out = U.redact_text(f"boom GET {url} failed")
    assert "SECRETVAL1" not in out
    assert f"{name}={U.REDACTED}" in out
    assert "q=1" in out


def test_redact_text_url_multiple_params_and_trailing_punct():
    out = U.redact_text("see https://a.org/w?token=AAA111&n=2&api%5Fkey=BBB222&mailto=c@d.e&key=CCC333.")
    assert not any(v in out for v in ("AAA111", "BBB222", "CCC333", "c@d.e"))
    assert "n=2" in out and out.count(U.REDACTED) == 4


def test_redact_text_url_userinfo_redacted():
    out = U.redact_text("GET https://user:PASSW0RDX@host.org/p?x=1 failed")
    assert "PASSW0RDX" not in out and "user:" not in out and "host.org/p?x=1" in out


def test_redact_text_url_non_key_params_untouched():
    s = "GET https://a.org/w?q=1&monkey=2&keyword=3#frag=4"
    assert U.redact_text(s) == s


def test_redact_text_url_encoded_name_with_loaded_key_too():
    out = U.redact_text(f"https://a.org/w?api%5Fkey={SENT}&other={SENT}", [SENT])
    assert SENT not in out and f"api%5Fkey={U.REDACTED}" in out


def test_key_like_reason_codes_exempt_but_other_hyphenated_upper_flagged():
    assert not U.key_like_value("R-ACQ-PREPRINT_LICENCE_UNVERIFIED")
    assert U.key_like_value("ABCDEFGHIJ-KLMNOPQRST-UVWXYZ0123")  # broad uppercase-hyphen exemption deliberately not granted


# FIX2-1028 D3: broadened param-name matcher (word-based), D5 path rules, D2 thread-safe quiet_urllib3, D4 env name.
@pytest.mark.parametrize("name", ["x-api-key", "X-API-KEY", "apikey", "ApiKey", "client_secret", "refresh%5Ftoken",
                                   "authToken", "signature", "X-Amz-Signature", "password", "auth", "sessionToken"])
def test_key_param_names_broadened(name):
    assert U._is_key_param(name)
    assert f"{name}={U.REDACTED}" in U.redact_text(f"GET https://a.org/w?q=1&{name}=SECRETVAL9 failed")
    assert "SECRETVAL9" not in U.redact_text(f"error x-api-key=SECRETVAL9 {name}: SECRETVAL9")


@pytest.mark.parametrize("name", ["keyword", "keywords", "monkey", "author", "authors", "q", "tokenize", "sort", "turkey"])
def test_ordinary_param_names_untouched(name):
    assert not U._is_key_param(name)
    s = f"GET https://a.org/w?{name}=abc123&n=2"
    assert U.redact_text(s) == s


def test_key_like_value_path_rules():
    sk = "AbCdEf0123456789AbCdEf0123456789"
    assert U.key_like_value(sk, "keys.openalex_api_key") and U.key_like_value("openalex-key", "keys.k")
    assert not U.key_like_value("OPENALEX_API_KEY", "keys.openalex_api_key_env")
    assert U.key_like_value("a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2", "keys.foo_env")  # *_env must be an env NAME
    assert U.key_like_value("openalex_key_value", "x.api_token")  # under a key path only env names pass
    assert not U.key_like_value("OPENALEX_API_KEY", "x.api_token")


def test_key_like_value_real_token_shapes_flagged_with_path():
    for v in ("ABCDEFGHIJKLMNOPQRSTUVWXYZ", "0123456789abcdef0123456789abcdef01234567", "0123456789abcdef" * 4,
              "123e4567-e89b-12d3-a456-426614174000", "a1b2-c3d4-e5f6-a7b8-c9d0-e1f2"):
        assert U.key_like_value(v, "x.y"), v


def test_key_like_value_sha_pin_allowlist_and_reason_codes_and_markers():
    h = "0123456789abcdef" * 4
    assert not U.key_like_value(h, "pins.manifest_sha256")
    assert not U.key_like_value(h, "pins.documents.wilson2017.text_sha256")
    assert U.key_like_value(h, "pins.documents.wilson2017.other")
    assert not U.key_like_value("R-ACQ-PREPRINT_LICENCE_UNVERIFIED", "reason_codes.preprint_licence_unverified")
    assert not U.key_like_value("vera-dar-auto-acquire-v1", "marker")
    assert not U.key_like_value("prose with spaces is never long and this is long enough to matter", "keys._note")


def test_shipped_config_loads_and_scan_uses_paths():
    from vera import acq_config
    cfg = json.loads((ROOT / "config" / "auto_acquire.json").read_text())
    assert acq_config.refuse_key_like_values(cfg) == []
    bad = {"keys": {"openalex_api_key_env": "0123456789abcdef" * 4}, "other": {"token": "plainlowercasevalue"},
           "x": {"y": "0123456789abcdef" * 4}}
    assert acq_config.refuse_key_like_values(bad) == ["keys.openalex_api_key_env", "other.token", "x.y"]


def test_quiet_urllib3_threadsafe_no_key_reaches_caplog(caplog):
    import logging
    import threading
    lg = logging.getLogger("urllib3.connectionpool")
    before = lg.level
    keys = ["THREADKEY-AAAA1111", "THREADKEY-BBBB2222"]
    barrier = threading.Barrier(2)

    def work(k, other):
        with U.quiet_urllib3([k]):
            barrier.wait()
            for _ in range(50):
                lg.debug('GET /w?q=1&api_key=%s HTTP/1.1', k)
                lg.warning('GET https://a.org/w?x-api-key=%s&z=1', k)
            barrier.wait()
    with caplog.at_level(logging.DEBUG, logger="urllib3"):
        ts = [threading.Thread(target=work, args=(keys[0], keys[1])), threading.Thread(target=work, args=(keys[1], keys[0]))]
        [t.start() for t in ts]
        [t.join() for t in ts]
    assert lg.level == before
    assert caplog.records and not any(k in caplog.text for k in keys)
    assert "q=1" in caplog.text and not U._ACTIVE


def test_openalex_key_env_name_single_source(monkeypatch):
    from vera import search_providers as sp
    monkeypatch.setenv("OA_FROM_CFG", "cfgkey12345678")
    monkeypatch.setenv("OA_FROM_SETTINGS", "setkey12345678")
    from vera import acq_config
    monkeypatch.setattr(acq_config, "load_config", lambda *a, **k: {"keys": {"openalex_api_key_env": "OA_FROM_CFG"}})
    assert sp.openalex_key_env_name({"api_key_env": "OA_FROM_SETTINGS"}) == "OA_FROM_CFG"
    assert sp.openalex_api_key({"api_key_env": "OA_FROM_SETTINGS"}) == "cfgkey12345678"
    monkeypatch.setattr(acq_config, "load_config", lambda *a, **k: None)
    assert sp.openalex_key_env_name({"api_key_env": "OA_FROM_SETTINGS"}) == "OA_FROM_SETTINGS"
    assert sp.openalex_key_env_name({}) == "OPENALEX_API_KEY"


@pytest.mark.parametrize("s", ["x=1;token=SECRET", "a=1?token=SECRET", "x=1#token=SECRET"])
def test_redact_text_free_text_separators_do_not_leak(s):
    out = U.redact_text(s)
    assert "SECRET" not in out and U.REDACTED in out


@pytest.mark.parametrize("s", ["keyword=abc", "monkey=abc", "author=abc", "turkey=abc"])
def test_redact_text_free_text_separator_negatives(s):
    assert U.redact_text(s) == s


def test_shipped_auto_acquire_json_passes_path_aware_refusal():
    from vera import acq_config
    assert acq_config.refuse_key_like_values(json.loads((ROOT / "config" / "auto_acquire.json").read_text())) == []


def test_main_loads_the_acquisition_config_at_startup_so_a_key_like_value_stops_the_service():
    from pathlib import Path
    src = (Path(__file__).resolve().parent.parent / "main.py").read_text(encoding="utf-8")
    assert "_acq_config.load_config()" in src and src.index("startup_checks()\n") < src.index("_acq_config.load_config()")


def test_main_degrades_when_acq_config_refused(monkeypatch, caplog):
    """HARDEN-FIX2: a refused acquisition config must not crash the service: ERROR log, acquisition off, /health reports it, no values."""
    import importlib
    import main
    from vera import acq_config

    def boom(*a, **k):
        raise acq_config.ConfigError("acquisition config refused: key-like value at path(s) x.y")
    monkeypatch.setattr(acq_config, "load_config", boom)
    try:
        with caplog.at_level("ERROR", logger="vera"):
            m = importlib.reload(main)  # must not raise
        assert m.ACQ_CONFIG_REFUSED is True
        assert any("acquisition config refused at startup" in r.getMessage() for r in caplog.records)
        h = m.health()
        assert h["status"] == "ok" and h["acquisition"] == "off_config_refused"
    finally:
        monkeypatch.undo()
        m = importlib.reload(main)
    assert m.ACQ_CONFIG_REFUSED is False and "acquisition" not in m.health()

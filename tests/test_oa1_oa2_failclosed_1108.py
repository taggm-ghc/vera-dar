"""OA-1 / OA-2 (OA12-1108): acquisition rate limiting fails closed for unlisted hosts; eutils left the fetch allowlist."""
import json
from pathlib import Path

import pytest

from vera import acq_config, rate_limit as rl

ROOT = Path(__file__).resolve().parent.parent
ACQ_HOSTS = ["www.ebi.ac.uk", "arxiv.org", "doi.org", "dx.doi.org"]  # every host the acquisition path calls


class _Fn:
    def __init__(self):
        self.urls = []

    def __call__(self, url, **kw):
        self.urls.append(url)
        return object()


def test_strict_refuses_unlisted_host_and_sends_nothing():
    f = _Fn()
    g = rl.limited_by_url(f, registry=rl.get_registry(), strict=True)
    with pytest.raises(rl.UnlistedHost) as e:
        g("https://not-a-listed-host.example/x")
    assert f.urls == [] and e.value.host == "not-a-listed-host.example"
    assert not isinstance(e.value, rl.RateLimited)


def test_nonstrict_default_unchanged_passes_unlisted():
    f = _Fn()
    rl.limited_by_url(f, registry=rl.get_registry())("https://not-a-listed-host.example/x")
    assert f.urls == ["https://not-a-listed-host.example/x"]


@pytest.mark.parametrize("host", ACQ_HOSTS)
def test_every_acquisition_host_has_a_portal(host):
    f = _Fn()
    g = rl.limited_by_url(f, registry=rl.get_registry(), strict=True)
    try:
        g(f"https://{host}/x")
    except rl.RateLimited:
        pass  # a pacing skip is fine; an UnlistedHost is not
    assert rl.get_registry().portal_for_url(f"https://{host}/x") is not None


def test_acquisition_deps_wiring_is_strict():
    src = (ROOT / "vera" / "auto_acquire.py").read_text()
    assert "limited_transport(selective_fetch.pinned_get, strict=True)" in src  # API-URL-GATE-1108b (4): strict kept


def test_config_hosts_all_listed_and_eutils_removed():
    cfg = json.loads((ROOT / "config" / "auto_acquire.json").read_text())
    assert cfg["fetch"]["allowed_hosts"] == ["www.ebi.ac.uk"]
    assert "eutils" not in json.dumps(cfg["fetch"])
    reg = rl.get_registry()
    hosts = (cfg["fetch"]["allowed_hosts"] + cfg["arxiv_route"]["hosts"] + cfg["landing_route"]["resolver_hosts"]
             + cfg["landing_route"].get("publisher_hosts", []))
    for h in hosts:
        assert reg.portal_for_url(f"https://{h}/") is not None, h


def test_unlisted_maps_to_host_code_in_auto_acquire_guarded_get():
    from vera import auto_acquire
    cfg = json.loads((ROOT / "config" / "auto_acquire.json").read_text())

    def boom(*a, **k):
        raise rl.UnlistedHost("x")
    assert acq_config.reason(cfg, "host") == "R-ACQ-HOST"
    src = (ROOT / "vera" / "auto_acquire.py").read_text() + (ROOT / "vera" / "acq_discovered.py").read_text()
    assert src.count("except rate_limit.UnlistedHost") == 2


def test_acquisition_path_refuses_unlisted_host_behaviourally(monkeypatch):
    """Fail-before check on behaviour: default_deps (the live wiring) must refuse an unlisted host with nothing sent.
    On the pre-fix code the host is passed to the transport unlimited and uncounted."""
    import socket

    import sqlalchemy

    from tests import acq_support as A
    from tests.test_acq_wiring import _Engine
    from vera import auto_acquire, selective_fetch
    sent = []
    monkeypatch.setattr(selective_fetch, "pinned_get", lambda url, **kw: sent.append(url))
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k: [(2, 1, 6, "", ("93.184.216.34", 443))])
    monkeypatch.setattr(sqlalchemy, "create_engine", lambda url, **kw: _Engine("ro"))
    d = auto_acquire.default_deps(A.cfg(), A.ENV_ON)
    err = None
    try:
        d.http_get("https://unlisted-publisher.example/x.xml", headers={}, timeout=1, allow_redirects=False, stream=True)
    except Exception as e:  # noqa: BLE001
        err = e
    assert sent == [], f"unlisted host was sent: {sent}"
    assert type(err).__name__ == "UnlistedHost"

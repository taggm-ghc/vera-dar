"""CMF-1022-I3 R2: pin hygiene (thread isolation; a refused check leaves no usable pin).

OFFLINE. socket.getaddrinfo is replaced by a scripted resolver (per host, a queue of answers; the last answer repeats)
that counts lookups, and urllib3's socket factory by a recorder that refuses every connect (so any connect attempt is
visible and nothing is sent). No DNS, no network, no DB. Plain http URLs, so no TLS layer is involved.
"""
import socket
import threading
from types import SimpleNamespace

import pytest
import urllib3

from vera import selective_fetch

PUB4 = "93.184.216.34"


@pytest.fixture
def net(monkeypatch):
    st = SimpleNamespace(answers={}, dns_calls=[], connects=[])
    lock = threading.Lock()

    def fake_gai(host, port, *a, **k):
        with lock:
            st.dns_calls.append((threading.get_ident(), host))
            q = st.answers.get(host)
            if not q:
                raise socket.gaierror(f"unknown test host {host!r}")
            ips = q.pop(0) if len(q) > 1 else q[0]
        return [((socket.AF_INET6 if ":" in ip else socket.AF_INET), socket.SOCK_STREAM, 6, "", (ip, port or 0))
                for ip in ips]
    monkeypatch.setattr(socket, "getaddrinfo", fake_gai)

    def fake_create_connection(address, *a, **k):
        with lock:
            st.connects.append((address[0].strip("[]"), address[1]))
        raise ConnectionRefusedError("test: no connect allowed")
    monkeypatch.setattr(urllib3.util.connection, "create_connection", fake_create_connection)
    selective_fetch._PIN.slot = None
    yield st
    selective_fetch._PIN.slot = None


def test_i3_pin_from_thread_a_is_invisible_to_thread_b_which_revalidates(net):
    url = "http://thr.example.org/x"
    net.answers["thr.example.org"] = [[PUB4], ["127.0.0.1"]]   # A sees public; B's own lookup sees a rebind
    assert selective_fetch.is_public_url(url) == (True, "")     # thread A (main): pin stored here
    out = {}

    def thread_b():
        out["slot"] = selective_fetch._PIN.slot
        try:
            selective_fetch.pinned_get(url, timeout=1)
            out["exc"] = None
        except Exception as exc:  # noqa: BLE001 - recorded and asserted below
            out["exc"] = exc
    t = threading.Thread(target=thread_b)
    t.start()
    t.join(5)
    assert not t.is_alive()
    assert out["slot"] is None                                    # A's pin not visible in B
    assert isinstance(out["exc"], selective_fetch.PinnedRefused)  # B re-validated and got the rebound answer
    assert net.connects == []
    assert len(net.dns_calls) == 2                                # one lookup in A, one fresh lookup in B
    assert net.dns_calls[0][0] != net.dns_calls[1][0]
    assert selective_fetch._take_pin(url) is not None             # A's own pin is untouched by B


def test_i3_refused_check_after_success_leaves_no_usable_pin(net):
    """R1 regression. Without the R1 clear at the start of is_public_url, the refused second check returns early
    (blocked non-public address) BEFORE the pin store, so the slot still holds the first check's pin for the SAME URL;
    pinned_get would then consume it without any lookup and attempt a connect to PUB4: connects == [(PUB4, 80)],
    dns_calls has 2 entries, and the exception is a ConnectionError, not PinnedRefused. Every assertion below fails."""
    url = "http://ref.example.org/y"
    net.answers["ref.example.org"] = [[PUB4], ["10.1.2.3"]]
    assert selective_fetch.is_public_url(url) == (True, "")
    ok, why = selective_fetch.is_public_url(url)
    assert ok is False and "non-public" in why
    assert selective_fetch._PIN.slot is None
    with pytest.raises(selective_fetch.PinnedRefused):
        selective_fetch.pinned_get(url, timeout=1)                # re-validates (3rd lookup) and refuses
    assert net.connects == []
    assert len(net.dns_calls) == 3


def test_i3_refused_check_of_another_url_also_clears_the_pin(net):
    net.answers["good.example.org"] = [[PUB4]]
    assert selective_fetch.is_public_url("http://good.example.org/") == (True, "")
    assert selective_fetch.is_public_url("ftp://good.example.org/") == (False, "only http(s) URLs are fetched")
    assert selective_fetch._PIN.slot is None

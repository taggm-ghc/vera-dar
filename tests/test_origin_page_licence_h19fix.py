"""H19-FIX-1109 on the OD-1109-ORIGIN path (six-1109 original-host follows): the real public_fetch extracts the
canonical link and third-party markup; vera.page_licence then applies A7, A8, A10 and A11 before any text is used, and
the page's licence is attached for the citation notice. SYNTHETIC fixtures; DNS and transport are fakes; no socket."""
import socket

import pytest

from vera import lane_evidence as le, public_fetch as pf, web_route as wr

CFG = {"user_agent": "UA-test", "deny_hosts": [], "max_redirects": 2, "max_compressed_bytes": 50000,
       "max_decompressed_bytes": 200000, "max_robots_bytes": 2000, "timeout_s": 5,
       "content_types": ["text/html", "application/xhtml+xml", "text/plain"]}
ORIGIN = "https://origin.example.net/papers/zorb"
PARA = "<p>" + "Zorblats are synthetic test organisms. " * 12 + "</p>"


class Resp:
    def __init__(self, status=200, headers=None, body=b""):
        self.status_code, self.headers, self.body = status, headers or {}, body

    def iter_content(self, n):
        for i in range(0, len(self.body), n):
            yield self.body[i:i + n]

    def close(self):
        pass


class Net:
    def __init__(self, body):
        self.body = body

    def resolve(self, host, port):
        return [(socket.AF_INET, 0, 0, "", ("93.184.216.34", port))]

    def transport(self, host, ip, port, target, headers, timeout):
        if target == "/robots.txt":
            return Resp(200, {"Content-Type": "text/plain"}, b"User-agent: *\nAllow: /\n")
        return Resp(200, {"Content-Type": "text/html; charset=utf-8"}, self.body.encode())

    def fetch(self, url, **kw):
        return pf.fetch(url, CFG, resolve=self.resolve, transport=self.transport, **kw)


def read(head="", body=PARA):
    html = f"<html><head>{head}</head><body>{body}</body></html>"
    c = {"url": ORIGIN, "title": "origin.example.net", "snippet": "", "doi": None, "discovered_url": ORIGIN,
         "portal_origin": True}
    return wr.read_web("What are zorblats?", [c], fetch=Net(html).fetch, registry={}, deny_list=[],
                       cfg=wr.load_config())


def test_plain_origin_page_still_read():
    out = read()
    assert len(out.pages) == 1 and out.pages[0].licence_info is None


def test_licence_link_attached_to_citation_notice():
    out = read('<link rel="license" href="https://creativecommons.org/licenses/by/4.0/">')
    assert len(out.pages) == 1
    att = le.attribution_for(out.pages[0])
    assert att["licence"] == "CC BY 4.0" and att["licence_url"] == "https://creativecommons.org/licenses/by/4.0/"


def test_blockquote_and_figcaption_removed_before_use():
    body = PARA + "<blockquote><p>Quoted words from another outlet.</p></blockquote><figure><figcaption>Photo credit line"\
                  "</figcaption></figure>"
    out = read(body=body)
    assert len(out.pages) == 1
    t = out.pages[0].text
    assert "Quoted words" not in t and "Photo credit" not in t and "Zorblats are synthetic" in t


@pytest.mark.parametrize("head,body", [  # re-verify F1/F2: not blocked on a non-ND original host
    ('<link rel="canonical" href="https://other-site.example.com/zorb">', PARA),
    ("", PARA + "<p>This story was originally published by Another Outlet.</p>"),
    ('<link rel="license" href="https://creativecommons.org/licenses/by/4.0/">', PARA + "<p>All Rights Reserved.</p>"),
])
def test_origin_page_not_blocked_by_nd_only_rules(head, body):
    out = read(head, body)
    assert len(out.pages) == 1


@pytest.mark.parametrize("head,body,token", [
    ('<link rel="license" href="https://creativecommons.org/licenses/by-nd/4.0/">', PARA, "use_licence_reject"),
    ("", "<p>BERLIN (Reuters) - " + "Zorblats are synthetic test organisms. " * 12 + "</p>", "use_acc_third_party_page"),
    ("", "<blockquote><p>" + "Entirely quoted. " * 10 + "</p></blockquote>", "use_acc_third_party_all"),
])
def test_origin_page_level_checks(head, body, token):
    out = read(head, body)
    assert out.pages == []
    assert token in [d["gate"].get("use") for d in out.dev]

"""OFFICIAL-LANE-S4A G12: an official-lane redirect to an off-registry host (or an avoided path) is refused BEFORE it
is requested. Uses the REAL vera.public_fetch.fetch and vera.web_route.gated_fetch with fake DNS and a fake
transport: no socket is opened, no DB, no env. Works from repo/tests or work/OFFICIAL-LANE-S4A/staged/tests."""
import importlib.util
import json
import socket
import sys
from pathlib import Path

from vera import public_fetch as pf

_BASE = Path(__file__).resolve().parent.parent


def _load(name):
    spec = importlib.util.spec_from_file_location(f"s4a_{name}", _BASE / "vera" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


ol = _load("official_lane")
REG = [r for r in json.loads((_BASE / "config" / "entity_registry.json").read_text(encoding="utf-8"))["entities"]
       if r.get("state") == "active"]
CFG = {"user_agent": "UA-test", "deny_hosts": [], "max_redirects": 2, "max_compressed_bytes": 50000,
       "max_decompressed_bytes": 200000, "max_robots_bytes": 2000, "timeout_s": 5,
       "content_types": ["text/html", "text/plain"]}
IP = ["93.184.216.34"]


class Resp:
    def __init__(self, status=200, headers=None, body=b""):
        self.status_code, self.headers, self.body = status, headers or {}, body

    def iter_content(self, n):
        for i in range(0, len(self.body), n):
            yield self.body[i:i + n]

    def close(self):
        pass


def html(text=b"<p>Official docs page about message formatting.</p>"):
    return Resp(200, {"Content-Type": "text/html; charset=utf-8"}, text)


def robots():
    return Resp(200, {"Content-Type": "text/plain"}, b"User-agent: *\nAllow: /\n")


def redirect(to):
    return Resp(302, {"Location": to})


class Net:
    """Fake DNS + transport. `sent` lists every (host, target) actually requested."""
    def __init__(self, pages):
        self.pages, self.sent = pages, []

    def resolve(self, host, port):
        return [(socket.AF_INET, 0, 0, "", (a, port)) for a in IP]

    def transport(self, host, ip, port, target, headers, timeout):
        self.sent.append((host, target))
        return self.pages[(host, target)]

    def fetch(self, url, **kw):
        return pf.fetch(url, CFG, resolve=self.resolve, transport=self.transport, **kw)


def _lane(net):
    return ol.OfficialLane({"enabled": True, "site_search": False}, registry=REG, fetch=net.fetch)


def _slack_row():
    return next(r for r in REG if r["entity"] == "Slack")


def test_hop_predicate_pins_row_hosts_and_avoid_paths():
    row = _slack_row()
    assert ol.hop_ok("https://docs.slack.dev/messaging", row)
    assert ol.hop_ok("https://api.slack.com/robots.txt", row)
    assert not ol.hop_ok("https://evil.example/x", row)
    assert not ol.hop_ok("https://sub.docs.slack.dev/x", row)  # exact host only, no suffix match
    assert not ol.hop_ok("https://slack.com/oauth/v2/authorize", row)  # avoid_paths
    assert not ol.hop_ok("not a url", row)


def test_non_canonical_paths_fail_closed_sec_m1():
    pul = next(r for r in REG if r["entity"] == "Pulsar")
    kafka = next(r for r in REG if r["entity"] == "Kafka")  # no avoid_paths: still fails closed
    for u in ("https://pulsar.apache.org/x/../docs/2.11.x/", "https://pulsar.apache.org//docs/2.11.x/",
              "https://pulsar.apache.org/docs/./2.11.x/", "https://pulsar.apache.org/a%2e%2e/docs/",
              "https://pulsar.apache.org/a%2Fb", "https://pulsar.apache.org/a\\b"):
        assert not ol.hop_ok(u, pul), u
        assert not ol.hop_ok(u.replace("pulsar", "kafka"), kafka), u
    assert not ol.page_ok("https://pulsar.apache.org/docs/%32.11.x/", pul)  # decoded once, then not allowed
    gc = next(r for r in REG if r["entity"] == "Google Cloud")
    assert not ol.hop_ok("https://docs.cloud.google.com/docs?authuser%3D1", gc)  # encoded query decoded
    assert ol.hop_ok("https://kafka.apache.org/documentation/", kafka)
    assert ol.hop_ok("https://pulsar.apache.org/docs/next/getting-started", pul)


def test_case_variant_paths_are_avoided():
    row = _slack_row()
    assert not ol.hop_ok("https://slack.com/OAuth/v2/authorize", row)
    assert not ol.hop_ok("https://slack.com/MESSAGES/x", row)
    gc = next(r for r in REG if r["entity"] == "Google Cloud")
    assert not ol.hop_ok("https://docs.cloud.google.com/docs?AuthUser=1", gc)


def test_deny_all_allowlist_semantics():
    rows = {r["entity"]: r for r in REG}
    gc, kafka, pul, sl = rows["Google Cloud"], rows["Kafka"], rows["Pulsar"], rows["Slack"]
    allowed = [("https://docs.cloud.google.com/docs", gc), ("https://docs.cloud.google.com/docs/run/quotas", gc),
               ("https://kafka.apache.org/", kafka), ("https://kafka.apache.org/documentation/streams/", kafka),
               ("https://pulsar.apache.org/docs/4.2.x/concepts", pul), ("https://pulsar.apache.org/docs/next/", pul),
               ("https://docs.slack.dev/apis/web-api", sl), ("https://docs.slack.dev/x#frag", sl)]
    for u, r in allowed:
        assert ol.page_ok(u, r), u
    denied = [("https://docs.cloud.google.com/docsx", gc), ("https://docs.cloud.google.com/python/docs/reference/a", gc),
              ("https://docs.cloud.google.com/docs?hl=en", gc), ("https://docs.cloud.google.com/DOCS/run", gc),
              ("https://docs.cloud.google.com/", gc), ("https://kafka.apache.org/blog", kafka),
              ("https://pulsar.apache.org/docs/4.1.x/x", pul), ("https://pulsar.apache.org/docs/2.11.x/", pul),
              ("https://pulsar.apache.org/api/admin/", pul), ("https://slack.com/help", sl),
              ("https://api.slack.com/", sl), ("https://docs.slack.dev/a;b", sl), ("https://docs.slack.dev/a%00b", sl),
              ("https://docs.slack.dev/a%zz", sl), ("https://docs.slack.dev/a/./b", sl), ("https://docs.slack.dev/a//b", sl),
              ("https://docs.slack.dev/a/%2e%2e/b", sl), ("https://docs.slack.dev/%252e", sl),
              ("https://docs.slack.dev/\uff41", sl), ("https://docs.slack.dev/a%C3%A9", sl),
              ("https://docs.slack.dev/a b", sl), ("https://docs.slack.dev/a\\b", sl)]
    for u, r in denied:
        assert not ol.page_ok(u, r), u
    for name in ("AWS", "Nvidia", "Azure", "Snowflake", "Teams", "Zoom"):
        assert rows[name]["allowed_paths"] == {}
        for u in rows[name].get("docs_entry_urls") or []:
            assert not ol.page_ok(u, rows[name])
    assert not ol.page_ok("https://docs.slack.dev/x", dict(sl, allowed_paths=None))  # malformed: nothing
    assert not ol.page_ok("https://docs.slack.dev/x", dict(sl, allowed_paths={"docs.slack.dev": {"prefix": ["/x"]}}))
    assert ol.hop_ok("https://pulsar.apache.org/robots.txt", pul) and not ol.page_ok("https://pulsar.apache.org/robots.txt", pul)


def test_d3_only_explicit_allow_is_read():
    base = {"domains": ["x.example"], "docs_entry_urls": ["https://x.example/d"], "state": "active",
            "allowed_paths": {"x.example": {"exact": ["/d"]}}}
    cases = {"U": {"access_verdict": "unknown"}, "M": {}, "T": {"terms_verdict": "unknown"},
             "O": {"access_verdict": "ALLOW "}, "F": {"access_verdict": "forbid"},
             "L": {"access_verdict": "allow", "terms_verdict": "links_only"}}
    for ent, extra in cases.items():
        row = dict(base, entity=ent, **extra)
        assert ol.verdict(row) == "links_only", ent
        net = Net({})
        got = ol.OfficialLane({"enabled": True}, registry=[row], fetch=net.fetch).run([ent])
        assert net.sent == [] and got["pages"] == [] and got["links"][0]["read"] is False, ent
    assert ol.verdict(dict(base, access_verdict="allow")) == "allow"
    assert ol.verdict(dict(base, terms_verdict="allow")) == "allow"  # access derived from an explicit terms allow
    assert ol.verdict(None) == "links_only"


def test_sec3_residuals_closed():
    rows = {r["entity"]: r for r in REG}
    gc, sl = rows["Google Cloud"], rows["Slack"]
    for bad in ({"prefix": "/docs/"}, {"exact": "/docs"}, {"exact": ["/docs"], "allow": ["/"]}):  # S3-1
        r = dict(gc, allowed_paths={"docs.cloud.google.com": bad})
        assert not ol.page_ok("https://docs.cloud.google.com/docs", r) and not ol.page_ok("https://docs.cloud.google.com/x", r)
    for u in ("https://docs.cloud.google.com/docs%2Fx", "https://docs.cloud.google.com/docs/%41b"):  # S3-2
        assert not ol.page_ok(u, gc), u
    for u in ("https://docs.slack.dev/robots.txt", "https://docs.slack.dev/llms.txt",
              "https://docs.slack.dev/.well-known/tdmrep.json"):  # S3-3
        assert not ol.page_ok(u, sl), u
    assert ol.hop_ok("https://docs.slack.dev/robots.txt", sl)  # still fetchable as robots.txt (to honour it)


def test_metadata_hosts_never_ok_even_if_a_row_lists_them():
    row = {"entity": "Evil", "state": "active", "domains": ["metadata.google.internal"],
           "docs_entry_urls": ["https://metadata.google.internal/computeMetadata/v1/"]}
    assert not ol.hop_ok("https://metadata.google.internal/computeMetadata/v1/", row)
    assert not ol.hop_ok("https://METADATA.GOOGLE.INTERNAL/x", row)
    net = Net({})
    got = ol.OfficialLane({"enabled": True}, registry=[row], fetch=net.fetch).run(["Evil"])
    assert got == {"links": [], "pages": []} and net.sent == []  # row dropped at load


def test_registry_host_ascii_idna_check_drops_bad_rows():
    for bad in ("docs.slаck.dev",  # Cyrillic a
                "Docs.Slack.Dev", "*.slack.dev", "slack.com:443", "slack.com/x", "user@slack.com", "93.184.216.34",
                "localhost", "a.internal", "xn--.com", "", None, "bücher.de"):
        assert not ol.valid_host(bad), bad
    for good in ("docs.slack.dev", "kafka.apache.org", "xn--bcher-kva.de"):
        assert ol.valid_host(good), good
    rows = [dict(_slack_row(), entity="Mixed", domains=["docs.slack.dev", "docs.slаck.dev"]), _slack_row()]
    lane = ol.OfficialLane({"enabled": True}, registry=rows, fetch=Net({}).fetch)
    assert set(lane._rows) == {"Slack"}  # one non-ASCII host drops the whole row
    assert all(ol.valid_host(h) for r in REG for h in r["domains"])  # every shipped active host passes


def test_sec2_rereview_residuals_fail_closed():
    pul = next(r for r in REG if r["entity"] == "Pulsar")
    for u in ("https://pulsar.apache.org/%252e%252e/docs/2.11.x/", "https://pulsar.apache.org/docs;x/2.11.x/",
              "https://pulsar.apache.org/\uff44ocs/2.11.x/"):
        assert not ol.hop_ok(u, pul), u
    assert not ol.valid_host("xn--zz.com")  # bogus A-label (N1)
    assert ol.valid_host("xn--bcher-kva.de")


def test_redirect_to_off_registry_host_is_never_requested():
    net = Net({("docs.slack.dev", "/robots.txt"): robots(), ("docs.slack.dev", "/"): redirect("https://evil.example/p"),
               ("evil.example", "/robots.txt"): robots(), ("evil.example", "/p"): html()})
    got = _lane(net).run(["Slack"])
    assert not got["pages"]
    assert got["links"][0]["reason"] == "read_lane_hop_refused"
    assert not any(h == "evil.example" for h, _ in net.sent)  # neither robots.txt nor the page was sent


def test_redirect_within_the_row_is_followed():
    net = Net({("docs.slack.dev", "/robots.txt"): robots(),
               ("docs.slack.dev", "/"): redirect("https://docs.slack.dev/messaging")})
    net.pages[("docs.slack.dev", "/messaging")] = html()
    got = _lane(net).run(["Slack"])
    assert got["links"][0]["read"] is True and len(got["pages"]) == 1
    assert got["pages"][0].url == "https://docs.slack.dev/messaging"
    # the real cloud.google.com/docs -> docs.cloud.google.com 301: both hosts allow /docs/
    gc = dict(next(r for r in REG if r["entity"] == "Google Cloud"), docs_entry_urls=["https://cloud.google.com/docs"])
    net = Net({("cloud.google.com", "/robots.txt"): robots(),
               ("cloud.google.com", "/docs"): redirect("https://docs.cloud.google.com/docs/overview"),
               ("docs.cloud.google.com", "/robots.txt"): robots(), ("docs.cloud.google.com", "/docs/overview"): html()})
    got = ol.OfficialLane({"enabled": True}, registry=[gc], fetch=net.fetch).run(["Google Cloud"])
    assert got["links"][0]["read"] is True


def test_redirect_within_row_host_to_unlisted_path_is_refused():
    net = Net({("docs.slack.dev", "/robots.txt"): robots(),
               ("docs.slack.dev", "/"): redirect("https://api.slack.com/docs"),
               ("api.slack.com", "/robots.txt"): robots(), ("api.slack.com", "/docs"): html()})
    got = _lane(net).run(["Slack"])
    # API-URL-GATE-1108b: the api.* host is now refused by the API gate before the read gate (either refusal holds)
    assert got["links"][0]["reason"] in ("read_lane_hop_refused", "api_denied") and not got["pages"]
    assert ("api.slack.com", "/docs") not in net.sent  # api.slack.com has no allowed path (deny all)


def test_redirect_to_avoided_path_on_row_host_is_refused():
    net = Net({("docs.slack.dev", "/robots.txt"): robots(),
               ("docs.slack.dev", "/"): redirect("https://slack.com/oauth/v2/authorize"),
               ("slack.com", "/robots.txt"): robots(), ("slack.com", "/oauth/v2/authorize"): html()})
    got = _lane(net).run(["Slack"])
    assert got["links"][0]["reason"] == "read_lane_hop_refused" and not got["pages"]
    assert ("slack.com", "/oauth/v2/authorize") not in net.sent


def test_link_only_rows_are_never_fetched():
    net = Net({})
    got = _lane(net).run(["AWS", "Nvidia", "Azure", "Snowflake", "Teams", "Zoom"])
    assert net.sent == [] and got["pages"] == []
    assert got["links"] and all(lk["read"] is False for lk in got["links"])
    hosts = {ol._host(lk["url"]) for lk in got["links"]}
    assert "support.zoom.com" in hosts and "support.zoom.us" not in hosts


def test_avoided_entry_url_gives_link_only_without_request():
    row = dict(_slack_row(), docs_entry_urls=["https://slack.com/messages/x"])
    net = Net({})
    lane = ol.OfficialLane({"enabled": True}, registry=[row], fetch=net.fetch)
    got = lane.run(["Slack"])
    assert got["links"][0]["reason"] == "path_not_allowed" and net.sent == []


def test_should_stop_is_checked_before_each_fetch():
    net = Net({})
    got = _lane(net).run(["Kafka", "Pulsar"], should_stop=lambda: "deadline")
    assert net.sent == [] and {lk["reason"] for lk in got["links"]} == {"deadline"}
    got = _lane(net).run(["Kafka"], should_stop=lambda: 1 / 0)  # a failing stop check stops (fail closed)
    assert net.sent == [] and got["links"][0]["reason"] == "stopped"


def test_flag_off_does_nothing():
    net = Net({})
    lane = ol.OfficialLane({"enabled": False}, registry=REG, fetch=net.fetch)
    assert lane.run(["Kafka", "Slack"]) == {"links": [], "pages": []} and net.sent == []


def test_shipped_flag_on_for_fec_census_site_search_off():
    p = _BASE / "config" / "subquery.json"
    if not p.exists():  # staged layout: the shipped file is the repo's
        p = _BASE.parents[2] / "repo" / "config" / "subquery.json"
    cfg = json.loads(p.read_text(encoding="utf-8"))
    assert cfg["official_lane"]["enabled"] is True and cfg["official_lane"]["site_search"] is False  # H-2


if __name__ == "__main__":
    n = 0
    for k, f in list(globals().items()):
        if k.startswith("test_") and callable(f):
            f()
            n += 1
    print(f"{n} passed")

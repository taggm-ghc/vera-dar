"""WPC-HARVEST-IMPL (USFED-WPC-1112): scripts/refresh_permitted_catalogue.py, offline. DNS and transport are fakes, sockets
raise; example.gov-style synthetic rows and pages; no personal data; nothing is written outside tmp_path."""
from __future__ import annotations

import copy
import importlib.util
import json
import socket
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from vera import acq_discovered, html_main, original_html as oh

_spec = importlib.util.spec_from_file_location(
    "refresh_permitted_catalogue", Path(__file__).resolve().parent.parent / "scripts" / "refresh_permitted_catalogue.py")
R = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(R)

SHIPPED = json.loads(Path(oh.CONFIG_PATH).read_text(encoding="utf-8"))
HA, HS = "www.example.gov", "science.example.gov"
ROW_A = {"id": "a", "host": HA, "path_prefixes": ["/explainers"], "path_stems": [], "licence_basis": "us_federal_pd",
         "agency": "Example Agency", "discovery": {"index_url": f"https://{HA}/explainers"}, "crawl_delay_s": 5,
         "text_only": True, "enabled": True, "store": True, "review": {"task": "T", "date": "2026-11-12"}}
ROW_S = {"id": "s", "host": HS, "path_prefixes": ["/dark"], "path_stems": [], "licence_basis": "us_federal_pd",
         "agency": "Example Agency", "discovery": {"sitemap_url": f"https://{HS}/sitemap.xml"}, "crawl_delay_s": 5,
         "text_only": True, "enabled": True, "store": False, "review": {"task": "T", "date": "2026-11-12"}}
ROBOTS_OK = b"User-agent: *\nAllow: /\n"
SENTINEL_TEXT = "SENTINEL-PAGE-TEXT-DO-NOT-PRINT"


class Resp:
    def __init__(self, status=200, ctype="text/html", body=b"", headers=None):
        self.status_code, self.body = status, body
        self.headers = {"Content-Type": ctype, **(headers or {})}

    def iter_content(self, n):
        for i in range(0, len(self.body), n):
            yield self.body[i:i + n]

    def close(self):
        pass


class Net:
    def __init__(self, pages):
        self.pages, self.calls, self.lookups = pages, [], []

    def resolve(self, host, port):
        self.lookups.append(host)
        return [(socket.AF_INET, 0, 0, "", ("93.184.216.34", port))]

    def transport(self, host, ip, port, target, headers, timeout):
        self.calls.append((host, target))
        return self.pages.get((host, target)) or Resp(404, "text/plain", b"")

    def targets(self, host=HA):
        return [t for h, t in self.calls if h == host]


class Clock:
    def __init__(self):
        self.t, self.sleeps = 1000.0, []

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.sleeps.append(s)
        self.t += s


def robots(txt=ROBOTS_OK):
    return Resp(200, "text/plain", txt)


def page(title, paras=(), heading="Shared heading"):
    body = "".join(f"<p>{p}</p>" for p in paras)
    return Resp(200, "text/html", (f"<html><head><title>{title}</title></head><body><main><h1>{title}</h1><h2>{heading}</h2>"
                                   f"{body}</main></body></html>").encode())


def index(*hrefs):
    links = "".join(f'<a href="{h}">link</a> ' for h in hrefs)
    return Resp(200, "text/html", f"<html><head><title>Index</title></head><body><main>{links}</main></body></html>".encode())


def sitemap(*locs, kind="urlset"):
    inner = "".join(f"<url><loc>{u}</loc></url>" if kind == "urlset" else f"<sitemap><loc>{u}</loc></sitemap>" for u in locs)
    return Resp(200, "application/xml", f'<?xml version="1.0"?><{kind} xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">{inner}</{kind}>'.encode())


def u(p, host=HA):
    return f"https://{host}{p}"


def site(n=3, **extra):
    """robots + an index linking /explainers/p0..p{n-1}, each a page."""
    p = {(HA, "/robots.txt"): robots(), (HA, "/explainers"): index(*[f"/explainers/p{i}" for i in range(n)])}
    for i in range(n):
        p[(HA, f"/explainers/p{i}")] = page(f"Page {i}", [f"Unique paragraph {i} " * 3, "Common notice text for all pages."])
    p.update(extra)
    return p


@pytest.fixture(autouse=True)
def env(monkeypatch, tmp_path):
    def boom(*a, **k):
        raise AssertionError("network used")
    monkeypatch.setattr(socket, "socket", boom)
    monkeypatch.setattr(socket, "create_connection", boom)
    monkeypatch.setattr(socket, "getaddrinfo", boom)
    acq_discovered.reset_for_tests()
    monkeypatch.setattr(R, "scratch_root", lambda: tmp_path.resolve() / "scratch")
    (tmp_path / "config").mkdir()
    (tmp_path / "scratch").mkdir()
    monkeypatch.setattr(oh, "_ROOT", tmp_path)
    cfg = copy.deepcopy(SHIPPED)
    cfg["rows"], cfg["mode"], cfg["snapshot"] = [copy.deepcopy(ROW_A), copy.deepcopy(ROW_S)], "shadow", {"path": "config/permitted_catalogue.json"}
    monkeypatch.setattr(oh, "load_config", lambda path=None: cfg)
    oh.reset_for_tests()
    yield cfg
    acq_discovered.reset_for_tests()
    oh.reset_for_tests()


SNAP = lambda tp: tp / "config" / "permitted_catalogue.json"  # noqa: E731


def evidence(tmp_path, row="a", host=HA, age_min=1, allowed=True):
    p = tmp_path / "scratch" / "ev.json"
    at = datetime.now(timezone.utc) - timedelta(minutes=age_min)
    p.write_text(json.dumps({"version": 1, "row": row, "host": host, "checked_at": R.iso(at), "allowed": allowed,
                             "read": "parsed_200", "crawl_delay_s": 0, "verdicts": {}}), encoding="utf-8")
    return str(p)


def harvest(tmp_path, pages, row="a", write=False, ev=None, clock=None, extra=(), **kw):
    net, clock = Net(pages), clock or Clock()
    argv = ["--harvest", "--row", row, "--robots-evidence", ev or evidence(tmp_path, row, HA if row == "a" else HS),
            *(["--write"] if write else []), *extra]
    rc = R.main(argv, resolve=net.resolve, transport=net.transport, clock=clock, sleep=clock.sleep, **kw)
    return rc, net, clock


def snap(tmp_path):
    return json.loads(SNAP(tmp_path).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------------------------- robots-only phase
def test_robots_only_fetches_only_robots_and_writes_evidence(tmp_path, capsys):
    net = Net({(HA, "/robots.txt"): robots(b"User-agent: *\nDisallow: /private\nCrawl-delay: 7\n"), (HA, "/explainers"): index()})
    out = tmp_path / "scratch" / "r.json"
    rc = R.main(["--robots-only", "--row", "a", "--robots-out", str(out)], resolve=net.resolve, transport=net.transport)
    assert rc == 0 and net.calls == [(HA, "/robots.txt")]
    ev = json.loads(out.read_text())
    assert ev["allowed"] is True and ev["host"] == HA and ev["row"] == "a" and ev["crawl_delay_s"] == 7.0
    assert ev["verdicts"] == {"discovery": "allow", "/explainers": "allow"}
    assert "discovery: allow" in capsys.readouterr().out


def test_robots_only_disallowed_is_not_allowed_evidence(tmp_path):
    net = Net({(HA, "/robots.txt"): robots(b"User-agent: *\nDisallow: /\n")})
    out = tmp_path / "scratch" / "r.json"
    assert R.main(["--robots-only", "--row", "a", "--robots-out", str(out)], resolve=net.resolve, transport=net.transport) == 2
    assert json.loads(out.read_text())["allowed"] is False and net.calls == [(HA, "/robots.txt")]


def test_robots_out_must_be_under_scratch(tmp_path):
    net = Net({})
    with pytest.raises(SystemExit):
        R.main(["--robots-only", "--row", "a", "--robots-out", str(tmp_path / "elsewhere.json")], resolve=net.resolve,
               transport=net.transport)
    assert net.calls == [] and not (tmp_path / "elsewhere.json").exists()


# ---------------------------------------------------------------------------------------------- evidence refusal
@pytest.mark.parametrize("kind", ["missing", "stale", "other_host", "other_row", "disallowed"])
def test_harvest_refuses_bad_evidence_before_any_request(tmp_path, kind):
    ev = {"missing": str(tmp_path / "scratch" / "none.json"),
          "stale": evidence(tmp_path, age_min=120),
          "other_host": evidence(tmp_path, host=HS),
          "other_row": evidence(tmp_path, row="s"),
          "disallowed": evidence(tmp_path, allowed=False)}[kind]
    net = Net(site())
    with pytest.raises(SystemExit) as e:
        R.main(["--harvest", "--row", "a", "--robots-evidence", ev, "--write"], resolve=net.resolve, transport=net.transport)
    assert "FAIL" in str(e.value) and net.calls == [] and net.lookups == [] and not SNAP(tmp_path).exists()


def test_harvest_needs_evidence_argument():
    with pytest.raises(SystemExit):
        R.main(["--harvest", "--row", "a"])


# ---------------------------------------------------------------------------------------------- stop rules
def test_robots_403_stops_host_with_no_page_request(tmp_path):
    pages = site()
    pages[(HA, "/robots.txt")] = Resp(403, "text/plain", b"")
    rc, net, _ = harvest(tmp_path, pages, write=True)
    assert rc == 2 and net.calls == [(HA, "/robots.txt")]
    s = snap(tmp_path)
    assert s["robots_status"][HA]["harvest"] == "stopped:robots_unreadable" and s["entries"] == [] and s["denied"] == []
    assert oh.load_snapshot() is not None


def test_robots_disallow_all_stops_host_and_removes_prior_entries(tmp_path):
    assert harvest(tmp_path, site(2), write=True)[0] == 0 and len(snap(tmp_path)["entries"]) == 2
    pages = site()
    pages[(HA, "/robots.txt")] = robots(b"User-agent: *\nDisallow: /\n")
    oh.reset_for_tests()
    rc, net, _ = harvest(tmp_path, pages, write=True)
    # the discovery URL is disallowed: nothing but robots.txt is requested and the host's earlier entries are removed
    s = snap(tmp_path)
    assert rc == 2 and net.calls == [(HA, "/robots.txt")] and s["entries"] == [] and HA not in s["boilerplate"]
    assert s["robots_status"][HA]["harvest"] == "stopped:robots_disallowed"


def test_page_level_robots_disallow_is_counted_not_a_stop(tmp_path, capsys):
    pages = site(3)
    pages[(HA, "/robots.txt")] = robots(b"User-agent: *\nDisallow: /explainers/p1\n")
    rc, net, _ = harvest(tmp_path, pages, write=True)
    assert rc == 0 and "/explainers/p1" not in net.targets()
    assert snap(tmp_path)["robots_status"][HA]["counts"]["robots_skipped"] == 1


def test_robots_only_cross_host_redirect_fails_closed_and_robots_requested_once(tmp_path):
    net = Net({(HA, "/robots.txt"): Resp(302, "text/plain", b"", {"Location": "https://evil.example.com/robots.txt"})})
    out = tmp_path / "scratch" / "r.json"
    rc = R.main(["--robots-only", "--row", "a", "--robots-out", str(out)], resolve=net.resolve, transport=net.transport)
    assert rc == 2 and net.calls == [(HA, "/robots.txt")]
    ev = json.loads(out.read_text())
    assert ev["allowed"] is False and ev["read"] == "unreadable" and ev["verdicts"]["discovery"] == "unreadable"
    assert set(ev["verdicts"].values()) <= {"unreadable", "not_checked"}


def test_discovery_gate_admits_only_discovery_child_and_robots_paths(tmp_path):
    pages = site(2)
    pages[(HA, "/explainers")] = Resp(302, "text/html", b"", {"Location": f"https://{HA}/somewhere/else"})
    pages[(HA, "/somewhere/else")] = index("/explainers/p0")
    rc, net, _ = harvest(tmp_path, pages, write=True)
    assert rc == 2 and "/somewhere/else" not in net.targets() and snap(tmp_path)["entries"] == []


def test_redirect_hop_is_paced(tmp_path):
    pages = site(1)
    pages[(HA, "/explainers/p0")] = Resp(302, "text/html", b"", {"Location": f"https://{HA}/explainers/p0b"})
    pages[(HA, "/explainers/p0b")] = page("Moved")
    rc, net, clock = harvest(tmp_path, pages)
    assert rc == 0 and net.targets()[-2:] == ["/explainers/p0", "/explainers/p0b"] and clock.sleeps == [5.0] * 3


@pytest.mark.parametrize("how", ["dns", "oserror"])
def test_mid_run_network_failure_stops_host_and_is_not_complete(tmp_path, how):
    pages = site(4)
    net = Net(pages)
    if how == "dns":
        real = net.resolve

        def resolve(host, port):
            if len(net.lookups) >= 3:  # robots, index, p0 resolve; p1 fails
                raise socket.gaierror("no")
            return real(host, port)
        res, tr = resolve, net.transport
    else:
        def tr(host, ip, port, target, headers, timeout):
            if target == "/explainers/p1":
                raise OSError("reset")
            return net.transport(host, ip, port, target, headers, timeout)
        res = net.resolve
    clock = Clock()
    ev = evidence(tmp_path)
    rc = R.main(["--harvest", "--row", "a", "--robots-evidence", ev, "--write"], resolve=res, transport=tr, clock=clock,
                sleep=clock.sleep)
    s = snap(tmp_path)
    assert rc == 2 and s["robots_status"][HA]["harvest"].startswith("stopped:") and "/explainers/p2" not in net.targets()
    assert s["robots_status"][HA]["harvest"] != "complete"


def test_write_refused_when_snapshot_does_not_validate(tmp_path, monkeypatch):
    monkeypatch.setattr(oh, "_snapshot_valid", lambda s, c: False)
    with pytest.raises(SystemExit) as e:
        harvest(tmp_path, site(2), write=True)
    assert "does not validate" in str(e.value) and not SNAP(tmp_path).exists()


def test_userinfo_port_and_lookalike_host_links_dropped(tmp_path):
    pages = {(HA, "/robots.txt"): robots(),
             (HA, "/explainers"): index(f"https://user@{HA}/explainers/u", f"https://{HA}:8443/explainers/port",
                                       f"https://{HA}.example.com/explainers/z", f"https://x{HA}/explainers/y",
                                       "/explainers/good"),
             (HA, "/explainers/good"): page("Good")}
    rc, net, _ = harvest(tmp_path, pages)
    assert rc == 0 and net.targets() == ["/robots.txt", "/explainers", "/explainers/good"]
    assert all(h == HA for h, _ in net.calls)


@pytest.mark.parametrize("code", [401, 403])
def test_page_denied_goes_to_denied_list_one_request_no_variant(tmp_path, code):
    pages = site(4)
    pages[(HA, "/explainers/p1")] = Resp(code, "text/html", b"")
    rc, net, _ = harvest(tmp_path, pages, write=True)
    assert rc == 2
    t = net.targets()
    assert t.count("/explainers/p1") == 1 and "/explainers/p2" not in t and "/explainers/p3" not in t
    assert set(t) <= {"/robots.txt", "/explainers", "/explainers/p0", "/explainers/p1"}  # no variants
    s = snap(tmp_path)
    assert [(d["url"], d["code"]) for d in s["denied"]] == [(u("/explainers/p1"), str(code))]
    assert s["denied"][0]["seen_at"] and s["robots_status"][HA]["harvest"] == f"stopped:denied:{code}"
    assert oh.load_snapshot() is not None  # still validates
    assert u("/explainers/p1") not in {e["url"] for e in s["entries"]}


def test_denied_url_is_never_requested_in_later_run(tmp_path):
    pages = site(3)
    pages[(HA, "/explainers/p1")] = Resp(403, "text/html", b"")
    harvest(tmp_path, pages, write=True)
    pages[(HA, "/explainers/p1")] = page("Page 1 now open")
    oh.reset_for_tests()
    rc, net, _ = harvest(tmp_path, pages, write=True)
    assert rc == 0 and "/explainers/p1" not in net.targets()
    s = snap(tmp_path)
    assert [e["url"] for e in s["entries"]] == [u("/explainers/p0"), u("/explainers/p2")] and len(s["denied"]) == 1


def test_index_403_is_denied_and_stops(tmp_path):
    pages = site()
    pages[(HA, "/explainers")] = Resp(403, "text/html", b"")
    rc, net, _ = harvest(tmp_path, pages, write=True)
    assert rc == 2 and net.targets() == ["/robots.txt", "/explainers"]
    assert snap(tmp_path)["denied"][0]["url"] == u("/explainers")


@pytest.mark.parametrize("code", [429, 503])
def test_429_and_5xx_stop_host_without_retry(tmp_path, code):
    pages = site(4)
    pages[(HA, "/explainers/p1")] = Resp(code, "text/html", b"")
    rc, net, _ = harvest(tmp_path, pages, write=True)
    t = net.targets()
    assert rc == 2 and t.count("/explainers/p1") == 1 and "/explainers/p2" not in t
    s = snap(tmp_path)
    assert s["robots_status"][HA]["harvest"] == f"stopped:status:{code}" and s["denied"] == []


# ---------------------------------------------------------------------------------------------- crawl delay
def test_robots_crawl_delay_larger_than_row_delay_is_honoured(tmp_path):
    pages = site(3)
    pages[(HA, "/robots.txt")] = robots(b"User-agent: *\nCrawl-delay: 9\n")
    rc, net, clock = harvest(tmp_path, pages)
    assert rc == 0 and clock.sleeps == [9.0] * 4  # robots.txt first, then a wait before the index and each of the 3 pages


def test_row_crawl_delay_used_when_robots_has_none(tmp_path):
    rc, net, clock = harvest(tmp_path, site(2))
    assert rc == 0 and clock.sleeps == [5.0] * 3  # the first discovery GET waits too


# ---------------------------------------------------------------------------------------------- caps
def test_max_urls_per_host_and_title_fetch_caps(tmp_path, env):
    env["harvest"] = {"max_urls_per_host": 3}
    rc, net, _ = harvest(tmp_path, site(6))
    assert [t for t in net.targets() if t.startswith("/explainers/")] == [f"/explainers/p{i}" for i in range(3)]
    env["harvest"] = {"max_urls_per_host": 50, "max_title_fetches_per_run": 2}
    rc, net, _ = harvest(tmp_path, site(6))
    assert len([t for t in net.targets() if t.startswith("/explainers/")]) == 2


def test_child_sitemap_cap(tmp_path, env):
    env["harvest"] = {"max_child_sitemaps": 2}
    kids = [u(f"/sitemap-{i}.xml", HS) for i in range(4)]
    pages = {(HS, "/robots.txt"): robots(), (HS, "/sitemap.xml"): sitemap(*kids, kind="sitemapindex")}
    for i in range(4):
        pages[(HS, f"/sitemap-{i}.xml")] = sitemap(u(f"/dark/k{i}", HS))
        pages[(HS, f"/dark/k{i}")] = page(f"K {i}")
    rc, net, _ = harvest(tmp_path, pages, row="s", write=True)
    t = net.targets(HS)
    assert rc == 0 and [x for x in t if x.startswith("/sitemap-")] == ["/sitemap-0.xml", "/sitemap-1.xml"]
    assert sorted(e["url"] for e in snap(tmp_path)["entries"]) == [u("/dark/k0", HS), u("/dark/k1", HS)]


# ---------------------------------------------------------------------------------------------- link filtering
def test_off_row_lookalike_query_fragment_links_are_dropped(tmp_path):
    pages = {(HA, "/robots.txt"): robots(),
             (HA, "/explainers"): index("/explainers/ok-one", f"https://{HA}/explainersX/bad", "/explainers/q?x=1",
                                       "https://evil.example.com/explainers/z", f"http://{HA}/explainers/plain",
                                       "/other/p", "/explainers/frag#a", "/explainers/../secret", "/explainers/ok-two"),
             (HA, "/explainers/ok-one"): page("One"), (HA, "/explainers/ok-two"): page("Two")}
    rc, net, _ = harvest(tmp_path, pages, write=True)
    assert rc == 0 and net.targets() == ["/robots.txt", "/explainers", "/explainers/ok-one", "/explainers/ok-two"]
    assert [h for h, _ in net.calls if h != HA] == []
    assert [e["url"] for e in snap(tmp_path)["entries"]] == [u("/explainers/ok-one"), u("/explainers/ok-two")]


def test_redirect_off_host_is_not_followed_and_entry_dropped(tmp_path):
    pages = site(2)
    pages[(HA, "/explainers/p0")] = Resp(302, "text/html", b"", {"Location": "https://evil.example.com/x"})
    rc, net, _ = harvest(tmp_path, pages, write=True)
    assert rc == 0 and all(h == HA for h, _ in net.calls)
    assert [e["url"] for e in snap(tmp_path)["entries"]] == [u("/explainers/p1")]


def test_non_html_page_dropped(tmp_path):
    pages = site(2)
    pages[(HA, "/explainers/p0")] = Resp(200, "application/pdf", b"%PDF")
    rc, net, _ = harvest(tmp_path, pages, write=True)
    assert rc == 0 and [e["url"] for e in snap(tmp_path)["entries"]] == [u("/explainers/p1")]


# ---------------------------------------------------------------------------------------------- sitemap refusal
@pytest.mark.parametrize("body", [b'<?xml version="1.0"?><!DOCTYPE urlset [<!ENTITY a "b">]><urlset></urlset>',
                                  b"\x1f\x8b\x08\x00 gzip bytes"])
def test_doctype_and_gzip_sitemap_refused_no_page_request(tmp_path, body):
    pages = {(HS, "/robots.txt"): robots(), (HS, "/sitemap.xml"): Resp(200, "application/xml", body),
             (HS, "/dark/x"): page("X")}
    rc, net, _ = harvest(tmp_path, pages, row="s", write=True)
    assert rc == 2 and net.targets(HS) == ["/robots.txt", "/sitemap.xml"]
    assert snap(tmp_path)["entries"] == [] and snap(tmp_path)["robots_status"][HS]["harvest"].startswith("stopped:discovery_refused")


# ---------------------------------------------------------------------------------------------- boilerplate
def test_boilerplate_threshold_and_headings_kept(tmp_path):
    common, half, one = "Common notice text for all pages.", "Shared by exactly two of four pages.", "Only one page has this."
    pages = {(HA, "/robots.txt"): robots(), (HA, "/explainers"): index(*[f"/explainers/p{i}" for i in range(4)])}
    extras = [[half, one], [half], [], []]
    for i in range(4):
        pages[(HA, f"/explainers/p{i}")] = page(f"Page {i}", [f"Unique paragraph number {i}.", common, *extras[i]],
                                                heading="Shared heading")
    rc, net, _ = harvest(tmp_path, pages, write=True)
    b = set(snap(tmp_path)["boilerplate"][HA])
    assert rc == 0
    assert html_main.line_hash(common) in b and html_main.line_hash(half) in b  # 2 >= max(floor 2, 4/2)
    assert html_main.line_hash(one) not in b and html_main.line_hash("Unique paragraph number 0.") not in b
    assert html_main.line_hash("Shared heading") not in b and html_main.line_hash("Page 0") not in b  # headings never


def test_boilerplate_needs_half_of_pages_when_host_is_larger(tmp_path):
    pages = {(HA, "/robots.txt"): robots(), (HA, "/explainers"): index(*[f"/explainers/p{i}" for i in range(5)])}
    two, three = "On two of five pages.", "On three of five pages."
    for i in range(5):
        pages[(HA, f"/explainers/p{i}")] = page(f"Page {i}", [x for x, ok in ((two, i < 2), (three, i < 3)) if ok] + [f"Own {i}."])
    harvest(tmp_path, pages, write=True)
    b = set(snap(tmp_path)["boilerplate"][HA])
    assert html_main.line_hash(three) in b and html_main.line_hash(two) not in b  # need max(2, 2.5): 3


# ---------------------------------------------------------------------------------------------- output
def test_written_snapshot_validates_and_index_urls_works(tmp_path):
    rc, net, _ = harvest(tmp_path, site(3), write=True)
    assert rc == 0
    s = oh.load_snapshot()
    assert s is not None and s["sha256"] == oh.snapshot_digest(s) and s["script_version"] == R.SCRIPT_VERSION
    idx = oh.index_urls()
    assert idx == frozenset(u(f"/explainers/p{i}") for i in range(3))
    e = s["entries"][0]
    assert e["title"] == "Page 0" and e["host"] == HA and e["slug_terms"] == ["p0"]
    assert s["robots_status"][HA]["harvest"] == "complete" and s["robots_status"][HA]["robots"] == "parsed_200"
    assert oh.match_subject("page 1") and oh.match_subject("page 1")[0]["url"] == u("/explainers/p1")


def test_merge_replaces_only_the_runs_host(tmp_path):
    harvest(tmp_path, {(HS, "/robots.txt"): robots(), (HS, "/sitemap.xml"): sitemap(u("/dark/m", HS)),
                       (HS, "/dark/m"): page("Dark matter", ["Text one.", "Text two."])}, row="s", write=True)
    before = snap(tmp_path)
    oh.reset_for_tests()
    harvest(tmp_path, site(2), write=True)
    s = snap(tmp_path)
    assert [e for e in s["entries"] if e["host"] == HS] == [e for e in before["entries"] if e["host"] == HS]
    assert {e["host"] for e in s["entries"]} == {HA, HS} and s["boilerplate"][HS] == before["boilerplate"][HS]
    assert s["robots_status"][HS] == before["robots_status"][HS] and oh.load_snapshot() is not None
    assert oh.index_urls() is not None


def test_stopped_run_keeps_earlier_entries_of_that_host(tmp_path):
    harvest(tmp_path, site(2), write=True)
    pages = site(2)
    pages[(HA, "/explainers/p0")] = Resp(429, "text/html", b"")
    oh.reset_for_tests()
    rc, *_ = harvest(tmp_path, pages, write=True)
    s = snap(tmp_path)
    assert rc == 2 and len(s["entries"]) == 2 and s["robots_status"][HA]["harvest"] == "stopped:status:429"
    assert oh.load_snapshot() is not None


def test_robots_failure_removes_hosts_earlier_entries(tmp_path):
    harvest(tmp_path, site(2), write=True)
    pages = site(2)
    pages[(HA, "/robots.txt")] = Resp(403, "text/plain", b"")
    oh.reset_for_tests()
    harvest(tmp_path, pages, write=True)
    s = snap(tmp_path)
    assert s["entries"] == [] and s["boilerplate"].get(HA) is None and oh.load_snapshot() is not None


def test_dry_run_writes_nothing_and_prints_counts_only(tmp_path, capsys):
    SNAP(tmp_path).write_text("{}", encoding="utf-8")  # an unrelated file is not touched (and a bad one is refused on --write)
    pages = site(2)
    pages[(HA, "/explainers/p0")] = page(SENTINEL_TEXT, [SENTINEL_TEXT])
    SNAP(tmp_path).unlink()
    rc, net, _ = harvest(tmp_path, pages)
    out = capsys.readouterr().out
    assert rc == 0 and not SNAP(tmp_path).exists() and "dry run" in out and "kept=2" in out and SENTINEL_TEXT not in out
    assert not list((tmp_path / "config").iterdir())


def test_write_refuses_to_overwrite_a_malformed_snapshot(tmp_path):
    SNAP(tmp_path).write_text("not json", encoding="utf-8")
    with pytest.raises(SystemExit):
        harvest(tmp_path, site(2), write=True)
    assert SNAP(tmp_path).read_text() == "not json"


def test_save_pages_only_under_scratch(tmp_path):
    d = tmp_path / "scratch" / "pages"
    rc, *_ = harvest(tmp_path, site(2), extra=["--save-pages", str(d)])
    assert rc == 0 and len(list(d.iterdir())) == 2 and all(f.read_bytes().startswith(b"<html>") for f in d.iterdir())
    with pytest.raises(SystemExit):
        harvest(tmp_path, site(2), extra=["--save-pages", str(tmp_path / "outside")])
    assert not (tmp_path / "outside").exists()


def test_slug_terms_use_stem_remainder():
    row = {"path_stems": ["/science/doe-explains"], "path_prefixes": []}
    assert R.slug_terms(row, "https://www.energy.gov/science/doe-explainsquarks-and-gluons") == ["quarks", "and", "gluons"]
    assert R.slug_terms({"path_prefixes": ["/dark-matter"]}, "https://science.nasa.gov/dark-matter/what-is") == ["what", "is"]


# ---------------------------------------------------------------------------------------------- GC-MINORS-1113
@pytest.mark.parametrize("enc", ["gzip", "deflate", "br"])
def test_gc_minor3_non_identity_content_encoding_is_refused_never_decoded(tmp_path, enc):
    import gzip
    pages = site(2)
    good = pages[(HA, "/explainers/p0")]
    bad = Resp(200, "text/html", gzip.compress(good.body), {"Content-Encoding": enc})
    pages[(HA, "/explainers/p1")] = bad
    rc, net, _ = harvest(tmp_path, pages, write=True)
    urls = [e["url"] for e in snap(tmp_path)["entries"]]
    assert u("/explainers/p1") not in urls and u("/explainers/p0") in urls
    assert bad.headers["Content-Encoding"] != enc  # rewritten to a refused value; the body was never decoded


def test_gc_minor3_identity_encoding_still_accepted(tmp_path):
    pages = site(1)
    pages[(HA, "/explainers/p0")].headers["Content-Encoding"] = "identity"
    harvest(tmp_path, pages, write=True)
    assert [e["url"] for e in snap(tmp_path)["entries"]] == [u("/explainers/p0")]


def test_gc_minor1_robots_evidence_outside_scratch_or_symlink_is_refused(tmp_path):
    good = evidence(tmp_path)
    outside = tmp_path / "ev_outside.json"
    outside.write_text(Path(good).read_text(), encoding="utf-8")
    link = tmp_path / "scratch" / "ev_link.json"
    link.symlink_to(good)
    for ev in (str(outside), str(link), str(tmp_path / "scratch" / ".." / "ev_outside.json")):
        net = Net(site())
        with pytest.raises(SystemExit) as e:
            R.main(["--harvest", "--row", "a", "--robots-evidence", ev, "--write"], resolve=net.resolve, transport=net.transport)
        assert "FAIL" in str(e.value) and net.calls == [] and not SNAP(tmp_path).exists()
    rc, net, _ = harvest(tmp_path, site(1), ev=good)  # control: the same content under scratch/ is accepted
    assert rc == 0

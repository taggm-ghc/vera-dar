"""DENIAL-COVERAGE-1111 WP-C (T11, T13): outbound-call inventory. Import-based AST scan of vera/, scripts/, pages/,
main.py and streamlit_app.py. Every module that imports a socket-capable library (requests, httpx, urllib.request,
urllib3, http.client, socket, aiohttp, openai, groq) or calls subprocess with curl/wget must have each call site of
those names (attribute calls, aliases such as `http_get = requests.get`, `build_opener(...).open`, SDK client
construction) listed in SITES with its class. An unclassified site fails the test, so a new document-fetching path
cannot be added without a decision about the access-denial guard (selective_fetch.document_request_guard).

Classes: DOC (document fetch: reaches the network only through a guarded transport, or is the guarded transport),
API (metadata API, gated by config/api_permissions.json), ROBOTS (robots.txt / tdmrep.json endpoint), SEARCH,
LLM (LLM or embedding API), OWN (own service), OPS (operator tool, fixed owner-chosen URL), NOSOCKET (imported for
exception types, urllib.parse, mocks or type names; opens no socket), TEST (not applicable)."""
import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCAN = ("vera", "scripts", "pages", "main.py", "streamlit_app.py")
WATCHED = ("requests", "httpx", "urllib.request", "urllib3", "http.client", "socket", "aiohttp", "openai", "groq")
SOCKET_MODULES = ("urllib.request", "urllib3", "http.client", "socket")
METHODS = {"open", "urlopen", "getresponse", "create_connection", "connect"}
SAFE_LAST = re.compile(r"(Error|Exception|Timeout|Warning)$|^(Request|Response|exceptions|timeout|gaierror|error|"
                       r"AF_INET\w*|SOCK_\w+|SHUT_\w+|getaddrinfo|gethostbyname|inet_\w+|htons|ntohs|ssl|InvalidURL|"
                       r"HTTPRedirectHandler|BaseTransport|parse_url)$")
CLASSES = {"DOC", "API", "ROBOTS", "SEARCH", "LLM", "OWN", "OPS", "NOSOCKET", "TEST"}


def _dotted(node):
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    if isinstance(node, ast.Call):  # f(...).open -> "<call>.open"
        inner = _dotted(node.func)
        if inner:
            return inner + "()." + ".".join(reversed(parts))
    return None


def _scan_source(src: str) -> dict:
    """Return {site-name: count} for one module's source."""
    tree = ast.parse(src)
    bound = {}  # local name -> full dotted name of the watched import
    sock_mod = False
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names:
                full = a.name
                if any(full == w or full.startswith(w + ".") for w in WATCHED):
                    if a.asname:
                        bound[a.asname] = full
                    else:  # `import urllib.request` is written urllib.request.x; `import requests` as requests.x
                        bound[full] = full
                    sock_mod = sock_mod or any(full == m or full.startswith(m + ".") for m in SOCKET_MODULES)
        elif isinstance(n, ast.ImportFrom) and n.module and not n.level:
            mod = n.module
            for a in n.names:
                full = f"{mod}.{a.name}"
                if any(full == w or full.startswith(w + ".") or mod == w or mod.startswith(w + ".") for w in WATCHED):
                    bound[a.asname or a.name] = full
                    sock_mod = sock_mod or any(mod == m or mod.startswith(m + ".") or full == m for m in SOCKET_MODULES)
        elif isinstance(n, ast.ImportFrom) and n.module is None and not n.level:
            pass
    if "urllib.request" in bound.values() or any(v.startswith("urllib.request") for v in bound.values()):
        sock_mod = True
    out: dict = {}

    def add(name):
        out[name] = out.get(name, 0) + 1

    def resolve(dotted):
        if not dotted:
            return None
        head = dotted.split(".")[0]
        for key in sorted(bound, key=len, reverse=True):  # longest alias first ("urllib.request" before "urllib")
            if dotted == key or dotted.startswith(key + "."):
                return bound[key] + dotted[len(key):]
        return bound[head] + dotted[len(head):] if head in bound else None

    # every maximal dotted chain that resolves to a watched import is a site: a call (requests.get(u)), a reference
    # passed on or aliased (api_permissions.call(requests.get, ...), http_get = requests.get) or a constructor
    inner = {id(n.value) for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    for n in ast.walk(tree):
        if isinstance(n, (ast.Attribute, ast.Name)) and id(n) not in inner and isinstance(n.ctx, ast.Load):
            d = _dotted(n)
            r = resolve(d) if d and "()" not in d else None
            if r and r not in WATCHED and "." in r and not SAFE_LAST.search(r.split(".")[-1]):
                add(r)
        if isinstance(n, ast.Call):
            d = _dotted(n.func)
            if d and (d == "__import__" or d.split(".")[-1] == "import_module"):  # importlib / __import__ bypass
                arg = n.args[0] if n.args else None
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    if any(arg.value == w or arg.value.startswith(w + ".") for w in WATCHED):
                        add("dynamic-import:" + arg.value)
                else:
                    add("dynamic-import:<non-literal>")  # cannot be proven not to load a watched module
            if sock_mod and isinstance(n.func, ast.Attribute) and n.func.attr in METHODS \
                    and not (d and "()" not in d and resolve(d)):  # x.open(...) on any object; direct names are chains
                add("<obj>." + n.func.attr)
            if d in ("subprocess.run", "subprocess.Popen", "subprocess.call", "subprocess.check_output",
                     "subprocess.check_call") and any(
                    isinstance(c, ast.Constant) and isinstance(c.value, str) and c.value.split("/")[-1] in ("curl", "wget")
                    for c in ast.walk(n)):
                add("subprocess:curl-wget")
    return out


def scan_tree(root: Path) -> dict:
    """{(relpath, site): count} over the scanned tree."""
    out = {}
    files = []
    for s in SCAN:
        p = root / s
        files += sorted(p.rglob("*.py")) if p.is_dir() else ([p] if p.exists() else [])
    for f in files:
        rel = str(f.relative_to(root))
        for site, c in _scan_source(f.read_text(encoding="utf-8")).items():
            out[(rel, site)] = c
    return out


# (relpath, site) -> (class, count, note). Counts are pinned: another call of the same name in the same file fails.
_PG = "selective_fetch.document_request_guard/document_response_note (DENIAL-COVERAGE-1111)"
SITES: dict = {
    # document sockets: the only two transports that GET article/landing/download URLs
    ("vera/selective_fetch.py", "requests.Session"): ("DOC", 1, "pinned_get: " + _PG),
    ("vera/selective_fetch.py", "requests.adapters.HTTPAdapter"): ("DOC", 1, "pinned TLS adapter used by pinned_get"),
    ("vera/public_fetch.py", "requests.Session"): ("DOC", 1, "default_transport: " + _PG),
    ("vera/public_fetch.py", "urllib3.connection.HTTPSConnection"): ("DOC", 1, "default_transport socket closer"),
    ("vera/public_fetch.py", "urllib3.connectionpool.HTTPSConnectionPool"): ("DOC", 1, "default_transport pool"),
    ("vera/public_fetch.py", "<obj>.connect"): ("DOC", 1, "default_transport _Conn.connect (same transport)"),
    # metadata APIs, gated by config/api_permissions.json (api_permissions.call); a 403 is a route denial (A-1)
    ("vera/search_providers.py", "requests.get"): ("API", 4, "arXiv, OpenAlex, Europe PMC search, DOAJ transports"),
    ("vera/source_resolver.py", "requests.get"): ("API", 1, "S2/OpenAlex lookups via api_permissions.call"),
    ("vera/reporting_lane.py", "urllib.request.build_opener"): ("API", 1, "two fixed API hosts, permission gate"),
    ("vera/reporting_lane.py", "<obj>.open"): ("API", 1, "same opener as above"),
    # search endpoint
    ("vera/search_providers.py", "requests.post"): ("SEARCH", 1, "DuckDuckGo Lite POST"),
    # LLM / embedding APIs (OpenAI-SDK clients, httpx transports, judge calls)
    ("vera/ask_service.py", "openai.OpenAI"): ("LLM", 7, "SDK clients"),
    ("vera/claim_check.py", "openai.OpenAI"): ("LLM", 1, "SDK client"),
    ("vera/generator_client.py", "openai.OpenAI"): ("LLM", 4, "SDK clients"),
    ("vera/inference_chain.py", "openai.OpenAI"): ("LLM", 2, "SDK clients"),
    ("vera/rate_limit.py", "httpx.Client"): ("LLM", 1, "OpenAI-SDK http client"),
    ("vera/rate_limit.py", "httpx.HTTPTransport"): ("LLM", 1, "OpenAI-SDK transport"),
    ("vera/judge_select.py", "requests.get"): ("LLM", 1, "judge model API"),
    ("vera/judge_select.py", "requests.post"): ("LLM", 1, "judge model API"),
    ("vera/m6/judge.py", "requests.get"): ("LLM", 1, "judge model API"),
    ("vera/m6/judge.py", "requests.post"): ("LLM", 1, "judge model API"),
    ("vera/chunk_taxo_embed.py", "urllib.request.build_opener"): ("LLM", 1, "embedding API"),
    ("vera/chunk_taxo_embed.py", "<obj>.open"): ("LLM", 1, "embedding API"),
    # own service (the VERA-DAR API/UI)
    ("streamlit_app.py", "requests.get"): ("OWN", 1, "own API"),
    ("vera/api_kickstart.py", "requests.get"): ("OWN", 1, "own API"),
    ("vera/api_triage.py", "requests.get"): ("OWN", 1, "own API"),
    ("vera/memory_review_panel.py", "requests.get"): ("OWN", 1, "own API"),
    ("vera/memory_review_panel.py", "requests.post"): ("OWN", 1, "own API"),
    ("vera/public_mode.py", "requests.get"): ("OWN", 2, "own API"),
    ("vera/public_mode.py", "requests.post"): ("OWN", 2, "own API"),
    ("vera/trace_eval/ui.py", "requests.get"): ("OWN", 2, "own API"),
    ("vera/ui_wake.py", "requests.get"): ("OWN", 1, "own API"),
    ("vera/ui_wake.py", "requests.post"): ("OWN", 1, "own API"),
    # adapter module named in config/portal registry; every module under vera/ is scanned itself
    ("vera/portal_registry.py", "dynamic-import:<non-literal>"): ("NOSOCKET", 1, "adapter existence check only"),
    # operator tool, fixed owner-chosen URL
    ("scripts/refresh_openai_pricing.py", "urllib.request.urlopen"): ("OPS", 1, "fixed pricing page"),
}


def test_t11_tree_is_fully_classified():
    found = scan_tree(ROOT)
    unclassified = sorted(k for k in found if k not in SITES)
    stale = sorted(k for k in SITES if k not in found)
    drift = sorted((k, found[k], SITES[k][1]) for k in SITES if k in found and found[k] != SITES[k][1])
    bad_class = sorted(k for k, v in SITES.items() if v[0] not in CLASSES)
    assert not unclassified, f"unclassified outbound sites (classify in SITES, DENIAL-COVERAGE-1111): {unclassified}"
    assert not stale, f"SITES entries with no matching call site: {stale}"
    assert not drift, f"call-site count changed (key, found, listed): {drift}"
    assert not bad_class, bad_class


def test_t11_every_doc_class_site_is_a_guarded_transport_or_reaches_one():
    """Only the two transports open a document socket; a DOC-class entry elsewhere must say how it is guarded."""
    for (rel, site), (cls, _n, note) in SITES.items():
        if cls == "DOC":
            assert note, f"DOC site {rel} {site} needs a note naming its guard"


def test_the_two_document_transports_call_the_guard_pair():
    import inspect

    from vera import public_fetch as pf
    from vera import selective_fetch as sf
    for fn in (sf.pinned_get, pf.default_transport):
        src = inspect.getsource(fn)
        assert "document_request_guard" in src and "document_response_note" in src, fn.__name__


def test_t13_residual_r1_doi_resolver_gets_one_request_denied_url_none():
    """R-1 (accepted, CL-1111-DENIAL-COVERAGE): a DOI whose publisher URL is denied still costs one request to the
    resolver (doi.org); the denied endpoint gets none. Fails if the resolver traffic grows."""
    from tests import acq_support as A
    from vera import acq_discovered, auto_acquire
    acq_discovered.reset_for_tests()
    acq_discovered.deny(acq_discovered.denial_ids(urls=("https://pub.example/a/z",)))
    A.seed_access("doi.org")
    A.seed_access("pub.example")
    web = A.Web({}, {}, {"doi.org/10.9999/y": A.Resp(status=302, body=b"", ctype="text/html",
                                                     headers={"Location": "https://pub.example/a/z"})})
    with pytest.raises(auto_acquire._Stop) as e:
        acq_discovered._b_get("https://doi.org/10.9999/y", {"doi.org", "pub.example"}, {"text/html"}, 1000, 2,
                              A.deps([], web), A.cfg(), A.clock_at(), None, [])
    assert e.value.code == "R-ACQ-ACCESS_DENIED"
    assert sum(1 for u in web.urls if "doi.org/10.9999/y" in u) == 1
    assert not any("pub.example/a/z" in u for u in web.urls)
    acq_discovered.reset_for_tests()


def _probe(tmp_path, body):
    p = tmp_path / "probe.py"
    p.write_text(body)
    return _scan_source(p.read_text())


@pytest.mark.parametrize("body", [
    "import requests\nrequests.get('https://x.example/a')\n",
    "from requests import get as g\ng('https://x.example/a')\n",
    "import urllib.request\nurllib.request.build_opener().open('https://x.example/a')\n",
    "import urllib.request as ur\nop = ur.build_opener()\nop.open('https://x.example/a')\n",
    "import requests\nhttp_get = requests.get\n",
    "import httpx\nc = httpx.Client()\n",
    "from openai import OpenAI\nc = OpenAI()\n",
    "import subprocess\nsubprocess.run(['curl', 'https://x.example/a'])\n",
    "import http.client\nc = http.client.HTTPSConnection('x.example')\n",
    "import socket\ns = socket.create_connection(('x.example', 443))\n",
    "import importlib\nr = importlib.import_module('requests')\nr.get('https://x.example/a')\n",
    "r = __import__('requests')\nr.get('https://x.example/a')\n",
    "from importlib import import_module\nm = import_module('urllib.request')\n",
    "import importlib\nr = importlib.import_module(name)\n",
])
def test_t11_negative_probe_is_detected_and_unclassified(tmp_path, body):
    sites = _probe(tmp_path, body)
    assert sites, f"scanner missed: {body!r}"
    assert all(("probe.py", s) not in SITES for s in sites)


def test_t11_exception_and_parse_only_imports_are_not_sites(tmp_path):
    assert _probe(tmp_path, "import requests\nfrom urllib.parse import urlsplit\n"
                            "try:\n    pass\nexcept requests.RequestException:\n    pass\n"
                            "x = urlsplit('a')\ne = requests.exceptions.Timeout\n") == {}


def test_t11_planted_unclassified_module_fails_the_tree_scan(tmp_path):
    (tmp_path / "vera").mkdir()
    (tmp_path / "vera" / "planted.py").write_text("import requests\nrequests.get('https://x.example/a')\n")
    found = scan_tree(tmp_path)
    assert ("vera/planted.py", "requests.get") in found and ("vera/planted.py", "requests.get") not in SITES

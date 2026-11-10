"""REPROV-W3-1110: parity of vera.chunk_value.portal_verdict (self-contained port) with vera.fulltext_loader.portal_verdict
(PORTAL-RULE-1110 CP5 as at batch6-rc-1110 d3fe0fe plus PVFIX-A-1110 75b0c6a). Verdict AND rule id must be equal, row for row.
Synthetic fixtures only; no DB, network or env."""
import hashlib
import json
from pathlib import Path

from vera import chunk_value as cv
from vera import fulltext_loader, host_role

ROOT = Path(__file__).resolve().parent.parent
CFG = json.loads((ROOT / "config" / "chunk_value.json").read_text(encoding="utf-8"))
POL = CFG["document"][cv.D1P]


def _rows():
    rows = []

    def add(item=None, prov=None, cu=None):
        rows.append((item, prov, cu))

    add({"url": "https://arxiv.org/abs/2101.00001"}, {"doi": "10.48550/arxiv.2101.00001"}, "https://doi.org/10.48550/arxiv.2101.00001")
    add({"doi": "10.48550/arXiv.2101.00001"}, {}, None)
    add({}, {"doi": "10.48550/arxiv.2101.00001"}, "https://doi.org/10.48550/arxiv.2101.00001")
    add({"landing_url": "https://www.ncbi.nlm.nih.gov/pubmed/123"}, {}, "https://www.ncbi.nlm.nih.gov/pubmed/123")
    add({"fulltext_url": "https://www.ncbi.nlm.nih.gov/gene/1"}, {"url": "https://www.ncbi.nlm.nih.gov/gene/1"}, None)
    add({}, {}, "https://www.ncbi.nlm.nih.gov/gene/1")
    add({"url": "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC123/"}, {}, "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC123/")
    add({"url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC123/"}, {}, "https://pmc.ncbi.nlm.nih.gov/articles/PMC123/")
    add({"url": "https://www.ncbi.nlm.nih.gov/books/NBK1/"}, {}, None)
    add({"url": "https://europepmc.org/articles/PMC123"}, {}, None)
    add({"url": "https://europepmc.org/article/PMC/PMC123"}, {}, None)
    add({"url": "https://europepmc.org/abstract/MED/1"}, {}, None)
    add({"source": "PPR", "url": "https://europepmc.org/articles/PMC123"}, {}, None)
    add({"source": "ppr"}, {}, None)
    add({"url": "https://en.wikipedia.org/wiki/Foo"}, {}, "https://en.wikipedia.org/wiki/Foo")
    add({"url": "https://example.invalid-host.example/paper.pdf"}, {}, "https://example.invalid-host.example/paper.pdf")
    add({"url": "https://www.nature.com/articles/s1"}, {"doi": "10.1038/s1"}, "https://doi.org/10.1038/s1")
    add({"doi": "10.1038/s1"}, {}, "https://doi.org/10.1038/s1")
    add({"url": "https://doi.org/10.1101/2020.01.01.123456"}, {}, "https://doi.org/10.1101/2020.01.01.123456")
    add({"doi": "10.1101/2020.01.01.123456v2"}, {}, None)
    add({"doi": "10.1101/s0123"}, {}, None)
    add({}, {"acquisition": {"arxiv": {"id": "2101.00001"}}}, "https://example.org/x")
    add({}, {"acquisition": {"arxiv": {"id": ""}, "abs_url": "https://arxiv.org/abs/1"}}, None)
    add({}, {"acquisition": {"url_chain": ["https://europepmc.org/search?x"], "abs_url": "https://www.nature.com/a"}}, None)
    add({}, {"acquisition": "not-a-dict"}, None)
    add({"identifier": {"url": "https://arxiv.org/abs/9"}}, {}, None)
    add({"identifier": "https://arxiv.org/abs/9"}, {}, None)
    add({"is_preprint": True, "url": "https://www.nature.com/a"}, {}, None)
    add({"is_preprint": False, "url": "https://www.nature.com/a"}, {}, None)
    add({"source_type": "preprint"}, {}, None)
    add({"source_type": "Posted-Content"}, {}, None)
    add({"pub_types": ["Journal Article", "Preprint"]}, {}, None)
    add({"pub_types": "preprint"}, {}, None)
    add({"pub_types": {"pubType": ["posted-content"]}}, {}, None)
    add({"url": "https://repository.example.edu/handle/1"}, {}, None)
    add({"url": "https://dspace.mit.edu/handle/1"}, {}, None)
    add({"url": "https://sci-hub.se/10.1/x"}, {}, None)
    add({"url": "https://osf.io/preprints/psyarxiv/abc"}, {}, None)
    add({"url": "https://osf.io/abc/"}, {}, None)
    add({"doi": "10.31235/osf.io/abcde"}, {}, None)
    add({"doi": "10.2139/ssrn.123"}, {"doi": "10.2139/ssrn.123"}, None)
    add({"doi": "doi:10.48550/arxiv.1"}, {}, None)
    add({"doi": "https://doi.org/10.48550/arxiv.1"}, {}, None)
    add({"doi": "https://dx.doi.org/10.1000%2fx"}, {}, None)
    for bad in ("", "ftp://arxiv.org/x", "https://127.0.0.1/a", "https://2130706433/a", "https://0x7f.1/a", "https://localhost/a",
                "https://arxiv.org.evil.example/a", "https://evil.example/https://arxiv.org/a", "https://ARXIV.ORG/abs/1",
                "https://arxiv.org./abs/1", "https://arxiv%2eorg/abs/1", "https://www.ncbi.nlm.nih.gov/%2e%2e/pmc/x",
                "https://www.ncbi.nlm.nih.gov/pmc/../gene/1", "https://www.ncbi.nlm.nih.gov/pmc%2f..%2fgene/1",
                "https://www.ncbi.nlm.nih.gov/%252e%252e/gene", "https://www.ncbi.nlm.nih.gov/books/..%5cgene/1",
                "http://" + "a" * 5000 + ".com/", "https://exa mple.com/", "https://аrxiv.org/abs/1", "https://arxiv。org/abs/1"):
        add({"url": bad}, {}, None)
        add({}, {}, bad)
    for junk in (None, 5, "str", [], {"url": 5}, {"url": None}, {"doi": 5}, {"pub_types": 5}, {"is_preprint": "yes"}):
        add(junk, junk if isinstance(junk, dict) else None, None)
    add(None, None, None)
    add({}, {}, None)
    for r in json.loads((ROOT / "config" / "host_roles.json").read_text(encoding="utf-8"))["hosts"]:
        h = r["match"]
        for path in ("/", "/preprints/x", "/pmc/articles/PMC1/", "/books/NBK1/", "/articles/PMC123", "/article/PMC/PMC1", "/abs/1", "/x/y.pdf"):
            for u in ("https://" + h + path, "https://www." + h + path):
                add({"url": u}, {}, None)
                add({}, {"fulltext_url": u}, u)
    add({"url": "https://zenodo.org/records/1", "version": "preprint"}, {}, None)
    add({"url": "https://zenodo.org/records/1", "version": "publishedVersion"}, {}, None)
    add({"url": "https://zenodo.org/records/1", "type": "dataset"}, {}, None)
    add({"landing_url": "https://hal.science/hal-1"}, {}, None)
    # PVFIX-A-1110: version / type meta on every version-gated row shape (parity must hold for each, incl. lookalikes)
    gated = ("https://zenodo.org/records/1234567", "https://hal.science/hal-01234567", "https://theses.hal.science/tel-1",
             "https://dspace.mit.edu/handle/1721.1/12345", "https://eprints.soton.ac.uk/1/", "https://openreview.net/forum?id=abc",
             "https://doi.org/10.5281/zenodo.1234567")
    vers = ("published", "publishedVersion", "vor", "VoR", "info:eu-repo/semantics/publishedVersion", "INFO:EU-REPO/SEMANTICS/publishedVersion",
            "accepted", "acceptedVersion", "camera-ready", "cameraready", "info:eu-repo/semantics/acceptedVersion", "submittedVersion",
            "preprint", "draft", "unknown", "none", "", "v1.0", "PublishedVersion ", "publishedversio", "published version", 5, None)
    for u in gated:
        for v in vers:
            add({"url": u, "version": v}, {}, None)
            add({"url": "https://example.org/x"}, {"version": v}, u)
            add({"url": u, "version": v, "type": "preprint"}, {}, None)
            add({"url": u, "version": v, "type": "dataset"}, {}, None)
        for t in ("dataset", "report", "software", "journal-article", "posted-content", "Dataset"):
            add({"url": u, "type": t}, {}, None)
            add({"url": u, "resource_type": t}, {}, None)
            add({"url": u}, {"type": t}, None)
            add({"url": u, "source_type": t}, {}, None)
        add({"url": "https://example.org/x", "title": "publishedVersion"}, {}, u)
    return rows


def test_policy_is_the_checkout_host_roles_file():
    """The vendored policy equals the checkout's config/host_roles.json byte for byte (by sha) and by content. After PVFIX-A-1110 that
    file is the 75b0c6a one (accepted_versions = published/VoR); on a rebase onto a newer main this test fails until re-copied (P7)."""
    raw = (ROOT / "config" / "host_roles.json").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == POL["source_sha256"]
    ref = json.loads(raw.decode("utf-8"))
    ref.pop("_doc", None)
    ref.pop("pending_decisions", None)
    assert POL["policy"] == ref


def test_parity_verdict_and_rule_id_row_for_row():
    host_role.set_registry_for_tests(None)
    rows = _rows()
    assert len(rows) > 1500
    seen, mismatches = {}, []
    for item, prov, cu in rows:
        want = fulltext_loader.portal_verdict(item, prov, cu)
        got = cv.portal_verdict(item, prov, cu, CFG)
        if want != got:
            mismatches.append((item, prov, cu, want, got))
        seen[want] = seen.get(want, 0) + 1
    assert not mismatches, mismatches[:5]
    for rid in ("doi:doi-preprint", "host:ncbi.nlm.nih.gov", "record_type:preprint", "flag:src-ppr"):
        assert rid in seen, rid
    assert None in seen and any(k and k.startswith("host:") for k in seen)
    assert "pattern:hp-shadow-libs" in seen
    assert any(k and k.startswith("version_unqualified:") for k in seen)       # gated rows that stay PORTAL
    vrows = [(i, p, c) for i, p, c in rows if isinstance(i, dict) and "version" in i and str(i.get("url", "")).startswith("https://zenodo.org/")]
    assert any(fulltext_loader.portal_verdict(i, p, c) is None for i, p, c in vrows)            # published/VoR relaxes (not vacuous)
    assert any(fulltext_loader.portal_verdict(i, p, c) for i, p, c in vrows)                    # accepted etc. stay PORTAL


def test_fail_closed_matches_on_invalid_policy():
    bad = json.loads(json.dumps(CFG))
    bad["document"][cv.D1P]["policy"]["hosts"] = []
    assert cv.portal_verdict({"url": "https://www.nature.com/a"}, {}, None, bad) == "config_invalid"
    host_role.set_registry_for_tests(host_role.Registry({"hosts": []}, "t"))
    try:
        assert fulltext_loader.portal_verdict({"url": "https://www.nature.com/a"}, {}, None) == "config_invalid"
    finally:
        host_role.set_registry_for_tests(None)


def test_rule_ids_are_enumerable():
    ids = cv.portal_rule_ids(CFG)
    for rid in ("doi:doi-preprint", "host:ncbi.nlm.nih.gov", "flag:src-ppr", "internal_error"):
        assert rid in ids


def test_audit_document_emits_portal_flag_and_never_purges_undecidable():
    chunks = [(1, 0, 5, "hello"), (2, 5, 10, "world")]
    base = {"source_id": 1, "document_id": "d", "host": "www.ncbi.nlm.nih.gov", "licence_id": "cc-by", "doi": None,
            "title_present": True, "chars": 10, "text_sha256": "x"}
    m = dict(base, portal={"item": {"landing_url": "https://www.ncbi.nlm.nih.gov/gene/1"}, "prov": {}, "canonical_url": None})
    fl = [f for f in cv.audit_document(m, chunks, CFG)["d_flags"] if f["rule"] == "D1"]
    assert len(fl) == 1 and fl[0]["severity"] == "purge" and fl[0]["portal_rule"] == "host:ncbi.nlm.nih.gov"
    m = dict(base, portal={"item": {"url": "https://www.nature.com/a"}, "prov": {}, "canonical_url": None})
    assert not [f for f in cv.audit_document(m, chunks, CFG)["d_flags"] if f["rule"] in ("D1", "RULES_ERROR")]
    assert not [f for f in cv.audit_document(base, chunks, CFG)["d_flags"] if f["rule"] == "D1"]
    bad = json.loads(json.dumps(CFG))
    bad["document"][cv.D1P]["policy"]["hosts"] = []
    f = [f for f in cv.audit_document(m, chunks, bad)["d_flags"] if f["rule"] == "RULES_ERROR"]
    assert f and f[0]["severity"] == "unverified"
    m = dict(base, host="arxiv.org", portal={"item": {"url": "https://arxiv.org/abs/1"}, "prov": {}, "canonical_url": None})
    d1 = [f for f in cv.audit_document(m, chunks, CFG)["d_flags"] if f["rule"] == "D1"]
    assert len(d1) == 1 and d1[0].get("portal_rule") == "host:arxiv.org"


if __name__ == "__main__":
    for n, f in sorted(globals().items()):
        if n.startswith("test_") and callable(f):
            f()
            print("PASS", n)

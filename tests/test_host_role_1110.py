"""PORTAL-RULE-1110 WP1: host-role policy classification (offline, pure)."""
import json
import tempfile
import unittest
from pathlib import Path

from vera import host_role as hr

O, R, P, RES, X, U = "ORIGINAL", "REPOSITORY_OF_RECORD", "PORTAL", "RESOLVER", "EXCLUDED", "UNKNOWN"


def role(u, m=None):
    return hr.role_for_url(u, m).role


class HostRoleTests(unittest.TestCase):
    def setUp(self):
        hr.set_registry_for_tests(None)

    def tearDown(self):
        hr.set_registry_for_tests(None)

    def test_config_valid_and_hashed(self):
        self.assertIsNone(hr.config_error())
        self.assertEqual(len(hr.config_hash()), 64)

    def test_inventory_hosts(self):
        cases = {
            P: ["arxiv.org", "export.arxiv.org", "ar5iv.org", "arxiv-vanity.com", "alphaxiv.org", "xxx.lanl.gov", "biorxiv.org",
                "medrxiv.org", "papers.ssrn.com", "ssrn.com", "chemrxiv.org", "researchsquare.com", "preprints.org", "psyarxiv.com",
                "api.openalex.org", "openalex.org", "inspirehep.net", "api.crossref.org", "doaj.org", "api.openaire.eu",
                "api.unpaywall.org", "api.semanticscholar.org", "semanticscholar.org", "pubmed.ncbi.nlm.nih.gov", "core.ac.uk",
                "base-search.net", "ui.adsabs.harvard.edu", "science.gov", "data.gov", "catalog.data.gov", "usa.gov",
                "lite.duckduckgo.com", "scholar.google.com", "hn.algolia.com", "api.gdeltproject.org", "en.wikipedia.org",
                "de.wikipedia.org", "www.wikidata.org", "en.wikinews.org", "web.archive.org", "archive.ph", "archive.today",
                "webcitation.org", "researchgate.net", "www.academia.edu", "news.google.com", "news.yahoo.com", "msn.com",
                "www.britannica.com", "scholarpedia.org"],
            RES: ["doi.org", "dx.doi.org", "hdl.handle.net", "identifiers.org"],
            X: ["openstax.org", "www.oapen.org", "cds.cern.ch", "libretexts.org", "chem.libretexts.org", "sci-hub.se", "libgen.is"],
            R: ["pmc.ncbi.nlm.nih.gov", "osti.gov", "www.osti.gov", "ntrs.nasa.gov", "repo.scoap3.org"],
            O: ["dl.acm.org", "ieeexplore.ieee.org", "link.springer.com", "www.sciencedirect.com", "onlinelibrary.wiley.com",
                "www.nature.com", "www.mdpi.com", "aclanthology.org", "journals.plos.org", "api.plos.org", "govinfo.gov",
                "github.com", "docs.aws.amazon.com", "learn.microsoft.com", "kubernetes.io", "www.census.gov", "www.fec.gov",
                "www.eac.gov", "www.cdc.gov", "nytimes.com", "www.reuters.com", "apnews.com", "plato.stanford.edu",
                "journals.aps.org"],
        }
        for expect, hosts in cases.items():
            for h in hosts:
                with self.subTest(host=h):
                    self.assertEqual(role("https://" + h + "/x"), expect)

    def test_arxiv_mirrors_and_subdomains(self):
        for u in ["https://browse.arxiv.org/pdf/1", "https://ar5iv.labs.arxiv.org/x", "https://www.arxiv.org/abs/1",
                  "https://ARXIV.ORG./abs/1", "https://huggingface.co/papers/2401.1", "https://osf.io/preprints/psyarxiv/abc"]:
            with self.subTest(u=u):
                self.assertEqual(role(u), P)
        self.assertEqual(role("https://huggingface.co/models"), U)
        self.assertEqual(role("https://osf.io/abc/"), U)

    def test_suffix_confusion(self):
        for u in ["https://notarxiv.org/", "https://arxiv.org.evil.com/", "https://evilwikipedia.org/", "https://xdoi.org/"]:
            with self.subTest(u=u):
                self.assertEqual(role(u), U)

    def test_paths_europepmc_ncbi(self):
        b = "https://www.ebi.ac.uk/europepmc/webservices/rest/"
        self.assertEqual(role(b + "PMC123456/fullTextXML"), R)
        self.assertEqual(role(b + "search?query=x"), P)
        self.assertEqual(role(b + "MED/1/references"), P)
        self.assertEqual(role("https://europepmc.org/article/MED/123"), P)
        self.assertEqual(role("https://europepmc.org/article/PMC/PMC123"), R)
        self.assertEqual(role("https://europepmc.org/articles/PMC123"), R)
        self.assertEqual(role("https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1/"), R)
        self.assertEqual(role("https://www.ncbi.nlm.nih.gov/books/NBK1/"), R)
        self.assertEqual(role("https://www.ncbi.nlm.nih.gov/pubmed/1"), P)
        self.assertEqual(role("https://www.ncbi.nlm.nih.gov/gene/1"), O)

    def test_federal_index_and_regulations(self):
        self.assertEqual(role("https://eric.ed.gov/?q=x"), P)
        self.assertEqual(role("https://eric.ed.gov/?id=ED123456"), R)
        self.assertEqual(role("https://www.regulations.gov/document/X"), O)
        self.assertEqual(role("https://www.regulations.gov/comment/X-1"), P)

    def test_archive_org_paths(self):
        self.assertEqual(role("https://archive.org/web/2020/http://x"), P)
        self.assertEqual(role("https://archive.org/details/foo"), U)

    def test_preprint_dois(self):
        for d in ["10.48550/arXiv.2401.1", "10.1101/2020.01.01.123456", "10.1101/123456v2", "10.2139/ssrn.123",
                  "10.21203/rs.3.rs-1/v1", "10.20944/preprints202001.0001.v1", "10.31219/osf.io/abc",
                  "10.31234/osf.io/abc", "10.26434/chemrxiv-2020-x", "https://doi.org/10.48550/arXiv.1", "doi:10.2139/ssrn.1"]:
            with self.subTest(d=d):
                self.assertEqual(hr.role_for_record({"doi": d}).role, P)
        self.assertEqual(role("https://doi.org/10.48550/arXiv.2401.1"), P)
        self.assertEqual(role("https://doi.org/10.1038/nature123"), RES)
        self.assertEqual(role("https://doi.org/10.1101/gad.123456"), RES)  # CSHL journal DOI, not a preprint
        self.assertEqual(hr.role_for_record({"doi": "10.1038/x", "url": "https://www.nature.com/a"}).role, O)

    def test_pmc_vs_ppr(self):
        pmc = {"source": "PMC", "pmcid": "PMC1", "url": "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC1/fullTextXML"}
        ppr = dict(pmc, source="PPR")
        self.assertEqual(hr.role_for_record(pmc).role, R)
        self.assertEqual(hr.role_for_record(ppr).role, P)
        self.assertEqual(hr.role_for_record(dict(pmc, source="ppr")).role, P)
        self.assertEqual(hr.role_for_record(dict(pmc, oa={"source": "PPR"})).role, P)
        self.assertEqual(hr.role_for_record(dict(pmc, pub_types=["Preprint"])).role, P)
        self.assertEqual(hr.role_for_record(dict(pmc, type="posted-content")).role, P)
        self.assertEqual(hr.role_for_record(dict(pmc, is_preprint=True)).role, P)

    def test_repository_version_gate(self):
        z = "https://zenodo.org/records/1"
        self.assertEqual(role(z), P)  # unknown version fails closed
        self.assertEqual(role(z, {"version": "unknown"}), P)
        self.assertEqual(role(z, {"version": "submittedVersion"}), P)
        self.assertEqual(role(z, {"type": "preprint"}), P)
        self.assertEqual(role(z, {"version": "publishedVersion"}), R)
        self.assertEqual(role(z, {"version": "acceptedVersion"}), R)
        self.assertEqual(role(z, {"type": "dataset"}), R)
        self.assertEqual(role(z, {"type": "report"}), R)
        self.assertEqual(role("https://hal.science/hal-1", {"version": "published"}), R)
        self.assertEqual(role("https://dspace.mit.edu/handle/1", {"version": "published"}), R)
        self.assertEqual(role("https://dspace.mit.edu/handle/1"), P)
        self.assertEqual(role("https://openreview.net/forum?id=x"), P)
        self.assertEqual(role("https://openreview.net/forum?id=x", {"version": "published"}), O)

    def test_encyclopedias_and_pending(self):
        self.assertEqual(role("https://plato.stanford.edu/entries/x/"), O)
        self.assertEqual(role("https://www.scholarpedia.org/article/X"), P)
        d = json.loads(Path(hr.DEFAULT_PATH).read_text())
        self.assertEqual({x["id"] for x in d["pending_decisions"]}, {"D-a", "D-b", "D-c"})
        self.assertTrue(all("pending" in x["status"] for x in d["pending_decisions"]))
        self.assertEqual(role("https://repo.scoap3.org/records/1"), R)

    def test_unknown_host_per_use(self):
        u = "https://example.org/page"
        self.assertEqual(role(u), U)
        self.assertFalse(hr.is_evidence_allowed(u, "store"))
        self.assertTrue(hr.is_evidence_allowed(u, "answer"))
        self.assertTrue(hr.is_hop_allowed(u))

    def test_evidence_decisions(self):
        for use in ("store", "answer"):
            self.assertTrue(hr.is_evidence_allowed("https://www.nature.com/a", use))
            self.assertTrue(hr.is_evidence_allowed("https://pmc.ncbi.nlm.nih.gov/articles/PMC1/", use))
            for bad in ("https://arxiv.org/abs/1", "https://en.wikipedia.org/wiki/X", "https://doi.org/10.1038/x", "https://openstax.org/"):
                self.assertFalse(hr.is_evidence_allowed(bad, use), bad)
            self.assertFalse(hr.is_evidence_allowed({"doi": "10.1101/2020.01.01.123456", "url": "https://www.biorxiv.org/x"}, use))
            self.assertFalse(hr.is_evidence_allowed("https://www.nature.com/a", "bogus") and use == "x")
        self.assertFalse(hr.is_evidence_allowed("https://www.nature.com/a", "bogus"))
        self.assertTrue(hr.is_hop_allowed("https://doi.org/10.1038/x"))
        self.assertFalse(hr.is_hop_allowed("https://ar5iv.org/x"))
        self.assertFalse(hr.is_hop_allowed("https://archive.ph/abc"))

    def test_malformed_never_raises_and_denies(self):
        bad = [None, "", "   ", 5, [], {}, "http://", "http://[bad", "ftp://arxiv.org/x", "javascript:alert(1)", "http://a b.com/",
               "http://127.0.0.1/", "http://[::1]/", "https://arxiv.org:99999/", "http://exa..mple.com", "\x00", "http://%61rxiv.org/",
               "https://" + "a" * 5000, "http://xn--/", "//", "https:///path"]
        for u in bad:
            with self.subTest(u=repr(u)[:40]):
                for use in ("store", "answer"):
                    self.assertFalse(hr.is_evidence_allowed(u, use))
                hr.role_for_url(u)
                hr.is_hop_allowed(u)
        self.assertFalse(hr.is_evidence_allowed(None, "answer"))
        self.assertEqual(hr.role_for_record(None).role, U)
        self.assertEqual(hr.role_for_record({}).role, U)
        self.assertFalse(hr.is_evidence_allowed({}, "answer"))

    def test_bare_host_and_userinfo(self):
        self.assertEqual(role("arxiv.org/abs/1"), P)
        self.assertEqual(role("https://user:pw@arxiv.org/abs/1"), P)
        self.assertEqual(role("https://arxiv.org:443/abs/1"), P)
        self.assertEqual(role("https://example.com@arxiv.org/"), P)

    def _reg(self, text):
        with tempfile.TemporaryDirectory(dir=str(Path(hr.__file__).resolve().parent.parent / "scratch")) if False else tempfile.TemporaryDirectory() as td:
            p = Path(td) / "h.json"
            p.write_text(text)
            return hr.load_registry(p)

    def test_invalid_config_fails_closed(self):
        row = {"match": "a.com", "role": "ORIGINAL", "kind": "x"}
        for text in ["", "{", "[]", json.dumps({"hosts": []}), json.dumps({"hosts": [dict(row, role="BOGUS")]}),
                     json.dumps({"hosts": [dict(row, kind="")]}), json.dumps({"hosts": [dict(row, path_regex="(")]}),
                     json.dumps({"hosts": [row, row]}), json.dumps({"hosts": [row], "doi_prefix_roles": [{"role": "PORTAL", "regex": "("}]}),
                     json.dumps({"hosts": [row], "use_defaults": {"unknown_host": {"store": "maybe"}}})]:
            with self.subTest(text=text[:40]):
                reg = self._reg(text)
                self.assertTrue(reg.error)
                hr.set_registry_for_tests(reg)
                self.assertEqual(hr.role_for_url("https://a.com/"), ("PORTAL", "config_invalid"))
                for use in ("store", "answer"):
                    self.assertFalse(hr.is_evidence_allowed("https://a.com/", use))
                    self.assertFalse(hr.is_evidence_allowed("https://unknown.org/", use))
        hr.set_registry_for_tests(hr.load_registry("/nonexistent/host_roles.json"))
        self.assertTrue(hr.config_error())
        self.assertEqual(role("https://nature.com/"), P)

    def test_precedence_specificity(self):
        cfg = {"hosts": [
            {"match": "a.com", "role": "ORIGINAL", "kind": "x"},
            {"match": "sub.a.com", "role": "PORTAL", "kind": "x"},
            {"match": "b.com", "role": "ORIGINAL", "kind": "x", "path_regex": "^/p"},
            {"match": "b.com", "role": "PORTAL", "kind": "x", "path_regex": "^/p/q"},
            {"match": "c.com", "role": "ORIGINAL", "kind": "x"},
            {"match": "c.com", "role": "PORTAL", "kind": "x", "path_regex": "^/z"},
        ]}
        hr.set_registry_for_tests(hr.Registry(cfg, "t"))
        self.assertEqual(role("https://sub.a.com/"), P)
        self.assertEqual(role("https://x.sub.a.com/"), P)
        self.assertEqual(role("https://a.com/"), O)
        self.assertEqual(role("https://b.com/p/q"), P)  # equal specificity: PORTAL beats ORIGINAL
        self.assertEqual(role("https://b.com/p/r"), O)
        self.assertEqual(role("https://c.com/z"), P)
        self.assertEqual(role("https://c.com/y"), O)

    def test_pinned_owner_portals_never_evidence(self):
        for h in ["arxiv.org", "en.wikipedia.org", "api.openalex.org", "inspirehep.net", "lite.duckduckgo.com", "biorxiv.org"]:
            self.assertFalse(hr.is_evidence_allowed(f"https://{h}/x", "store"), h)
            self.assertFalse(hr.is_evidence_allowed(f"https://{h}/x", "answer"), h)


if __name__ == "__main__":
    unittest.main()

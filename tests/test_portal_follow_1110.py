"""PORTAL-WIRE-B-1110: vera/portal_follow.py follows promising portal records to the original host (offline; the
metadata lookups are fakes, nothing is fetched; host roles come from the shipped config/host_roles.json)."""
import pytest

from vera import arxiv_portal, portal_follow as pf

Q = "quark confinement lattice"
PMC = "https://pmc.ncbi.nlm.nih.gov/articles/PMC123/"
PUB = "https://link.springer.com/article/10.1007/q1"


def cfg(**kw):
    return {**pf.DEFAULTS, "enabled": True, **kw}


class Fake:
    """Recording metadata lookups: arxiv[aid] -> (own, refs); works[filter] -> {"doi", "locations", "refs"}."""

    def __init__(self, arxiv=None, works=None, openalex=True):
        self.arxiv_map, self.works, self.calls = dict(arxiv or {}), dict(works or {}), []
        self.lookups = pf.Lookups(self.arxiv, self.openalex if openalex else None)

    def arxiv(self, aid, counts, deadline, max_refs):
        self.calls.append(("arxiv", aid))
        counts["arxivportal_calls"] = counts.get("arxivportal_calls", 0) + 1  # one INSPIRE request (as the real lookup counts)
        got = self.arxiv_map.get(aid)
        return (got[0], list(got[1])[:max_refs]) if got else None, "inspire"

    def openalex(self, flt, counts, deadline, max_refs):
        self.calls.append(("openalex", flt))
        w = self.works.get(flt)
        counts["oa_calls"] = counts.get("oa_calls", 0) + 1  # as openalex_work: 1 request, +1 for a references batch
        if w is not None and max_refs and w.get("refs"):
            counts["oa_calls"] += 1
        return None if w is None else {**w, "refs": list(w.get("refs") or ())[:max_refs]}


def rec(**kw):
    return {"title": "Quark confinement on the lattice", "snippet": "an abstract about quark confinement",
            "provider": "openalex", **kw}


def run(records, fake, **kw):
    return pf.follow(Q, records, cfg(**kw), lookups=fake.lookups, clock=lambda: 0.0)


def test_portal_record_is_followed_through_doi_and_locations_to_the_original_host():
    fake = Fake(works={"doi:https://doi.org/10.1007/q1": {"doi": "10.1007/q1", "locations": [
        "https://arxiv.org/abs/2101.00001", PUB, PMC], "refs": []}})
    out, counts = run([rec(doi="10.1007/q1", url="https://doi.org/10.1007/q1")], fake)
    assert [(c["url"], c["host_role"]) for c in out] == [(PUB, "ORIGINAL"), (PMC, "REPOSITORY_OF_RECORD")]
    c = out[0]
    assert c["acq_source"] == "portal_follow" and c["portal_kind"] == "original" and c["snippet"] == ""
    assert c["title"] == "Quark confinement on the lattice"  # metadata only (the abstract is never carried)
    assert c["portal_via"].startswith("via openalex record") and c["portal_hops"] == 1
    assert ("arxiv", "2101.00001") in fake.calls  # the arXiv location is a portal: followed by identifier, not read


def test_without_an_openalex_lookup_a_non_portal_doi_is_handed_off_as_an_identifier():
    fake = Fake(openalex=False)
    out, _ = run([rec(doi="10.1007/q1", url="https://en.wikipedia.org/wiki/Quark")], fake)
    assert [(c["doi"], c["url"], c["host_role"]) for c in out] == [
        ("10.1007/q1", "https://doi.org/10.1007/q1", "RESOLVER")]  # acquisition resolves it under its own hop gates


def test_portal_to_portal_chain_reaches_the_published_version_and_references():
    # Wikipedia link -> DOI in the link (arXiv DOI, a portal) -> arXiv id -> INSPIRE -> journal DOI + reference DOIs
    fake = Fake(arxiv={"2101.00001": (["10.1103/pub"], ["10.1103/ref.a", "10.1103/ref.b"])}, openalex=False)
    r = rec(provider="duckduckgo_lite", url="https://en.wikipedia.org/wiki/Quark?ref=https://doi.org/10.48550/arXiv.2101.00001")
    out, counts = run([r], fake)
    assert [(c["doi"], c["portal_kind"]) for c in out] == [
        ("10.1103/pub", "original"), ("10.1103/ref.a", "reference"), ("10.1103/ref.b", "reference")]
    assert all(c["portal_hops"] <= 4 for c in out) and counts["portal_hop"] == 1
    assert "arXiv 2101.00001" in out[0]["portal_via"]


def test_hop_cap_stops_the_walk():
    fake = Fake(arxiv={"2101.00001": (["10.1103/pub"], [])}, openalex=False)
    r = rec(provider="duckduckgo_lite", url="https://en.wikipedia.org/wiki/Q?x=https://doi.org/10.48550/arXiv.2101.00001")
    out, counts = run([r], fake, max_hops_per_record=2)  # url(0) -> doi(1) -> arxiv(2) -> pub doi(3): over the cap
    assert out == [] and counts["hop_cap"] >= 1
    out, _ = run([r], Fake(arxiv={"2101.00001": (["10.1103/pub"], [])}, openalex=False), max_hops_per_record=3)
    assert [c["doi"] for c in out] == ["10.1103/pub"]


def test_cycles_are_detected_and_nothing_is_handed_off_twice():
    # the published version's OpenAlex locations point back to the arXiv e-print, whose lookup points to the same DOI
    fake = Fake(arxiv={"2101.00001": (["10.1103/pub"], [])},
                works={"doi:https://doi.org/10.1103/pub": {"doi": "10.1103/pub", "locations": [
                    "https://arxiv.org/abs/2101.00001", "https://doi.org/10.1103/pub", PUB], "refs": []}})
    out, counts = run([rec(doi="10.48550/arXiv.2101.00001", provider="arxiv")], fake)
    assert [c["url"] for c in out] == [PUB] and counts["cycle"] >= 2
    assert fake.calls.count(("arxiv", "2101.00001")) == 1


def test_dead_ends_excluded_unknown_preprint_and_portal_without_identifier():
    fake = Fake(works={"doi:https://doi.org/10.1000/x": {"doi": "10.1000/x", "locations": [
        "https://openstax.org/books/x", "https://unknown.example/paper", "https://www.researchgate.net/publication/1"],
        "refs": []}})
    out, counts = run([rec(doi="10.1000/x"), rec(doi="10.1101/2020.01.01.123456", title="Quark preprint")], fake)
    assert out == []
    assert counts["dead_excluded"] == 1 and counts["dead_unknown"] == 1 and counts["dead_portal"] == 1
    assert counts["dead_preprint_doi"] == 1  # non-arXiv preprint DOI: no route yet (plan D-b)


def test_references_of_top_records_are_followed_once_not_recursively():
    fake = Fake(works={
        "doi:https://doi.org/10.1000/top": {"doi": "10.1000/top", "locations": [PUB], "refs": ["10.1000/r1", "10.1000/r2"]},
        "doi:https://doi.org/10.1000/r1": {"doi": "10.1000/r1", "locations": [PMC], "refs": ["10.1000/deeper"]}})
    out, _ = run([rec(doi="10.1000/top")], fake)
    assert sorted((c["url"], c["portal_kind"]) for c in out) == sorted([
        (PUB, "original"), (PMC, "reference"), ("https://doi.org/10.1000/r2", "reference")])
    assert ("openalex", "doi:https://doi.org/10.1000/deeper") not in fake.calls  # references of references: no


def test_budgets_records_lookups_handoffs_and_deadline():
    recs = [rec(doi=f"10.1000/q{i}", title=f"Quark paper {i}") for i in range(6)] + [rec(doi="10.1000/off", title="Cats", snippet="")]
    fake = Fake(openalex=False)
    out, counts = run(recs, fake, max_records_per_question=2)
    assert counts["records"] == 2 and [c["doi"] for c in out] == ["10.1000/q0", "10.1000/q1"]
    assert "10.1000/off" not in [c["doi"] for c in pf.follow(Q, recs, cfg(max_records_per_question=10),
                                                          lookups=fake.lookups, clock=lambda: 0.0)[0]]  # not relevant
    fake = Fake(works={f"doi:https://doi.org/10.1000/q{i}": {"doi": f"10.1000/q{i}", "locations": [], "refs": []}
                       for i in range(3)})
    _, counts = run(recs[:3], fake, max_lookups_per_question=3)  # HTTP-call cap: each lookup reserves its worst case (2 with references): 2 fit
    assert len([c for c in fake.calls if c[0] == "openalex"]) == 2 and counts["lookup_cap"] == 1
    fake = Fake(works={"doi:https://doi.org/10.1000/top": {"doi": "10.1000/top", "locations": [],
                                                       "refs": [f"10.1000/r{i}" for i in range(9)]}}, openalex=True)
    fake.lookups.openalex = (lambda f, c, d, m, _o=fake.openalex: _o(f, c, d, m) if f.endswith("/top") else None)
    out, counts = run([rec(doi="10.1000/top")], fake, max_handoffs_per_record=3)
    assert len(out) == 3 and counts["handoff_cap"] == 1
    t = iter([0.0, 0.0, 100.0, 100.0, 100.0, 100.0])
    out, counts = pf.follow(Q, recs[:3], cfg(), lookups=Fake(openalex=False).lookups, clock=lambda: next(t, 100.0))
    assert counts["deadline"] >= 1 and len(out) < 3


def test_already_candidates_and_non_portal_records_are_skipped():
    fake = Fake(openalex=False)
    out, counts = pf.follow(Q, [rec(doi="10.1000/q")], cfg(), lookups=fake.lookups, clock=lambda: 0.0, have={"10.1000/q"})
    assert out == [] and counts["already_candidate"] == 1
    stored = rec(doi="10.1000/s", dar_span={"start": 0, "end": 1})
    own = rec(doi="10.1000/o", provider="europepmc", url=PMC)
    assert pf.select(Q, [stored, own], cfg()) == []


def test_disabled_or_invalid_config_does_nothing(tmp_path):
    bad = tmp_path / "pf.json"
    bad.write_text("{not json")
    assert pf.load_config(bad)["enabled"] is False
    assert pf.follow(Q, [rec(doi="10.1000/q")], {**pf.DEFAULTS, "enabled": False}) == ([], {})
    shipped = pf.load_config()
    assert shipped["enabled"] is True and shipped["max_hops_per_record"] == 4  # owner default hop cap


def test_arxiv_portal_extend_hands_follow_ups_to_acquisition(monkeypatch):
    seen = {}

    def fake_extend(question, cands, acq, **kw):
        seen["cands"] = list(cands)
        return list(cands) + [{"doi": "10.1000/followed", "acq_source": "portal_follow"}]
    monkeypatch.setattr(pf, "extend", fake_extend)
    monkeypatch.setattr(arxiv_portal, "switches", lambda acq=None: None)  # arXiv lookups off
    base = [rec(doi="10.1000/q")]
    out = arxiv_portal.extend(Q, base, {})
    assert out[-1]["acq_source"] == "portal_follow" and seen["cands"] == base


def test_extend_never_follows_arxiv_seeds_twice(monkeypatch):
    fake = Fake(arxiv={"2101.00001": (["10.1103/pub"], [])}, openalex=False)
    seed = {"url": "https://arxiv.org/abs/2101.00001", "title": "Quark seed", "provider": "arxiv", "arxiv_id": "2101.00001"}
    other = rec(doi="10.1000/q")
    out = pf.extend(Q, [seed, other], {}, cfg=cfg(), lookups=fake.lookups, clock=lambda: 0.0)
    assert ("arxiv", "2101.00001") not in fake.calls and out[:2] == [seed, other]


@pytest.mark.parametrize("url,ids", [
    ("https://arxiv.org/abs/2101.00001v2", [("arxiv", "2101.00001")]),
    ("https://doi.org/10.1103/PhysRev.1", [("doi", "10.1103/physrev.1")]),
    ("https://openalex.org/W123", [("openalex", "W123")]),
    ("https://en.wikipedia.org/wiki/Quark", []),
])
def test_ids_in_link(url, ids):
    assert pf.ids_in_link(url) == ids


def test_lookup_cap_counts_http_calls_not_expander_calls():
    """Review F2: three records whose works have references made 7 HTTP calls under the old expander-call cap of 6."""
    recs = [rec(doi=f"10.1000/q{i}", title=f"Quark paper {i}") for i in range(3)]
    for cap in range(1, 9):
        fake = Fake(works={f"doi:https://doi.org/10.1000/q{i}": {"doi": f"10.1000/q{i}", "locations": [],
                                                                "refs": [f"10.1000/r{i}"]} for i in range(3)})
        out, counts = run(recs, fake, max_lookups_per_question=cap)
        assert pf.http_calls(counts) <= cap, (cap, counts)  # actual requests (fake counts like the real lookups)
    fake = Fake(works={f"doi:https://doi.org/10.1000/q{i}": {"doi": f"10.1000/q{i}", "locations": [],
                                                            "refs": [f"10.1000/r{i}"]} for i in range(3)})
    out, counts = run(recs, fake)  # shipped cap 6
    assert pf.http_calls(counts) <= 6 and counts["lookup_cap"] >= 1
    assert {c["doi"] for c in out} >= {"10.1000/q2"}  # records past the cap are still handed off as identifiers


def test_default_lookups_cost_models_inspire_plus_openalex(monkeypatch):
    sw = {"insp": True, "oa_on": True, "pc": {}, "rcfg": {"caps": {"timeout_s": 5}}}
    monkeypatch.setattr(arxiv_portal, "switches", lambda acq=None: sw)
    lk = pf.default_lookups()
    assert lk.cost("arxiv", True) == 3 and lk.cost("arxiv", False) == 2 and lk.cost("openalex", True) == 2
    sw.update(oa_on=False)
    assert pf.default_lookups().cost("arxiv", True) == 1

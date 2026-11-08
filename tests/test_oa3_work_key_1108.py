"""OA-3a (OA37-1108): canonical work key (vera/work_key.py). OFFLINE, synthetic identifiers only."""
import pytest

from vera import work_key as W

DOI = "10.9999/oa3.work"
PMC = "PMC7300011"


@pytest.mark.parametrize("rec,want", [
    ({"doi": "https://doi.org/10.9999/OA3.Work"}, f"doi:{DOI}"),
    ({"doi": f"doi:{DOI}", "pmcid": PMC, "url": "https://x.org/a"}, f"doi:{DOI}"),
    ({"pmcid": "pmc7300011"}, f"pmcid:{PMC}"),
    ({"oa": {"pmcid": PMC}}, f"pmcid:{PMC}"),
    ({"pmid": "12345678"}, "pmid:12345678"),
    ({"pmid": "not-a-pmid"}, None),
    ({"arxiv": {"id": "2605.23989", "version": "v2"}}, "arxiv:2605.23989"),
    ({"identifier": "arXiv:2605.23989v1"}, "arxiv:2605.23989"),
    ({"doi": "10.48550/arXiv.2605.23989"}, "arxiv:2605.23989"),  # DataCite arXiv DOI is the arXiv id
    ({"url": "https://arxiv.org/abs/2605.23989v3"}, "arxiv:2605.23989"),
    ({"url": f"https://www.ncbi.nlm.nih.gov/pmc/articles/{PMC}/"}, f"pmcid:{PMC}"),
    ({"url": f"https://doi.org/{DOI}"}, f"doi:{DOI}"),
    ({"url": "https://Example.org/paper/?utm_source=x"}, "url:https://example.org/paper"),
    ({"url": "https://hdl.handle.net/1234/5678"}, "url:https://hdl.handle.net/1234/5678"),  # no DOI invented
    ({}, None),
    (None, None),
    ("https://doi.org/10.9999/x", None),
])
def test_work_key_priority_and_normalisation(rec, want):
    assert W.work_key(rec) == want


def test_keys_highest_priority_first():
    rec = {"url": f"https://www.ncbi.nlm.nih.gov/pmc/articles/{PMC}/", "doi": DOI, "pmid": "123"}
    assert W.keys(rec) == [f"doi:{DOI}", f"pmcid:{PMC}", "pmid:123",
                           f"url:https://www.ncbi.nlm.nih.gov/pmc/articles/{PMC}"]


def test_same_work_doi_only_pmcid_only_url_only_is_one_group():
    """Acceptance (backlog OA-3): DOI-only, PMCID-only and URL-only sightings resolve to one key once any record links
    them; the canonical key is the DOI."""
    ix = W.KeyIndex()
    a = ix.add({"doi": DOI})
    b = ix.add({"pmcid": PMC})
    c = ix.add({"url": f"https://europepmc.org/article/PMC/{PMC}"})
    assert a != b  # not yet linked: no record names both
    ix.add({"doi": DOI, "pmcid": PMC})  # e.g. the Europe PMC record
    groups = {ix.group(r) for r in ({"doi": DOI}, {"pmcid": PMC}, {"url": f"https://europepmc.org/article/PMC/{PMC}"})}
    assert len(groups) == 1 and ix.canonical(groups.pop()) == f"doi:{DOI}"
    assert c is not None


def test_url_only_with_doi_in_url_joins_doi_group():
    ix = W.KeyIndex()
    g = ix.add({"doi": DOI})
    assert ix.add({"url": f"https://doi.org/{DOI}"}) == g


def test_distinct_works_stay_distinct():
    ix = W.KeyIndex()
    assert ix.add({"doi": "10.9999/a"}) != ix.add({"doi": "10.9999/b"})
    assert ix.add({}) is None and ix.group({"doi": "10.9999/c"}) is None


def test_preprint_and_published_are_linked_not_merged():
    rec = {"arxiv": {"id": "2605.23989"}, "doi": "10.1234/journal.5"}
    assert W.version_link(rec) == {"preprint": "arxiv:2605.23989", "published": "doi:10.1234/journal.5"}
    assert W.work_key(rec) == "doi:10.1234/journal.5"  # published version wins
    assert W.version_link({"doi": "10.48550/arxiv.2605.23989"}) is None
    assert W.work_key({"doi": "10.48550/arxiv.2605.23989"}) != W.work_key({"doi": "10.1234/journal.5"})


def test_group_merge_keeps_earliest_id_and_best_key():
    ix = W.KeyIndex()
    g1 = ix.add({"url": "https://a.org/x"})
    ix.add({"pmid": "77"})
    g = ix.add({"url": "https://a.org/x", "pmid": "77"})
    assert g == g1 and ix.canonical(g) == "pmid:77"


def test_module_is_pure():
    import ast
    import pathlib
    src = (pathlib.Path(W.__file__)).read_text()
    names = {a.name.split(".")[0] for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Import) for a in n.names}
    froms = {(n.module or "") for n in ast.walk(ast.parse(src)) if isinstance(n, ast.ImportFrom)}
    assert not names & {"requests", "httpx", "socket", "urllib", "sqlalchemy", "psycopg", "psycopg2"}
    assert froms <= {"__future__", "vera", "pathlib"}  # reads only its own config file


@pytest.mark.parametrize("rec,want", [
    ({"identifier": "arXiv:hep-th/9901001v3"}, "arxiv:hep-th/9901001"),
    ({"url": "https://arxiv.org/abs/hep-th/9901001"}, "arxiv:hep-th/9901001"),
    ({"doi": "10.48550/ARXIV.2605.23989"}, "arxiv:2605.23989"),
    ({"pmid": "0123"}, "pmid:123"),
    ({"pmid": 123}, "pmid:123"),
    ({"pmid": "0"}, None),
    ({"url": ["x"]}, None),
    ({"url": 5, "doi": None, "pmcid": None, "arxiv": None}, None),
])
def test_work_key_edge_inputs(rec, want):
    assert W.work_key(rec) == want


def test_three_way_group_merge():
    ix = W.KeyIndex()
    a, b, c = ix.add({"url": "https://a.org/1"}), ix.add({"pmid": "9"}), ix.add({"doi": "10.9999/t"})
    assert len({a, b, c}) == 3
    g = ix.add({"url": "https://a.org/1", "pmid": "9", "doi": "10.9999/t"})
    assert g == a and {ix.group({"pmid": "9"}), ix.group({"doi": "10.9999/t"})} == {a}
    assert ix.canonical(b) == ix.canonical(c) == "doi:10.9999/t"


def test_mode_defaults_off_and_reads_config(tmp_path):
    assert W.mode(tmp_path / "missing.json") == "off"
    (tmp_path / "w.json").write_text('{"mode": "sideways"}')
    assert W.mode(tmp_path / "w.json") == "off"
    (tmp_path / "w.json").write_text('{"mode": "on"}')
    assert W.mode(tmp_path / "w.json") == "on"


def test_extra_keys_never_raises():
    assert W.extra_keys(None, set()) == set()
    assert W.extra_keys({"doi": DOI}, {f"doi:{DOI}"}) == set()
    assert W.extra_keys({"doi": DOI, "pmcid": PMC}, {f"doi:{DOI}"}) == {f"pmcid:{PMC}"}

"""PROVFIX-1028: provenance LABEL fixes for new rows (declared_via, embedded licence flag, licence version,
discovered_via). Labels only: no licence decision changes. Synthetic fixtures, no DB, no network."""
from vera import fulltext_loader as fl

CFG = fl.load_config()
ITEM = {"fulltext_url": "https://example.org/x.xml", "landing_url": "https://example.org/x",
        "licence": "cc by", "licence_url": "https://creativecommons.org/licenses/by/"}


def _meta(emb=None, jl=None):
    m = {"meta": {"licence": emb or {"id": None, "url": None}}}
    if jl is not None:
        m["jats_licences"] = jl
    return m


def test_declared_via_defaults_to_manifest_declared():  # curated path unchanged
    r = fl.check_licence(dict(ITEM), _meta(), CFG)
    assert r["licence"]["content"]["source"] == "manifest declared"


def test_declared_via_reflects_item_licence_source():
    r = fl.check_licence({**ITEM, "licence_source": "Europe PMC licence field + JATS licence elements"}, _meta(), CFG)
    assert r["licence"]["content"]["source"] == "Europe PMC licence field + JATS licence elements"
    assert r["licence_decision"] == "allow"


def test_embedded_recorded_from_jats_licence_element_without_href():
    jl = [{"href": None, "type": None, "ref": None, "text": "This is an open access article under CC BY."}]
    r = fl.check_licence(dict(ITEM), _meta(jl=jl), CFG)
    assert r["licence"]["embedded"]["statement"].startswith("This is an open access") and r["licence_decision"] == "allow"
    assert "embedded" not in fl.check_licence(dict(ITEM), _meta(jl=[]), CFG)["licence"]
    assert "embedded" not in fl.check_licence(dict(ITEM), _meta(), CFG)["licence"]


def test_embedded_unchanged_when_converter_found_one():
    emb = {"id": "open-access", "url": "https://creativecommons.org/licenses/by/4.0/", "statement": "s"}
    r = fl.check_licence({**ITEM, "licence_url": "https://creativecommons.org/licenses/by/4.0/"}, _meta(emb), CFG)
    assert r["licence"]["embedded"]["url"] == emb["url"]


def test_licence_version_from_url_and_decision_unchanged():
    it = {**ITEM, "licence": "CC BY (version not stated in article record)",
          "licence_url": "https://creativecommons.org/licenses/by/4.0/"}
    r = fl.check_licence(it, _meta(), CFG)
    assert r["licence"]["content"]["version"] == "4.0" and r["licence"]["content"]["id"] == it["licence"]
    assert r["licence_decision"] == "allow"
    assert "version" not in fl.check_licence(dict(ITEM), _meta(), CFG)["licence"]["content"]  # version-less URL
    assert fl.licence_version("CC BY 3.0") == "3.0" and fl.licence_version("https://x/licenses/by-nc-sa/4.0") == "4.0"
    assert fl.licence_version("cc by") is None and fl.licence_version(None) is None


def test_discovered_via_set_on_acquired_rows_and_absent_on_curated(tmp_path):
    from tests import acq_support as A
    from tests import fulltext_synth as fs
    from vera import auto_acquire
    web, _ = A.epi_setup(topic="epi")
    store = A.Store()
    auto_acquire.run_job(A.job("what is the serial interval of an epidemic"), A.cfg(),
                         A.deps([A.cand("10.9999/acq.epi4")], web, store=store), clock=A.clock_at(), env=A.ENV_ON)
    prov = store.plans[0]["source"]["provenance"]
    assert prov["discovered_via"] == "europepmc" and prov["item"]["licence_source"] == auto_acquire.LICENCE_SOURCE_EPMC
    assert prov["licence"]["content"]["source"] == auto_acquire.LICENCE_SOURCE_EPMC
    from scripts import ingest_fulltext as cli
    m = fs.write_corpus(tmp_path)
    assert cli.main(["convert", str(m), "--raw-dir", str(tmp_path / "raw")]) in (0, 2)
    import json
    rec = fl.prepare(json.loads(m.read_text())["sources"][0], m.parent, CFG, now="2026-10-18T00:00:00+00:00")
    assert "discovered_via" not in rec["provenance"] and rec["provenance"]["licence"]["content"]["source"] == "manifest declared"

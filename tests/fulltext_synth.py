"""SYNTHETIC fixtures for the DARINT full-text ingest tests. Every article, DOI (10.9999/...), author and sentence here
is invented; nothing is copied from a real document. Generated deterministically (no randomness, no I/O except
write_corpus, which writes only into the directory it is given)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

TOPICS = {
    "epi": ["transmission", "incidence", "outbreak", "infection", "reproduction", "surveillance", "vaccination",
            "contact", "pathogen", "forecast", "hospital", "serial", "interval", "cases", "epidemic"],
    "clim": ["temperature", "radiative", "forcing", "ensemble", "precipitation", "ocean", "warming", "aerosol",
             "carbon", "circulation", "projection", "glacier", "anomaly", "atmosphere", "model"],
    "repro": ["reproducibility", "software", "provenance", "version", "workflow", "notebook", "archive", "script",
              "dependency", "pipeline", "repository", "rerun", "container", "metadata", "commit"],
}


def _sentences(topic: str, n: int, seed: int) -> list[str]:
    w = TOPICS[topic]
    out = []
    for i in range(n):
        a, b, c, d = (w[(seed + i * k) % len(w)] for k in (1, 3, 5, 7))
        out.append(f"SYNTHETIC sentence {i}: the {a} of {b} was compared with {c} under {d} conditions.")
    return out


def jats(doc_id: str, topic: str, doi: str, *, licence_url="https://creativecommons.org/licenses/by/4.0/",
         body_paras=12, abstract_sents=3, with_body=True, extra_prolog="", seed=1) -> bytes:
    def para(k):
        s = " ".join(_sentences(topic, 6, seed + k))
        return f'<p>{s} See <xref ref-type="bibr" rid="r{k % 3 + 1}">[{k % 3 + 1}]</xref>.</p>'
    secs = ""
    if with_body:
        half = body_paras // 2
        secs = (f'<body><sec id="s1"><label>1.</label><title>Introduction</title>{"".join(para(k) for k in range(half))}'
                f'<sec id="s1a"><title>Background</title>{para(99)}</sec></sec>'
                f'<sec id="s2"><label>2.</label><title>Methods</title>{"".join(para(k) for k in range(half, body_paras))}'
                f'<disp-formula><tex-math>R_0</tex-math></disp-formula></sec></body>')
    lic = f'<permissions><license xlink:href="{licence_url}"><license-p>SYNTHETIC licence line.</license-p></license>' \
          f'</permissions>' if licence_url else ""
    refs = "".join(f'<ref id="r{k}"><label>{k}</label><mixed-citation>Synthetic Author {k}. A synthetic reference. '
                   f'doi:10.9999/ref.{k}</mixed-citation></ref>' for k in (1, 2, 3))
    x = (f'<?xml version="1.0" encoding="UTF-8"?>{extra_prolog}'
         f'<article xmlns:xlink="http://www.w3.org/1999/xlink"><front><journal-meta><journal-title-group>'
         f'<journal-title>SYNTHETIC Journal</journal-title></journal-title-group></journal-meta><article-meta>'
         f'<article-id pub-id-type="doi">{doi}</article-id><title-group><article-title>SYNTHETIC {doc_id} study'
         f'</article-title></title-group><contrib-group><contrib contrib-type="author"><name><surname>Synth'
         f'</surname><given-names>A.</given-names></name></contrib></contrib-group><pub-date><year>2021</year>'
         f'</pub-date>{lic}<abstract><p>{" ".join(_sentences(topic, abstract_sents, seed + 50))}</p></abstract>'
         f'</article-meta></front>{secs}<back><ack><p>SYNTHETIC thanks.</p></ack><ref-list>{refs}</ref-list></back>'
         f'</article>')
    return x.encode("utf-8")


def html(doc_id: str, topic: str, doi: str) -> bytes:
    paras = "".join(f"<p>{' '.join(_sentences(topic, 6, k))} <a href='#ref1'>[1]</a></p>" for k in range(12))
    h = (f"<!doctype html><html><head><title>SYNTHETIC {doc_id}</title><meta name='citation_doi' content='{doi}'>"
         f"<link rel='license' href='https://creativecommons.org/licenses/by/4.0/'><script>var x=1;</script></head>"
         f"<body><nav>menu</nav><article><h1>SYNTHETIC {doc_id}</h1><h2>Abstract</h2><p>"
         f"{' '.join(_sentences(topic, 3, 50))}</p><h2>Introduction</h2>{paras}<h2>References</h2><ol>"
         f"<li id='ref1'>Synthetic ref. doi:10.9999/ref.1</li></ol></article><footer>foot</footer></body></html>")
    return h.encode("utf-8")


def item(doc_id: str, doi: str, **over) -> dict:
    base = {"document_id": doc_id, "domain_hint": "SYNTHETIC-HINT-NEVER-READ", "title": f"SYNTHETIC {doc_id} study",
            "authors": "Synth et al.", "published_year": 2021, "doi": doi, "landing_url": f"https://doi.org/{doi}",
            "fulltext_url": f"https://example.invalid/{doc_id}.xml", "fulltext_format": "jats-xml",
            "text_file": f"texts/{doc_id}.txt", "licence": "CC BY 4.0",
            "licence_url": "https://creativecommons.org/licenses/by/4.0/",
            "attribution": f"SYNTHETIC attribution for {doc_id}", "fetched_at": "2026-10-18T00:00:00Z",
            "verifier_status": "VERIFIED", "read_status": "VERBATIM"}
    base.update(over)
    return base


DEFAULT_DOCS = (("syn_epi", "epi", "10.9999/syn.epi"), ("syn_clim", "clim", "10.9999/syn.clim"),
                ("syn_repro", "repro", "10.9999/syn.repro"))


def write_corpus(root: Path, docs=DEFAULT_DOCS, items_over: dict | None = None, raw_over: dict | None = None):
    """raw/<id>.xml + manifest.json under root. Returns the manifest path. Conversion is done by the code under test."""
    root = Path(root)
    (root / "raw").mkdir(parents=True, exist_ok=True)
    items = []
    for i, (did, topic, doi) in enumerate(docs):
        raw = (raw_over or {}).get(did) or jats(did, topic, doi, seed=i + 1)
        (root / "raw" / f"{did}.xml").write_bytes(raw)
        items.append(item(did, doi, **((items_over or {}).get(did) or {})))
    m = root / "manifest.json"
    m.write_text(json.dumps({"_doc": "SYNTHETIC test manifest", "schema_version": 1, "sources": items}, indent=1))
    return m


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

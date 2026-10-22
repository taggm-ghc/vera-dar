"""PLAN-NONDOI-1022 AC1 golden: canonical JSON of the DOI-path outputs (curated prepare_from_parts, acquisition
resolve -> build_document -> build_plan). SYNTHETIC inputs only (tests/fulltext_synth.py, tests/acq_support.py); no
network, no DB. Uses only APIs that exist before the change, so the same function produced the pinned fixture on the
pre-change tree (65af0ac + DAR-TRACE-1022 fix.diff) and is re-run by tests/test_non_doi_identity.py after it."""
from __future__ import annotations

import hashlib
import json

from tests import acq_support as A
from tests import fulltext_synth as fs
from vera import acq_config, auto_acquire, jats_text
from vera import fulltext_loader as fl

NOW = "2026-10-22T00:00:00+00:00"


def _parts(raw: bytes, did: str, cfg: dict) -> tuple[str, dict]:
    conv = jats_text.convert(raw, "jats-xml", cfg["converter"])
    text = conv["text"]
    meta = {k: v for k, v in conv.items() if k != "text"}
    meta.update({"document_id": did, "raw_sha256": hashlib.sha256(raw).hexdigest(),
                 "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(), "raw_file": None})
    return text, meta


def doi_outputs() -> dict:
    out: dict = {"curated": {}, "acquired": {}}
    base = fl.load_config()
    for i, (did, topic, doi) in enumerate(fs.DEFAULT_DOCS):
        item = fs.item(did, doi)
        text, meta = _parts(fs.jats(did, topic, doi, seed=i + 1), did, base)
        rec = fl.prepare_from_parts(item, text, meta, base, now=NOW)
        out["curated"][did] = rec
    cfg = A.cfg()
    icfg = acq_config.ingest_cfg(base, cfg)
    for doi, pmcid in (("10.9999/acq.epi4", "PMC1234567"), ("10.9999/acq.epi5", "PMC7654321")):
        web, raw = A.epi_setup(doi=doi, pmcid=pmcid)
        d = A.deps([A.cand(doi)], web)
        res = auto_acquire.resolve_oa(doi, cfg, d)
        res = {k: v for k, v in res.items() if k != "doi"}  # PLAN-NONDOI-1022 adds res["doi"]; never stored
        job = auto_acquire.Job(question=A.Q_EPI, trigger="insufficient", job_id="0" * 32)
        url = cfg["resolve"]["fulltext_url_template"].format(pmcid=pmcid)
        rec, item, meta, text = auto_acquire.build_document(doi, A.cand(doi), res, raw, url, cfg, icfg, job, 1.0, NOW)
        plan = auto_acquire.build_plan(rec, item, meta, cfg, job, 1.0, {"open_access": True, "licence_family": "by",
                                                                         "pub_types_n": 1}, None, NOW)
        out["acquired"][doi] = {"rec": rec, "item": item, "meta": meta, "plan": plan}
    return out


def dumps(obj) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, indent=0)

"""SYNTHETIC offline fixtures for the WED-ACQ2 (P1.7) automatic acquisition tests. No network, no DB, no model: search,
HTTP, the read-only lookup, the write-only store and the DB-API connection are all fakes. All documents are invented
(tests/fulltext_synth.py); DOIs are 10.9999/..., PMC ids are fake."""
from __future__ import annotations

import copy
import json

from tests import fulltext_synth
from vera import acq_config, auto_acquire, fulldoc_db

SENTINEL = "ZQXJ-SENTINEL-QUESTION-7781"
Q_EPI = f"How was the transmission of the pathogen related to incidence during an outbreak {SENTINEL}?"
ENV_ON = {"VERA_DB_URL_WO": "postgresql://wo.invalid/x", "VERA_DB_URL_RO": "postgresql://ro.invalid/x"}


def cfg(**bounds) -> dict:
    c = copy.deepcopy(acq_config.load_config())
    c["bounds"].update(bounds)
    return c


class Resp:
    def __init__(self, status=200, body=b"", ctype="application/json", headers=None):
        self.status_code, self._body = status, body
        self.headers = {"Content-Type": ctype, **(headers or {})}

    def iter_content(self, n):
        for i in range(0, len(self._body), n):
            yield self._body[i:i + n]


def epmc_record(doi, pmcid="PMC1234567", licence="cc by", oa="Y", source="MED", pub_types=("research-article",),
                book=None, title="SYNTHETIC acquired outbreak study"):
    r = {"doi": doi, "pmcid": pmcid, "isOpenAccess": oa, "license": licence, "source": source,
         "pubTypeList": {"pubType": list(pub_types)}, "title": title, "authorString": "Synth A, Example B.",
         "pubYear": "2021"}
    if book:
        r["bookOrReportDetails"] = book
    return r


class Web:
    """Fake http_get: Europe PMC search JSON by DOI, then JATS full text by PMC id. Records every URL requested."""

    def __init__(self, records: dict, jats: dict, overrides: dict | None = None):
        self.records, self.jats, self.over, self.urls = records, jats, overrides or {}, []

    def __call__(self, url, headers=None, timeout=None, allow_redirects=None, stream=None):
        self.urls.append(url)
        for frag, resp in self.over.items():
            if frag in url:
                if isinstance(resp, Exception):
                    raise resp
                return resp
        if "/rest/search" in url:
            for doi, rec in self.records.items():
                if doi in url.replace("%2F", "/").replace("%22", '"'):
                    return Resp(body=json.dumps({"resultList": {"result": [rec]}}).encode())
            return Resp(body=b'{"resultList": {"result": []}}')
        for pmcid, raw in self.jats.items():
            if f"/{pmcid}/fullTextXML" in url:
                return Resp(body=raw, ctype="application/xml")
        return Resp(status=404)


class Lookup:
    def __init__(self, known=(), day=0, known_isbn=()):
        self.known_dois, self.day, self.known_isbn, self.calls = set(known), day, set(known_isbn), []

    def known(self, doi, url, isbn):
        self.calls.append((doi, url, isbn))
        return doi in self.known_dois or (isbn is not None and isbn in self.known_isbn)

    def docs_last_24h(self):
        return self.day


class Store:
    def __init__(self, status="stored"):
        self.plans, self.status = [], status

    def write(self, plan):
        self.plans.append(json.loads(json.dumps(plan)))
        return {"status": self.status, "run_id": 100 + len(self.plans), "source_id": 1000 + len(self.plans)}


def cand(doi, title="SYNTHETIC outbreak transmission incidence study", snippet="transmission incidence outbreak pathogen",
         **kw):
    return {"doi": doi, "url": f"https://doi.org/{doi}", "title": title, "snippet": snippet, **kw}


def deps(cands, web, lookup=None, store=None, keys=()):
    return auto_acquire.Deps(search=lambda q, n, t: list(cands), http_get=web, lookup=lookup or Lookup(),
                             store=store or Store(), corpus_keys=lambda: set(keys), url_check=lambda u: (True, ""),
                             refresh=None)


def epi_setup(doi="10.9999/acq.epi4", pmcid="PMC1234567", topic="epi", jats_kw=None, rec_kw=None, over=None):
    raw = fulltext_synth.jats("acq_epi", topic, doi, **(jats_kw or {}))
    web = Web({doi: epmc_record(doi, pmcid=pmcid, **(rec_kw or {}))}, {pmcid: raw}, over)
    return web, raw


def job(question=Q_EPI, trigger="insufficient"):
    return auto_acquire.Job(question=question, trigger=trigger, created=0.0)


def clock_at(t=1.0):
    return lambda: t


def rows_from_plans(plans, base_run=1):
    """(sources, spans) as the acquired SELECTs return them, from stored plans (newest = highest source_id)."""
    sources, spans = [], []
    for i, p in enumerate(plans, start=1):
        sid = 1000 + i
        sources.append({"run_id": base_run + i, "source_id": sid, "canonical_url": p["canonical_url"],
                        "content": p["source"]["content"], "content_hash": p["source"]["content_hash"],
                        "provenance": json.dumps(p["source"]["provenance"])})
        spans += [{"source_id": sid, "start_index": s["start_index"], "end_index": s["end_index"], "text": s["text"]}
                  for s in p["spans"]]
    return sources, spans


class FakeConn:
    """DB-API stand-in that accepts ONLY the acquired-path SELECT constants (plus the span SELECT)."""

    def __init__(self, plans, log=None):
        self.sources, self.spans = rows_from_plans(plans)
        self.log = [] if log is None else log

    def cursor(self):
        return self

    def execute(self, sql, params):
        assert sql in (fulldoc_db.SQL_ACQ_MAX, fulldoc_db.SQL_ACQ_SOURCES, fulldoc_db.SQL_SPANS), "unexpected SQL"
        self.log.append(sql)
        mk = [s for s in self.sources if json.loads(s["provenance"]).get("acquisition", {}).get("marker")
              == params.get("marker", None)] if "marker" in params else None
        if sql == fulldoc_db.SQL_ACQ_MAX:
            self._rows = [(max((s["source_id"] for s in mk), default=None), len(mk))]
        elif sql == fulldoc_db.SQL_ACQ_SOURCES:
            self._rows = [(s["run_id"], s["source_id"], s["canonical_url"], s["content"], s["content_hash"],
                           s["provenance"]) for s in sorted(mk, key=lambda s: -s["source_id"])][: params["lim"]]
        else:
            ids = set(params["ids"])
            self._rows = [(x["source_id"], x["start_index"], x["end_index"], x["text"]) for x in self.spans
                          if x["source_id"] in ids]

    def fetchall(self):
        return list(self._rows)

    def close(self):
        pass

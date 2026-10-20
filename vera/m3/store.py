"""Persistence for M3 derived records, written against M1's schema
(db/migrations/001 + 003, schema vera_vjay) and reading M2's `sources`/`candidates`.

SQLAlchemy Core, so the same code runs on Postgres (schema vera_vjay) and on SQLite
with an attached 'vera_vjay' database (unit/integration tests).

Mapping from M3's internal rubric (1-5, 5 = no concern; see appraisal_rubric) to M1's
appraisals columns (003: risk dimensions 0-5 where higher = MORE concern; applicability
higher = better fit; overall_quality is a GRADE certainty label):
  bias/inconsistency/indirectness/imprecision/publication_bias_score = 5 - rubric score  (0..4)
  applicability_score = rubric score (1..5)
  NULL stays NULL (unknown is not zero)
  overall_quality: >=4.0 high | >=3.0 moderate | >=2.0 low | else very_low
The rubric version and Gate B policy version are written to appraisals.rubric_version /
policy_version (columns added by migration 005; previously folded into `rationale`). The numeric
overall has no column and stays as a prefix of `rationale` and in the corpus JSON.

Derived records are append-only: spans are upserted by (source_id,start,end) -- never
deleted, since evidence_relations may reference them -- and every run inserts a new
appraisals row (history kept; the newest row per source is current).
"""
from __future__ import annotations

import sqlalchemy as sa

SCHEMA = "vera_vjay"
metadata = sa.MetaData()
_ID = sa.BigInteger().with_variant(sa.Integer, "sqlite")

# Subsets of M1/M2 tables: only the columns M3 reads.
candidates = sa.Table(
    "candidates", metadata,
    sa.Column("candidate_id", _ID, primary_key=True),
    sa.Column("run_id", _ID), sa.Column("source_url", sa.Text), sa.Column("title", sa.Text),
    schema=SCHEMA)
sources = sa.Table(
    "sources", metadata,
    sa.Column("source_id", _ID, primary_key=True),
    sa.Column("candidate_id", _ID), sa.Column("version", sa.Integer),
    sa.Column("content", sa.Text), sa.Column("content_hash", sa.Text), sa.Column("provenance", sa.JSON),
    schema=SCHEMA)

evidence_spans = sa.Table(
    "evidence_spans", metadata,
    sa.Column("span_id", _ID, primary_key=True, autoincrement=True),
    sa.Column("source_id", _ID, nullable=False),
    sa.Column("text", sa.Text, nullable=False),
    sa.Column("start_index", sa.Integer, nullable=False),
    sa.Column("end_index", sa.Integer, nullable=False),
    sa.Column("relevance_score", sa.Numeric(3, 2)),
    sa.Column("evidence_type", sa.Text),
    schema=SCHEMA)

appraisals = sa.Table(
    "appraisals", metadata,
    sa.Column("appraisal_id", _ID, primary_key=True, autoincrement=True),
    sa.Column("source_id", _ID, nullable=False),
    sa.Column("bias_score", sa.Numeric(2, 1)),
    sa.Column("inconsistency_score", sa.Numeric(2, 1)),
    sa.Column("indirectness_score", sa.Numeric(2, 1)),
    sa.Column("imprecision_score", sa.Numeric(2, 1)),
    sa.Column("publication_bias_score", sa.Numeric(2, 1)),
    sa.Column("applicability_score", sa.Numeric(2, 1)),
    sa.Column("overall_quality", sa.Text),
    sa.Column("gate_b_decision", sa.Text),
    sa.Column("rationale", sa.Text),
    sa.Column("rubric_version", sa.Text),   # migration 005
    sa.Column("policy_version", sa.Text),   # migration 005
    schema=SCHEMA)

_RISK_COLS = {"risk_of_bias": "bias_score", "inconsistency": "inconsistency_score",
              "indirectness": "indirectness_score", "imprecision": "imprecision_score",
              "publication_bias": "publication_bias_score"}


def db_scores(scores: dict) -> dict:
    out = {col: (None if scores.get(d) is None else 5 - scores[d]) for d, col in _RISK_COLS.items()}
    out["applicability_score"] = scores.get("applicability")
    return out


def certainty_label(overall: float | None) -> str | None:
    if overall is None:
        return None
    return "high" if overall >= 4.0 else "moderate" if overall >= 3.0 else "low" if overall >= 2.0 else "very_low"


def apply_provenance(r: dict) -> dict:
    """Lift provenance fields onto a source row (in place) so M3 readers (e.g. appraisal_rubric._year) see them."""
    prov = r.get("provenance") if isinstance(r.get("provenance"), dict) else {}
    r["url"] = r.pop("source_url", None) or prov.get("url")
    r["content"] = r.get("content") or ""
    for k in ("published_date", "published_at", "publication_date", "year", "publisher", "source_type"):
        if prov.get(k) is not None:
            r.setdefault(k, prov[k])
    # G7 (item #71): m2_runner stores the search date as provenance "published"; M3's year reader looks for
    # published_date/published_at/..., so map it (without overriding an explicit date key).
    if prov.get("published") is not None:
        r.setdefault("published_date", prov["published"])
    return r


def load_sources(engine, run_id: int | None = None, limit: int = 100, skip_appraised: bool = False) -> list[dict]:
    """Read M2's acquired sources (joined to their candidate for title/url/run). Bounded by
    `limit`. skip_appraised omits sources that already have any appraisal row."""
    if not sa.inspect(engine).has_table("sources", schema=SCHEMA):
        raise RuntimeError(f"{SCHEMA}.sources does not exist: M1/M2 have not shipped; nothing to appraise")
    q = (sa.select(sources, candidates.c.title, candidates.c.source_url, candidates.c.run_id)
         .select_from(sources.outerjoin(candidates, candidates.c.candidate_id == sources.c.candidate_id))
         .order_by(sources.c.source_id).limit(limit))
    if run_id is not None:
        q = q.where(candidates.c.run_id == run_id)
    if skip_appraised:
        q = q.where(~sa.exists().where(appraisals.c.source_id == sources.c.source_id))
    with engine.connect() as c:
        rows = [dict(r._mapping) for r in c.execute(q)]
    for r in rows:
        apply_provenance(r)
    return rows


def save_source_result(engine, source_id: int, spans: list[dict], gate: dict) -> list[int]:
    """One transaction: upsert this source's spans (returns their span_ids, in order) and
    append an appraisal row."""
    ap = gate["appraisal"]
    rationale = (f"[overall "
                 f"{ap['overall_quality'] if ap['overall_quality'] is not None else 'n/a'}/5] {gate['rationale']}")
    with engine.begin() as c:
        ids = []
        for s in spans:
            found = c.execute(sa.select(evidence_spans.c.span_id).where(sa.and_(
                evidence_spans.c.source_id == source_id, evidence_spans.c.start_index == s["start_index"],
                evidence_spans.c.end_index == s["end_index"]))).scalar()
            if found is None:
                found = c.execute(sa.insert(evidence_spans).values(
                    source_id=source_id, text=s["text"], start_index=s["start_index"], end_index=s["end_index"],
                    relevance_score=round(s["relevance_score"], 2), evidence_type=s["evidence_type"]
                )).inserted_primary_key[0]
            ids.append(found)
        c.execute(sa.insert(appraisals).values(
            source_id=source_id, **db_scores(gate["scores"]), overall_quality=certainty_label(ap["overall_quality"]),
            gate_b_decision=gate["decision"], rationale=rationale,
            rubric_version=ap["rubric_version"], policy_version=gate["policy"]["version"]))
    return ids

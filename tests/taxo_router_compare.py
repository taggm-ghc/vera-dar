"""TAXO-ROUTE-1110 offline harness (fixtures only; no network, no DB): global vs masked admission on one served corpus.

compare(corpus, questions) -> one row per question: route code, the globally admitted span ids, the span ids admitted when
every candidate outside the routed set R is removed (a post-retrieval mask: a lower bound on what real routing would
admit, since a real mask would also back-fill from inside R), counts lost and gained, and the shadow record's
missed_admitted. The caller sets the router mode to shadow and builds the index first."""
from __future__ import annotations

from vera import fulldoc_admission as fa, taxo_router as tr
from vera.fulldoc_corpus import load_config


def compare(corpus, questions) -> list:
    rows = []
    cfg = load_config()
    orig = corpus.retrieve
    for q in questions:
        g = fa.admit(q, corpus, cfg)
        gids = [a["span_id"] for a in g.admitted]
        rec = g.trace.get("route_shadow") or {}

        def masked(question, *a, **k):
            out = orig(question, *a, **k)
            inr = dict(getattr(tr._TL, "cand", None) or {})
            return [x for x in out if inr.get(x["span_id"], True)]
        corpus.retrieve = masked
        try:
            m = fa.admit(q, corpus, cfg)
        finally:
            corpus.retrieve = orig
        mids = [a["span_id"] for a in m.admitted]
        rows.append({"route": rec.get("route"), "global_admitted": gids, "masked_admitted": mids,
                     "lost": len(set(gids) - set(mids)), "gained": len(set(mids) - set(gids)),
                     "missed_admitted": rec.get("missed_admitted")})
    return rows

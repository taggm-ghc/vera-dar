"""DRAFTVERIFY-IMPL-1111 stage 3 (query side): turn triggered claims into an ordered list of gap queries for the existing
acquisition rounds (inline_acquire._extra_rounds). TERMS ONLY: each query is extracted through the PII-filtered term path
(vera.draft_verify.terms_query) and is never the raw draft sentence. Bounds: max_queries_per_claim per claim and
max_gap_queries per ask. A discrepancy also gets the contradicting chunk's key terms and a disconfirming variant (seed
words from config). Dedupe uses query_variants_acq.norm; degenerate queries are dropped."""
from __future__ import annotations


def build(triggered, claims, chunks, question: str, cfg: dict) -> list:
    from vera import query_variants_acq as qv
    from vera import search_query
    from vera.draft_verify import terms_query
    per, cap = int(cfg["max_queries_per_claim"]), int(cfg["max_gap_queries"])
    seen, out = set(), []

    def add(q, mine):
        n = qv.norm(q)
        if q and n and n not in seen and not search_query.is_degenerate_query(q) and len(mine) < per and len(out) < cap:
            seen.add(n)
            mine.append(q)
            out.append(q)

    for t in triggered:
        mine: list = []
        base = terms_query(claims[t["claim"]])
        add(base, mine)
        if t["kind"] in ("rebutted", "rejected", "mixed"):
            words = " ".join(cfg["disconfirm_terms"][:1])  # disconfirming variant first, then the contradicting terms
            add((base + " " + words).strip() if base else "", mine)
            ci = t.get("chunk")
            if isinstance(ci, int) and 0 <= ci < len(chunks):  # key terms of the contradicting assertion
                add(terms_query(str(chunks[ci].get("text") or "")[:300]), mine)
    return out

"""NONASK-1103 A6-2 (D1): the query variants of the inline acquisition round loop.

A thin wrapper over vera.query_variants_acq.variants (drop terms, term-group lexicon, entity-central terms from
vera.subquery (S23), config seed list, OR form). Pure: no LLM, no network, no clock. The result is the finite ordered
list V of queries to try AFTER the first search: never the raw sentence, no operators other than the single OR form, no
hosts, de-duplicated by aa._qkey against the queries already run. Never raises ([] on failure = no more rounds).
"""
from __future__ import annotations

from vera import query_variants_acq


def variants(question: str, tried=(), **kw) -> list[str]:
    try:
        from vera import auto_acquire as aa
        raw = aa._qkey(question)
        seen = {aa._qkey(t) for t in tried} | {raw}
        out: list[str] = []
        for v in query_variants_acq.variants(question, tried=tried, **kw):
            k = aa._qkey(v)
            if v.strip() and k not in seen:
                seen.add(k)
                out.append(v.strip())
        return out
    except Exception:  # noqa: BLE001
        return []

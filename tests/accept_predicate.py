"""ACCEPT-D-1023: the single grounded-answer acceptance predicate, reused by every acceptance test.

Mirrors work/dar-pipeline-1022/smokefix/dar_ask_smoke.sh --require-grounded (http 200, result_kind=answer, outcome=answer,
>=1 admitted passage, >=1 citation) and is STRICTER on the owner's DEMOQ-1023 acceptance list: full-text (not abstract)
passages, every [n] and every citation bound to an admitted passage of an examined (stored) document, claim-check
verdicts present, build identity recorded. A reading list, an abstention or an HTTP 200 alone never passes.
Pure function of the response bodies: no I/O.
"""
from __future__ import annotations

import re

NON_FULLTEXT = ("abstract", "abstract_metadata", "snippet", "metadata", "index_record")
VERDICTS = ("supported", "partial", "partly_supported", "partially_supported")
_MARK = re.compile(r"\[(\d+)\]")
_SHA = re.compile(r"^[0-9a-f]{7,40}$")


def grounded_failures(http, body, version=None, expect_commit=None, stored_doc_ids=None) -> list:
    """Return the list of failed criteria (empty list = PASS)."""
    f = []
    if http != 200:
        return [f"http={http}"]
    if not isinstance(body, dict):
        return ["body not an object"]
    if body.get("result_kind") != "answer":
        f.append(f"result_kind={body.get('result_kind')!r}")
    if body.get("outcome") != "answer":
        f.append(f"outcome={body.get('outcome')!r}")
    if (body.get("contract") or {}).get("fallback") is not None:
        f.append("fallback present")
    adm = body.get("admitted_evidence") or []
    cits = body.get("citations") or []
    srcs = body.get("sources") or []
    exm = {d.get("document_id"): d for d in body.get("examined_full_documents") or [] if isinstance(d, dict)}
    if not adm:
        f.append("no admitted passage")
    if not cits:
        f.append("no citations")
    if not exm:
        f.append("no examined full document")
    by_span = {a.get("span_id"): a for a in adm if isinstance(a, dict)}
    for a in adm:
        if a.get("document_id") not in exm:
            f.append(f"admitted passage {a.get('span_id')} from a document not examined")
        if str(a.get("source_type") or a.get("text_kind") or "") in NON_FULLTEXT:
            f.append(f"admitted passage {a.get('span_id')} is not full text")
    for d in exm.values():
        if not d.get("text_sha256") or not d.get("chars") or str(d.get("text_kind") or "") in NON_FULLTEXT:
            f.append(f"examined document {d.get('document_id')} is not a full text")
    by_n = {s.get("n"): s for s in srcs if isinstance(s, dict)}
    for s in srcs:
        if s.get("source_type") != "full_text_passage" or s.get("span_id") not in by_span:
            f.append(f"source [{s.get('n')}] is not an admitted full-text passage")
    if not any(s.get("source_type") == "full_text_passage" for s in srcs):
        f.append("no full-text passage among sources")
    for c in cits:
        a = by_span.get(c.get("span_id"))
        if a is None:
            f.append(f"citation {c.get('span_id')} not bound to an admitted passage")
            continue
        if c.get("document_id") != a.get("document_id") or c.get("document_id") not in exm:
            f.append(f"citation {c.get('span_id')} does not resolve to its examined document")
        if stored_doc_ids is not None and c.get("document_id") not in stored_doc_ids:
            f.append(f"citation {c.get('span_id')} document is not in the store")
        if not c.get("quote"):
            f.append(f"citation {c.get('span_id')} has no quote")
        if c.get("verdict") not in VERDICTS:
            f.append(f"citation {c.get('span_id')} verdict={c.get('verdict')!r}")
        s = by_n.get(c.get("n"))
        if s is None or s.get("span_id") != c.get("span_id"):
            f.append(f"citation n={c.get('n')} does not match a numbered source")
    marks = {int(m) for m in _MARK.findall(body.get("answer") or "")}
    if not marks:
        f.append("answer text carries no [n] citation marker")
    for m in marks:
        if m not in by_n or not any(c.get("n") == m for c in cits):
            f.append(f"marker [{m}] has no citation")
    cc = body.get("claim_check") or {}
    if cc.get("outcome") != "checked" or not cc.get("checked"):
        f.append("claim check did not run")
    elif cc.get("removed", 0) >= cc["checked"] or cc.get("supported", 0) + cc.get("partial", 0) < 1:
        f.append("claim check kept no supported claim")
    if version is not None or expect_commit is not None:
        v = (version or {}).get("commit") if isinstance(version, dict) else None
        if not isinstance(v, str) or not _SHA.match(v) or (expect_commit and not v.startswith(expect_commit.lower())):
            f.append(f"build identity missing or unexpected ({v!r})")
    return f


def is_grounded(http, body, **kw) -> bool:
    return not grounded_failures(http, body, **kw)

"""Evidence relation graph: LLM-extracted supports/refutes/qualifies edges between spans,
plus deterministic dependency detection (shared source / derived_from).

Graph = {"nodes":[{span_id, source_id, label}], "edges":[{span1_id, span2_id, relation_type,
confidence, rationale}], "contradictions":[edge...], "dependencies":[{span_ids, reason}]}.
Edge direction: span1 <relation> span2 ("span1 refutes span2").
"""
import logging

from vera.pipeline_llm import UNTRUSTED_NOTE, LLMCallable, LLMError, call
from vera.m4.common import labels, ordered_spans

logger = logging.getLogger("vera")

RELATION_TYPES = ("supports", "refutes", "qualifies")
Graph = dict


def _dependencies(spans: list[dict]) -> list[dict]:
    deps, by_src = [], {}
    for s in spans:
        by_src.setdefault(s.get("source_id"), []).append(s["span_id"])
        for parent in s.get("derived_from") or []:
            deps.append({"span_ids": [s["span_id"]], "depends_on_source": parent, "reason": "derived_from"})
    for src, ids in by_src.items():
        if len(ids) > 1:
            deps.append({"span_ids": ids, "source_id": src, "reason": "same_source"})
    return deps


def build_relations(evidence_corpus: dict, *, llm: LLMCallable | None = None) -> Graph:
    spans = ordered_spans(evidence_corpus)
    lab = labels(spans)
    nodes = [{"span_id": s["span_id"], "source_id": s.get("source_id"), "label": lab[s["span_id"]]} for s in spans]
    edges: list[dict] = []
    if len(spans) >= 2:
        body = "\n".join(
            f"[{lab[s['span_id']]}] ({s.get('source_type', '?')}, {s.get('year', '?')}, {s.get('source_title', '')}) "
            f"{s['text'][:400]}" for s in spans)
        system = (
            "Identify pairwise relations between evidence units about the research question. "
            "Relation 'A supports B' = A reports a finding consistent with B; 'A refutes B' = A reports a "
            "finding that contradicts B; 'A qualifies B' = A narrows, conditions or explains the scope of B "
            "(e.g. different population, task or measure). Only report relations clearly stated by the text; "
            "do not infer from outside knowledge. " + UNTRUSTED_NOTE +
            ' Reply JSON: {"relations":[{"from":"E1","to":"E2","type":"supports|refutes|qualifies",'
            '"confidence":0..1,"rationale":"short"}]}. At most 60 relations.'
        )
        user = f"Question: {evidence_corpus.get('question', '')}\n<evidence>\n{body}\n</evidence>"
        try:
            raw = call(llm, system, user).data.get("relations", [])
        except LLMError as exc:
            raise LLMError(f"relation extraction failed for run {evidence_corpus.get('run_id')}: {exc}") from exc
        inv = {v: k for k, v in lab.items()}
        seen = set()
        for r in raw[:60] if isinstance(raw, list) else []:
            try:
                a, b, t = inv[r["from"]], inv[r["to"]], str(r["type"]).lower()
                conf = min(1.0, max(0.0, float(r.get("confidence", 0.5))))
            except (KeyError, TypeError, ValueError):
                logger.info("dropping malformed/unknown relation: %r", r)
                continue
            if a == b or t not in RELATION_TYPES or (a, b, t) in seen:
                continue
            seen.add((a, b, t))
            edges.append({"span1_id": a, "span2_id": b, "relation_type": t,
                          "confidence": round(conf, 3), "rationale": str(r.get("rationale", ""))[:300]})
    return {"nodes": nodes, "edges": edges,
            "contradictions": [e for e in edges if e["relation_type"] == "refutes"],
            "dependencies": _dependencies(spans)}

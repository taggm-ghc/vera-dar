"""Reasoning-context construction: compact, auditable text with evidence-unit labels
(E1..En), their span_ids, relations, dependencies, gaps and claim-to-evidence pointers.
Deterministic: same inputs -> same context (auditable for M7)."""
from vera.m4.common import labels, ordered_spans, sub_questions
from vera.source_labels import source_label

UNIT_CHARS = 450
LABEL_FMT = "[E{n}]"


def build_context_with_index(evidence_corpus: dict, relations: dict, gate_c: dict | None = None) -> tuple[str, dict]:
    spans = ordered_spans(evidence_corpus)
    lab = labels(spans)
    sqs = sub_questions(evidence_corpus)
    out = [f"REASONING CONTEXT (run {evidence_corpus.get('run_id', '?')})",
           f"Question: {evidence_corpus.get('question', '')}"]
    if gate_c:
        out.append(f"Gate C: decision={gate_c['decision']} qualified={gate_c.get('qualified', False)} "
                   f"coverage={gate_c.get('coverage_ratio')}")
        for g in gate_c.get("residual_gaps", []):
            out.append(f"  UNRESOLVED GAP [{g.get('sub_question_id')}]: {g['need']}")
    total = len(evidence_corpus.get("spans", []))
    if total > len(spans):
        out.append(f"NOTE: {total - len(spans)} further span(s) omitted by the {len(spans)}-unit context cap.")

    out.append("\n## Evidence units (cite as [E#])")
    placed = set()
    for sq in sqs:
        mine = [s for s in spans if sq["id"] in (s.get("sub_question_ids") or []) and s["span_id"] not in placed]
        out.append(f"### {sq['id']}: {sq['text']}" + ("" if mine else "  -- NO EVIDENCE ADMITTED"))
        for s in mine:
            placed.add(s["span_id"])
            meta = ", ".join(str(x) for x in (s.get("source_title"), s.get("year"), s.get("source_type")) if x)
            text = " ".join(s["text"].split())
            text = text if len(text) <= UNIT_CHARS else text[:UNIT_CHARS].rstrip() + "..."
            cls = source_label(s)  # R72-f: attribution label (news / vendor claim); no behavioural effect claimed
            meta = f"{cls} {meta}".strip() if cls else meta
            out.append(f"{LABEL_FMT.format(n=lab[s['span_id']][1:])} span_id={s['span_id']} | {meta} | \"{text}\"")
    rest = [s for s in spans if s["span_id"] not in placed]
    if rest:
        out.append("### unassigned")
        for s in rest:
            cls = source_label(s)
            head = f" | {cls}" if cls else ""
            out.append(f"[{lab[s['span_id']]}] span_id={s['span_id']}{head} | \"{' '.join(s['text'].split())[:UNIT_CHARS]}\"")

    edges = [e for e in relations.get("edges", []) if e["span1_id"] in lab and e["span2_id"] in lab]
    if edges:
        out.append("\n## Evidence relations (A <relation> B)")
        for e in sorted(edges, key=lambda e: (e["relation_type"], lab[e["span1_id"]], lab[e["span2_id"]])):
            out.append(f"{lab[e['span1_id']]} {e['relation_type'].upper()} {lab[e['span2_id']]} "
                       f"(conf {e['confidence']:.2f}): {e.get('rationale', '')}")
    deps = relations.get("dependencies", [])
    if deps:
        out.append("\n## Non-independence")
        for d in deps:
            ids = ", ".join(lab.get(i, i) for i in d["span_ids"])
            out.append(f"{ids}: {d['reason']}" + (f" ({d.get('source_id') or d.get('depends_on_source')})" if (d.get('source_id') or d.get('depends_on_source')) else ""))

    pointers = []
    for s in spans:
        sid, l = s["span_id"], lab[s["span_id"]]
        sup = [lab[e["span1_id"]] for e in edges if e["span2_id"] == sid and e["relation_type"] == "supports"]
        ref = [lab[e["span1_id"]] for e in edges if e["span2_id"] == sid and e["relation_type"] == "refutes"]
        qua = [lab[e["span1_id"]] for e in edges if e["span2_id"] == sid and e["relation_type"] == "qualifies"]
        if sup or ref or qua:
            parts = [f"{l} itself"] + ([f"supported by {', '.join(sup)}"] if sup else []) \
                + ([f"contradicted by {', '.join(ref)}"] if ref else []) + ([f"qualified by {', '.join(qua)}"] if qua else [])
            claim = " ".join(s["text"].split())[:120]
            pointers.append(f"Claim \"{claim}...\" -> " + "; ".join(parts))
    if pointers:
        out.append("\n## Claim-to-evidence pointers")
        out.extend(pointers)

    index = {"label_to_span_id": {v: k for k, v in lab.items()}, "unit_count": len(spans)}
    return "\n".join(out), index


def build_reasoning_context(evidence_corpus: dict, relations: dict, gate_c: dict | None = None) -> str:
    return build_context_with_index(evidence_corpus, relations, gate_c)[0]

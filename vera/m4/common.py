"""Shared helpers for M4 so Gate C, relations and context agree on span order and labels."""
MAX_UNITS = 30  # hard cap on spans entering relations/context (bounded cost + compact context)

# Sub-questions for the locked research question (used when the corpus carries none).
DEFAULT_SUB_QUESTIONS = [
    {"id": "sq_controlled", "required": True,
     "text": "Controlled experiments (RCTs) measuring AI coding assistant effects on task speed/output"},
    {"id": "sq_field", "required": True,
     "text": "Field/telemetry/observational evidence from real development settings"},
    {"id": "sq_quality", "required": True,
     "text": "Effects on code quality, defects, security or maintainability"},
    {"id": "sq_disagreement", "required": True,
     "text": "Explanations for why findings disagree (task type, developer experience, measurement, perception vs measured)"},
]


def sub_questions(corpus: dict) -> list[dict]:
    sqs = corpus.get("sub_questions") or DEFAULT_SUB_QUESTIONS
    return [{"id": s["id"], "text": s.get("text", s["id"]), "required": s.get("required", True)} for s in sqs]


def ordered_spans(corpus: dict, limit: int = MAX_UNITS) -> list[dict]:
    return list(corpus.get("spans", []))[:limit]


def labels(spans: list[dict]) -> dict[str, str]:
    """span_id -> 'E1'.. (1-based, in ordered_spans order)."""
    return {s["span_id"]: f"E{i}" for i, s in enumerate(spans, 1)}


def independence_key(span: dict) -> str:
    if span.get("independence_group"):
        return str(span["independence_group"])
    if span.get("derived_from"):
        return str(span["derived_from"][0])
    return str(span.get("source_id", span["span_id"]))

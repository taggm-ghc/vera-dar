"""M3 -> M4 adapter: flatten M3's nested corpus and tag spans with the frozen requirement ids.

M3 returns `sources[].spans[]` (vera/m3/m3_runner.py); M4 reads a flat `corpus["spans"]` plus
`sub_questions` (vera/m4/__init__.py contract). Nothing else in vera/ sets `sub_question_ids`
(Gate C and the context builder read them), so without this adapter every required
sub-question is `missing` and Gate C returns `insufficient` for a plumbing reason.

Decision RD-1 (PROVISIONAL, slim-live-mvp plan): one bounded, metered LLM call per batch of
spans over the 8 frozen requirements, which also become Gate C's sub-questions.

Rules (each is tested):
  * tags reference ONLY the frozen requirement ids; unknown ids are dropped and counted;
  * untagged spans are reported (`untagged_spans`), never silently assigned to a requirement;
  * LLM failure or malformed output raises (no keyword fallback, no silent untagged run);
  * zero spans, a span without an int `span_id`, or zero tags overall raise AdapterError;
  * spans beyond `max_batches * batch_size` are NOT tagged; they are counted, logged and listed;
  * spans beyond M4's MAX_UNITS (relations/context cap) are kept in the corpus (Gate C reads all
    spans) but are ordered after the top MAX_UNITS by relevance; they are counted and logged.

Bounds: the LLM call itself goes through vera.pipeline_llm.call (hard request timeout,
max_tokens and one JSON retry in complete_json; the S1 observer meters it). The tagger adds a
hard batch cap. The `llm` callable is a parameter, like the other pipeline stages.

appraisal.overall: M3 provides only `overall_quality` (float 1-5, None if nothing assessable) per
source. M5's `weak` rule reads `span["appraisal"]["overall"]` on a 0-1 scale (< 0.4). Per plan
CP-D5 the adapter does NOT carry it by default (inert arm, disclosed). `carry_appraisal=True`
opts in with a PROVISIONAL linear map; None is never invented (no `appraisal` key is set).
"""
import logging
from dataclasses import dataclass, field

from vera.m4.common import MAX_UNITS
from vera.pipeline_llm import UNTRUSTED_NOTE, LLMError, call

logger = logging.getLogger("vera")

# --- named constants (accepted divergence from the hardcoded-data standard; fold into
# config/vera_eval_run.json in goal #2 at the keys shown) -------------------------------------
TAG_MAX_BATCHES = 4        # why: bounds tagger calls/cost (~120 spans at 30 per batch). key pipeline.tagging.max_batches
TAG_SPAN_CHARS = 400       # why: same truncation M4 relations use, so tag input matches what M4 reads. key pipeline.tagging.span_chars
TAG_MAX_REQ_PER_SPAN = 3   # why: a span tagged to most requirements is noise; extras are dropped and counted. key pipeline.tagging.max_req_per_span
# PROVISIONAL mapping of M3 overall_quality (1-5) onto M5's 0-1 appraisal.overall: (q - lo) / (hi - lo).
# Why: the rubric scale is 1-5 (appraisal_rubric.synthesize_overall); M5 MIN_APPRAISAL=0.4 is on 0-1.
# Not validated against any outcome. key pipeline.appraisal_overall_map
APPRAISAL_QUALITY_RANGE = (1.0, 5.0)

SYSTEM_TAG = (
    "You tag evidence spans against a fixed list of requirement ids. For each span, list the ids of "
    "requirements the span DIRECTLY supports evidence for (zero, one or a few; leave empty if none). "
    "Use ONLY the given ids. " + UNTRUSTED_NOTE +
    ' Reply JSON: {"tags":[{"span":"S1","req_ids":["req_x"]}]}. Include every span label once.'
)


class AdapterError(RuntimeError):
    """Raised verbosely when the M3 corpus cannot be turned into a usable M4 corpus."""


@dataclass
class TagResult:
    tags: dict[int, list[str]] = field(default_factory=dict)   # span_id -> valid requirement ids
    dropped_ids: list[str] = field(default_factory=list)       # unknown/over-limit ids rejected
    untagged_spans: list[int] = field(default_factory=list)    # tagged-scope spans with no tag
    batches: int = 0
    skipped_spans: list[int] = field(default_factory=list)     # beyond the batch cap, not sent
    unknown_labels: int = 0                                    # rows whose span label was not in the batch

    def summary(self) -> dict:
        return {"batches": self.batches, "tagged_spans": sum(1 for v in self.tags.values() if v),
                "untagged_spans": list(self.untagged_spans), "skipped_spans": list(self.skipped_spans),
                "dropped_ids": list(self.dropped_ids), "unknown_labels": self.unknown_labels}


def _render(batch: list[tuple[str, dict]], requirements: list[dict], span_chars: int) -> str:
    reqs = "\n".join(f"- {r['id']}: {r['text']}" for r in requirements)
    spans = "\n".join(f"{label}: {sp['text'][:span_chars]}" for label, sp in batch)
    return f"Requirements:\n{reqs}\n<evidence>\n{spans}\n</evidence>"


def tag_spans(spans: list[dict], requirements: list[dict], *, llm=None, batch_size: int = MAX_UNITS,
              max_batches: int = TAG_MAX_BATCHES, span_chars: int = TAG_SPAN_CHARS) -> TagResult:
    if not requirements:
        raise AdapterError("tagger needs at least one requirement; got none")
    if batch_size < 1 or max_batches < 1:
        raise AdapterError(f"batch_size ({batch_size}) and max_batches ({max_batches}) must be >= 1")
    valid = [r["id"] for r in requirements]
    valid_set = set(valid)
    res = TagResult()
    scope = spans[: batch_size * max_batches]
    res.skipped_spans = [sp["span_id"] for sp in spans[batch_size * max_batches:]]
    if res.skipped_spans:
        logger.warning("tagger: %d span(s) beyond batch cap (%d x %d) are NOT tagged: %s",
                       len(res.skipped_spans), max_batches, batch_size, res.skipped_spans[:10])
    for b in range(0, len(scope), batch_size):
        chunk = scope[b: b + batch_size]
        labelled = [(f"S{i}", sp) for i, sp in enumerate(chunk, 1)]
        by_label = {lab: sp["span_id"] for lab, sp in labelled}
        ctx = f"tagging batch {res.batches + 1} ({len(chunk)} spans)"
        try:
            data = call(llm, SYSTEM_TAG, _render(labelled, requirements, span_chars)).data
        except LLMError as exc:
            raise LLMError(f"{ctx} failed: {exc}") from exc
        rows = data.get("tags") if isinstance(data, dict) else None
        if not isinstance(rows, list):
            raise AdapterError(f"{ctx}: malformed LLM output, 'tags' is not a list: {str(data)[:200]!r}")
        res.batches += 1
        for row in rows:
            if not isinstance(row, dict) or not isinstance(row.get("req_ids"), list):
                raise AdapterError(f"{ctx}: malformed tag row (need {{'span','req_ids':[..]}}): {str(row)[:200]!r}")
            sid = by_label.get(row.get("span"))
            if sid is None:
                res.unknown_labels += 1
                continue
            ids, seen = [], set()
            for i in row["req_ids"]:
                if isinstance(i, str) and i in valid_set and i not in seen and len(ids) < TAG_MAX_REQ_PER_SPAN:
                    ids.append(i)
                    seen.add(i)
                else:
                    res.dropped_ids.append(str(i))
            res.tags.setdefault(sid, [])
            res.tags[sid] = res.tags[sid] + [i for i in ids if i not in res.tags[sid]]
    res.untagged_spans = [sp["span_id"] for sp in scope if not res.tags.get(sp["span_id"])]
    if res.dropped_ids:
        logger.warning("tagger: dropped %d unknown/excess requirement id(s): %s",
                       len(res.dropped_ids), res.dropped_ids[:10])
    return res


def _appraisal_overall(quality) -> float | None:
    if not isinstance(quality, (int, float)) or isinstance(quality, bool):
        return None  # never invent a value
    lo, hi = APPRAISAL_QUALITY_RANGE
    return max(0.0, min(1.0, (float(quality) - lo) / (hi - lo)))


def adapt_m3_to_m4(m3_corpus: dict, requirements: list[dict], *, tagger=tag_spans, llm=None,
                   carry_appraisal: bool = False) -> dict:
    """Return {"spans", "sub_questions", "tagging"} for run_m4. Does not mutate `m3_corpus`."""
    spans = []
    for s in m3_corpus.get("sources", []):
        for sp in s.get("spans", []):
            out = dict(sp, source_id=s.get("source_id"), source_title=s.get("title"),
                       year=s.get("year"), source_type=s.get("source_type"))
            ov = _appraisal_overall(s.get("overall_quality")) if carry_appraisal else None
            if ov is not None:
                out["appraisal"] = {"overall": ov}
            spans.append(out)
    if not spans:
        raise AdapterError(f"M3 admitted 0 spans: nothing for Gate C (M3 counts: {m3_corpus.get('counts')})")
    bad = [sp for sp in spans if not isinstance(sp.get("span_id"), int) or isinstance(sp.get("span_id"), bool)]
    if bad:
        raise AdapterError(f"{len(bad)} span(s) have no int DB span_id (M3 persist failed?): "
                           f"{[sp.get('span_id') for sp in bad][:5]}; M3 failures: {m3_corpus.get('failures')}")
    ids = [sp["span_id"] for sp in spans]
    if len(set(ids)) != len(ids):
        raise AdapterError(f"duplicate span_id(s) in M3 corpus: {sorted({i for i in ids if ids.count(i) > 1})[:5]}")
    if any(not isinstance(sp.get("text"), str) or not sp["text"] for sp in spans):
        raise AdapterError("a span has empty or non-string text; refusing to tag")
    spans.sort(key=lambda sp: (-(sp.get("relevance_score") or 0), sp["span_id"]))  # M4 keeps the first MAX_UNITS
    beyond = [sp["span_id"] for sp in spans[MAX_UNITS:]]
    if beyond:
        logger.warning("adapter: %d span(s) rank beyond M4 MAX_UNITS=%d: kept for Gate C, excluded from "
                       "relations/context: %s", len(beyond), MAX_UNITS, beyond[:10])
    tr = tagger(spans, requirements, llm=llm)
    for sp in spans:
        sp["sub_question_ids"] = list(tr.tags.get(sp["span_id"], []))
    if not any(sp["sub_question_ids"] for sp in spans):
        raise AdapterError(f"tagging produced no tags over {len(spans)} spans (dropped ids {tr.dropped_ids}, "
                           f"skipped {len(tr.skipped_spans)}); Gate C would be an artefact")
    summary = tr.summary()
    summary.update({"spans_total": len(spans), "beyond_m4_max_units": beyond, "m4_max_units": MAX_UNITS,
                    "appraisal_carried": carry_appraisal})
    corpus = {"spans": spans,
              "sub_questions": [{"id": r["id"], "text": r["text"], "required": True} for r in requirements],
              "tagging": summary}
    summary["coverage"] = coverage_table(corpus)
    return corpus


def coverage_table(corpus: dict) -> dict[str, int]:
    """Requirement id -> number of spans tagged to it (all 8 listed, zeros included)."""
    table = {sq["id"]: 0 for sq in corpus.get("sub_questions", [])}
    for sp in corpus.get("spans", []):
        for i in sp.get("sub_question_ids") or []:
            if i in table:
                table[i] += 1
    return table

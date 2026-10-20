"""D-055 Option C: the evidence record the L3 seam (vera.answer_contract.examined_full_documents) honours.

Stdlib only, no I/O. It is a frozen dataclass, never a dict, so neither config, env nor a candidate dict can forge it:
only vera.fulldoc_admission.admit builds one. Two lists stay distinct by design:
  examined_full_documents  acquired, verified full documents whose full text was inspected for this question;
  admitted                 the selected passages of those documents that enter reasoning context.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class EvidenceBundle:
    question: str
    examined_full_documents: tuple = ()  # dicts: document_id, canonical_source_id, title, url, doi, licence,
    #                                      licence_decision, text_sha256, chars, chunks_inspected, candidate_spans, admitted_spans
    admitted: tuple = ()  # dicts: span_id, document_id, canonical_source_id, title, url, published, licence, start, end,
    #                       text (internal only, never in a response), text_sha256, relevance
    unusable: tuple = ()  # dicts: document_id, code, reason (never grounded on)
    rejected: tuple = ()  # dicts: span_id, document_id, reason (candidates evaluated and not admitted)
    status: str = "insufficient"  # admitted | insufficient | corpus_unusable
    trace: dict = field(default_factory=dict)  # counts only

    @property
    def grounded(self) -> bool:
        """True only when at least one verified document was inspected AND at least one passage was admitted."""
        return bool(self.examined_full_documents) and bool(self.admitted) and self.status == "admitted"

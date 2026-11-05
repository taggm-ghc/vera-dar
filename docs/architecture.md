# Public architecture

[← Project overview](../README.md)

VERA-DAR explores provenance-grounded context engineering: discovering evidence, evaluating its suitability, constructing context for generation, and examining whether resulting answers are better supported and more useful than direct generation.

## Current question-answering workflow

The public `/ask` path checks a question for safety and then answers from full documents when it can, or returns a reading list when it cannot. The steps, in order (the diagram in the [project overview](../README.md#how-it-works) shows the same flow; [API and data use](api.md#default-behaviour) describes the behaviour as seen by a caller):

1. **Safety check (scope router module).** A rule layer and then a model verdict decide only whether the request is safe to handle (injection, harmful intent, personal advice, rule changes, code-writing requests). Subject matter is never a reason to decline. The router loads at API start and fails closed on bad configuration or an unreadable verdict. A declined request gets a fixed text. Whether the collection can answer is decided after retrieval (steps 2 and 3), never by subject.
2. **Full-document retrieval and admission.** Passages are retrieved from the configured full-document corpus (a verified bundle or the database-backed corpus) and each passage is admitted or rejected with a reason code. This path is enabled in the shipped configuration (`fulldoc.enabled` in config/fulldoc_answer.json) but takes effect only when exactly one corpus source is configured; configuring both is refused and the corpus is reported unusable. No admitted passage means no generated answer.
3. **Acquisition.** When the corpus has no usable passages (an insufficient corpus), the service can try to acquire and vet further full documents, subject to its licence, content and relevance gates. The acquisition is offline-tested only and has not been validated live in production; it also depends on configuration and a kill switch, so whether it runs in a given deployment is not established by this document.
4. **Answer and claim check.** When passages are admitted, an answer with citations is generated from them and each cited claim is checked; unsupported sentences are removed.
5. **Reading list.** When nothing is admitted or acquired, the response is a reading list of candidate documents with links, with no generated answer and no claim check. Abstracts and snippets are not used as evidence. A no-sources or decline result is returned when nothing usable is found.

Only a response with `result_kind: "answer"` is an evidentiary answer. The current service is **not** equivalent to the broader staged research pipeline, which has not been exercised live end to end; some stage-to-stage interfaces remain incomplete.

## Four design objectives

1. **Grounding:** link research claims to relevant observed evidence.
2. **Provenance:** maintain traceability to source records and retrieval context.
3. **Auditability:** expose consequential source-selection and evaluation decisions.
4. **Response quality:** demonstrate that the final answer improves rather than merely accumulating process.

These are objectives and are not all established by current tests.

## Evidence inspection boundary

On the reading-list path the service handles index records and links, not full documents. Some pages are consulted to recover DOI or arXiv identifiers, not used as research-paper evidence. The full-document path is enabled in the shipped configuration (config/fulldoc_answer.json) but produces answers only when a corpus source is configured; only on that path are full documents inspected and passages selected as possible evidence.

Accordingly, reading-list source records do not include definitive document content hashes, immutable full-text versions or precise examined passages, and a reading list should not be described as verification of any original paper.

## Status note (as of 2026-10-29) - Superseded 2026-11-03

*Superseded 2026-11-03: `POST /ask/stream` exists (see [API and data use](api.md#streaming-request-post-askstream)). The text below is kept as history.*

On the main branch there is no `/ask/stream` or `/ingest` route and no progressive stage updates; the interface is a single blocking `/ask` plus `/health`, `/health/dar`, `/version`, `/corpus-summary`, `/trace/*` and owner-key `/memory/*` routes (`main.py`). Progressive updates and the staged retrieval switches are planned or shipped off on branches ahead of main. For the deployment timeline and host plan see the Status section of the [project overview](../README.md).

## Additions of 2026-11-03 (offline-tested, not deployed, live unverified)

- **Evidence trace.** `contract["trace"]` carries a plain-language projection of which passages were selected, cited or not used, built from an allowlist (`vera/evidence_trace.py`, labels in `config/evidence_trace.json`). It is omitted when the label file is missing or invalid. A dev-only view with raw codes exists only when public mode is off.
- **arXiv acquisition.** arXiv is an acquisition provider called once per job, separately from OpenAlex, with its own search-call slot (`config/auto_acquire.json`). The licence is still decided at fetch.
- **Handle recognition.** `hdl.handle.net/<prefix>/<suffix>` and `hdl:` identifiers (prefix not starting `10.`) are recognised as persistent identifiers. No automatic fetch route exists for them.
- **Live statistics (flag-off).** When `config/live_stats.json` `enabled` is true, one anonymous row of enums and counts is written in a background task after a successful result; no question text. `GET /trace/live` (public) and `GET /trace/live/owner` (owner key) read day-bucketed aggregates. It ships off and needs the owner-run draft migration 013, which is not applied.
- **Citations (push 2, offline-tested, not deployed, live unverified).** A pure APA 7 formatter produces `reference`, `in_text` and `characterisation` for public sources and candidates; see [Citations](citations.md).
- **Sub-queries (push 2, same status).** A deterministic decomposer splits multi-entity and comparison questions into parts; adequacy is judged per part and reported in the public trace. Two optional lanes (Official, Current-reporting) run per sub-query. Both ship off; a request does no lane work while they are off. See [Citations](citations.md#search-lanes).

## Reliability and limitations

- Retrieval relevance can be weak, especially when questions are translated into poor search queries.
- The model-based claim checker has an incompletely measured error rate.
- Licence decisions applied to metadata cannot establish article-text reuse rights.
- Historical note: earlier research workflows included owner-reviewed candidate memories. The public demo configuration was described as disabling new memory and corpus persistence; this page does not re-verify that against the current configuration, so treat it as history, not as a current guarantee.
- Comparative performance against leading research assistants is not yet established.

This document deliberately omits private scoring formulas, internal decision structures, orchestration details and other unpublished implementation mechanisms. For measured findings see [Evaluation](evaluation.md); for externally visible behaviour and the endpoint list see [API and data use](api.md); for the project overview see the [README](../README.md).

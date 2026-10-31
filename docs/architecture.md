# Public architecture and research scope

[← Project overview](../README.md)

VERA-DAR explores provenance-grounded context engineering: discovering evidence, evaluating its suitability, constructing context for generation, and examining whether resulting answers are better supported and more useful than direct generation.

## Current question-answering workflow

The public `/ask` path accepts or declines a bounded question and then answers from full documents when it can, or returns a reading list when it cannot. The steps, in order (the diagram in the [project overview](../README.md#how-it-works) shows the same flow; [API and data use](api.md#default-behaviour) describes the behaviour as seen by a caller):

1. **Scope router.** A rule layer and then a model verdict decide whether the question is in scope. The router loads at API start and fails closed on bad configuration. A declined question gets a fixed text. A question the second layer judges off-topic may instead go to inline acquisition (step 3), when acquisition is enabled.
2. **Full-document retrieval and admission.** Passages are retrieved from the configured full-document corpus (a verified bundle or the database-backed corpus) and each passage is admitted or rejected with a reason code. This path is enabled in the shipped configuration (`fulldoc.enabled` in config/fulldoc_answer.json) but takes effect only when exactly one corpus source is configured; configuring both is refused and the corpus is reported unusable. No admitted passage means no generated answer.
3. **Acquisition.** When the corpus has no usable passages, the service can try to acquire and vet further full documents, subject to its licence, content and relevance gates. The acquisition is offline-tested only and has not been validated live in production; it also depends on configuration and a kill switch, so whether it runs in a given deployment is not established by this document.
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

## Status note (as of 2026-10-29)

On the main branch there is no `/ask/stream` or `/ingest` route and no progressive stage updates; the interface is a single blocking `/ask` plus `/health`, `/health/dar`, `/version`, `/corpus-summary`, `/trace/*` and owner-key `/memory/*` routes (`main.py`). Progressive updates and the staged retrieval switches are planned or shipped off on branches ahead of main. For the deployment timeline and host plan see the Status section of the [project overview](../README.md).

## Reliability and limitations

- Retrieval relevance can be weak, especially when questions are translated into poor search queries.
- The model-based claim checker has an incompletely measured error rate.
- Licence decisions applied to metadata cannot establish article-text reuse rights.
- Historical note: earlier research workflows included owner-reviewed candidate memories. The public demo configuration was described as disabling new memory and corpus persistence; this page does not re-verify that against the current configuration, so treat it as history, not as a current guarantee.
- Comparative performance against leading research assistants is not yet established.

This document deliberately omits private scoring formulas, internal decision structures, orchestration details and other unpublished implementation mechanisms. For measured findings see [Evaluation](evaluation.md); for externally visible behaviour and the endpoint list see [API and data use](api.md); for the project overview see the [README](../README.md).

# Public architecture and research scope

[← Project overview](../README.md)

VERA-DAR explores provenance-grounded context engineering: discovering evidence, evaluating its suitability, constructing context for generation, and examining whether resulting answers are better supported and more useful than direct generation.

## Current question-answering workflow

The public `/ask` path accepts or declines a bounded question and searches scholarly index services and the web. **By default it returns a reading list of candidate documents with links; it generates no answer and runs no claim check**, and abstracts or snippets are not used as evidence. A no-sources or decline result is returned when nothing usable is found.

An optional full-document path can produce a grounded answer: verified full documents are acquired, relevant passages are admitted, an answer with citations is generated from them, and cited claims are checked. That path is enabled only when `fulldoc.enabled` is true in config/fulldoc_answer.json and a verified bundle is configured via VERA_FULLDOC_BUNDLE_DIR.

The current service is **not** equivalent to the broader staged research pipeline, which has not been exercised live end to end; some stage-to-stage interfaces remain incomplete.

## Four design objectives

1. **Grounding:** link research claims to relevant observed evidence.
2. **Provenance:** maintain traceability to source records and retrieval context.
3. **Auditability:** expose consequential source-selection and evaluation decisions.
4. **Response quality:** demonstrate that the final answer improves rather than merely accumulating process.

These are objectives and are not all established by current tests.

## Evidence inspection boundary

By default the service handles index records and links, not full documents. Some pages are consulted to recover DOI or arXiv identifiers, not used as research-paper evidence. An optional verified full-document path exists and is enabled in the shipped configuration (config/fulldoc_answer.json) but produces answers only when a corpus source is configured; only on that path are full documents inspected and passages selected as possible evidence.

Accordingly, default-path source records do not include definitive document content hashes, immutable full-text versions or precise examined passages, and a reading list should not be described as verification of any original paper.

## Reliability and limitations

- Retrieval relevance can be weak, especially when questions are translated into poor search queries.
- The model-based claim checker has an incompletely measured error rate.
- Licence decisions applied to metadata cannot establish article-text reuse rights.
- Earlier research workflows included owner-reviewed candidate memories; the public demo configuration disables new memory and corpus persistence.
- Comparative performance against leading research assistants is not yet established.

This document deliberately omits private scoring formulas, internal decision structures, orchestration details and other unpublished implementation mechanisms. For measured findings see [Evaluation](evaluation.md); for externally visible behaviour see [API and data use](api.md).

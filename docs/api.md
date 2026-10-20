# API and data use

[← Project overview](../README.md)

VERA-DAR provides a FastAPI service for bounded research questions. This page describes the public behaviour of the endpoints; it is not a statement about any particular deployed release.

## Endpoints

| Endpoint | Purpose | Authentication |
|---|---|---|
| `GET /health` | Basic service health; includes a coarse `dar` readiness field (`ready`, `degraded`, `unavailable`, `disabled` or `unknown`) | None |
| `GET /health/dar` | Detailed readiness | `VERA_API_KEY` |
| `GET /version` | Exposes build commit identifier | None |
| `POST /ask` | Research question: reading list, or grounded answer on the optional full-document path | `VERA_API_KEY` |
| `GET /corpus-summary` | Aggregated summary of admitted source records | `VERA_API_KEY` |
| `GET /memory/pending` | Pending findings for owner review | Owner key |
| `POST /memory/{claim_id}/confirm` | Confirm a pending finding | Owner key |
| `POST /memory/{claim_id}/reject` | Reject a finding | Owner key |

Endpoints that require `VERA_API_KEY` expect it in the `X-API-Key` header.

The owner-only endpoints require `VERA_OWNER_KEY` in the `X-Owner-Key` header; without that configuration, the owner routes remain unavailable. The public demonstration does not expose owner actions.

## Research request

`POST /ask` accepts a JSON body such as:

```json
{"question": "What does recent research say about this topic?", "mode": "normal"}
```

The response always carries `answer`, `model`, `tokens_used` and `cost_usd`, and a `result_kind` that says what kind of result it is. **Only `result_kind: "answer"` is an evidentiary answer.** Other kinds (for example a reading list, discovery result or decline) are not answers. Further fields include:

- `candidates`: reading-list entries (document links) for non-answer results.
- `contract`: the enabled result layers and any fallback reason.
- `sources` and `traced_sources`: source records, and records traced from links but not used.
- `claim_check`: a summary of the claim check, present when an answer was checked.
- Evidence fields for the full-document path (`evidence_status`, `examined_full_documents`, `admitted_evidence`, `citations`), omitted otherwise.

By default `/ask` returns a reading list: it searches scholarly indexes and the web, lists documents that may be relevant, and generates no answer and no claim check. A grounded answer is produced only on the optional verified full-document path, which is enabled only when `fulldoc.enabled` is true in config/fulldoc_answer.json and a verified bundle is configured via VERA_FULLDOC_BUNDLE_DIR. The precise schema is defined by the application.

## Provenance and evidence limits

Source entries include bibliographic and retrieval metadata. On the default path they do **not** establish that the complete paper was read, identify an immutable document version, or supply page-level provenance. Labels implying full-document verification would exceed the inspection performed. Abstracts and snippets are not used as evidence.

Recorded licence decisions are based on licence information attributed to indexing services; metadata licences are not licences to reproduce the article. Rights have not been confirmed with individual rights holders.

## Data handling

Questions are transmitted to third-party AI and search services, each with its own data-handling practices. The demonstration configuration disables newly persisted corpus additions, discovery notes, memory writes, and the DuckDuckGo query cache. That does not establish the retention behaviour of external providers or the hosting platform. Records written by earlier configurations are not automatically erased by switching to demonstration mode. The disclosure text shown to visitors is quoted in the [README](../README.md#privacy-and-disclosure).

## Usage safeguards

The application includes authorization checks and configurable public request/cost caps. In-memory caps reset on service restart and are not equivalent to a centralized quota service. Keys belong in local environment configuration or the hosting provider's secret settings, never in version control.

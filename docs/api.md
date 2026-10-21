# API and data use

[← Project overview](../README.md)

VERA-DAR provides a FastAPI service for bounded research questions. This page describes the public behaviour of the endpoints; it is not a statement about any particular deployed release.

## Endpoints

| Endpoint | Purpose | Authentication |
|---|---|---|
| `GET /health` | Basic service health; returns only `status` and a coarse `dar` readiness field (`ready`, `degraded`, `unavailable`, `disabled` or `unknown`) | None |
| `GET /health/dar` | Detailed readiness, including a `fulldoc` block for the full-document path | `VERA_API_KEY` |
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

- `outcome`: the single response-level disposition (see below).
- `candidates`: reading-list entries (document links) for non-answer results.
- `contract`: the enabled result layers and any fallback reason.
- `sources` and `traced_sources`: source records, and records traced from links but not used.
- `claim_check`: a summary of the claim check, present when an answer was checked.
- Evidence fields for the full-document path (`evidence_status`, `corpus_source`, `examined_full_documents`, `admitted_evidence`, `citations`), omitted otherwise. `corpus_source` is `{"kind": "db"}` or `{"kind": "bundle"}` and is present only when the full-document corpus loaded successfully (it says where the corpus was loaded from); it is omitted on failure and on web-only responses.

### The `outcome` field

`outcome` is a top-level field on every `/ask` response. It is derived from `result_kind` and the fallback reason (if any):

| `outcome` | Meaning |
|---|---|
| `answer` | An evidentiary answer (`result_kind` is `answer`). |
| `insufficient_evidence` | No answer could be supported. Any fallback reason not listed below, and any unknown value, is reported this way. |
| `reading_list` | A reading list with at least one candidate document. |
| `no_sources` | A reading-list result with no candidate documents. |
| `corpus_unusable` | The full-document corpus could not be used. |
| `decline` | The request was declined. |

`outcome` differs from `result_kind`, which names the kind of result (answer, reading list, discovery, decline), and from `claim_check.outcome`, which is the claim checker's own internal summary. Use `outcome` for the overall disposition and `result_kind` to decide whether the text is an evidentiary answer.

### Health details

`GET /health/dar` requires the API key. Besides the readiness details it includes a `fulldoc` block with codes and counts only: `enabled`, `bundle_configured`, `status` (`disabled`, `not_configured`, `ok`, `unusable` or `error`), `n_acquired`, `n_unusable`, `pins_ok` and `source`. Public `GET /health` returns only `status` and `dar`.

## Default behaviour

By default `/ask` returns a reading list: it searches scholarly indexes and the web, lists documents that may be relevant, and generates no answer and no claim check. A grounded answer is produced only on the optional verified full-document path, which is enabled only when `fulldoc.enabled` is true in config/fulldoc_answer.json and a verified bundle is configured via VERA_FULLDOC_BUNDLE_DIR. The precise schema is defined by the application.

## Provenance and evidence limits

Source entries include bibliographic and retrieval metadata. On the default path they do **not** establish that the complete paper was read, identify an immutable document version, or supply page-level provenance. Labels implying full-document verification would exceed the inspection performed. A reading-list entry for a text that is stored but was not inspected for the answer is labelled "full document stored (not inspected for this answer)". A document that declares any text kind other than the configured full-document kind (for example an abstract only, or an unrecognised label) is refused rather than treated as a full document. Abstracts and snippets are not used as evidence.

Recorded licence decisions are based on licence information attributed to indexing services; metadata licences are not licences to reproduce the article. Rights have not been confirmed with individual rights holders.

## Data handling

Questions are transmitted to third-party AI and search services, each with its own data-handling practices. When the optional full-document path is enabled, up to six selected passages (text, title, link) and the draft answer sentences are transmitted too. The demonstration configuration disables newly persisted corpus additions, discovery notes, memory writes, and the DuckDuckGo query cache. That does not establish the retention behaviour of external providers or the hosting platform. Records written by earlier configurations are not automatically erased by switching to demonstration mode. The disclosure text shown to visitors is quoted in the [README](../README.md#privacy-and-disclosure).

## Usage safeguards

The application includes authorization checks and configurable public request/cost caps. In-memory caps reset on service restart and are not equivalent to a centralized quota service. Keys belong in local environment configuration or the hosting provider's secret settings, never in version control.

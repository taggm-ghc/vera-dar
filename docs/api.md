# API and data use

[← Project overview](../README.md)

VERA-DAR provides a FastAPI service for bounded research questions. This page describes the public behaviour of the endpoints; it is not a statement about any particular deployed release. The route list below matches the routes defined in `main.py` as of 2026-10-29; there is no `/ingest` route. `POST /ask/stream` is the streaming variant of `/ask` (newline-delimited JSON; see [Streaming request](#streaming-request-post-askstream)).

## Endpoints

| Endpoint | Purpose | Authentication |
|---|---|---|
| `GET /health` | Basic service health; returns only `status` and a coarse `dar` readiness field (`ready`, `degraded`, `unavailable`, `disabled` or `unknown`) | None |
| `GET /health/dar` | Detailed readiness, including a `fulldoc` block for the full-document path | `VERA_API_KEY` |
| `GET /version` | Exposes build commit identifier (`commit`, `short`) | None |
| `GET /trace/live` | Anonymous live aggregate statistics (flag-off; see [Live statistics](#live-statistics-trace-live)) | None |
| `GET /trace/live/owner` | Same aggregates plus token and cost totals | Owner key |
| `GET /trace/public` | Public TRACE evaluation view (historical aggregate, no secrets) | None |
| `GET /trace/private` | Private TRACE evaluation view | Separate trace-viz key (`VERA_TRACE_VIZ_KEY`) |
| `POST /ask` | Research question: grounded answer from admitted full-document passages, otherwise a reading list or decline (see [Default behaviour](#default-behaviour)) | `VERA_API_KEY` |
| `POST /ask/stream` | Same request body as `POST /ask`; answers with newline-delimited JSON progress lines, then one result line (see below) | `VERA_API_KEY` |
| `GET /corpus-summary` | Agentic digest of the stored corpus (CORPUS-DIGEST-1104): counts, model-written digest checked against recorded facts, subject topics with document shares (topics covering 80% of sampled documents, computed in code), and a generated description of current coverage | `VERA_API_KEY` |
| `GET /trace/request/{id}` | One request's trace (codes and counts; see [Request trace](#endpoints)) | Owner key |
| `GET /memory/pending` | Pending findings for owner review | Owner key |
| `POST /memory/{claim_id}/confirm` | Confirm a pending finding | Owner key |
| `POST /memory/{claim_id}/reject` | Reject a finding | Owner key |

Endpoints that require `VERA_API_KEY` expect it in the `X-API-Key` header.

**Request trace.** When request tracing is on (`config/request_trace.json`, `enabled`; it ships on), `/ask` and `/ask/stream` responses, including error responses, carry an `X-VERA-Trace-Id` header: a server-generated 32-character hex id, never a client value. The id is for owner diagnosis only. It never appears in a response body or in the UI. The trace holds stage events with codes, counts and size-capped, scrubbed details, never page text. It is written to log lines and, with `db_sink.enabled`, to the database (migration 015, owner-applied). The question text is kept owner-only (not in host logs while `log_sink.include_question` is false) and purged after 30 days by an owner-run script. Recent traces can be read through the owner route `/trace/request/{id}` (owner key; recent in-memory window; the database view is behind `db_sink.owner_route_db_view`, which ships false). Lane trace events: one `lane` event per request, `done` or `skipped`, with counts (`items`, `pages`, `links`) and closed per-portal tokens (`called` or `skipped`, plus a reason such as `off`, `no_terms`, `circuit`, `spacing`, `throttled`, `failed` or `ok`). The reading-list fallback writes `reading_list` events (`start`, then `done` with a candidate count).

The owner-only endpoints require `VERA_OWNER_KEY` in the `X-Owner-Key` header; without that configuration, the owner routes remain unavailable. The public demonstration does not expose owner actions.

## Streaming request (`POST /ask/stream`)

Contract version `1.3`. The body is the `POST /ask` body. Authentication (401) and the public-mode usage caps (429) are checked before streaming starts, and one POST uses one cap slot. The response is `application/x-ndjson` with `X-Accel-Buffering: no` and `Cache-Control: no-cache`; each line is flushed as it is written. Nothing is stored server side beyond the request, except that when the owner enables live stats, one anonymous record per answer is kept (time rounded to the hour; no question text or identifiers); while the flag is off, nothing changes.

Lines (one JSON object each):

- `{"v":"1.3","type":"event","seq":n,"ts":"<iso>","stage":S,"state":T,"counts":{k:int}}`. Stages: `received scope corpus corpus_verdict search acquire write check retry done`. The `acquire` stage also has the states `limited` and `busy`. Counts are integers only: `inspected admitted found checked usable budget stored target queued attempt shortlisted held step ceiling why`. `why` is a closed code: 1 no more open sources found, 2 sources busy, 3 the day's search limit reached, 4 stalled, 5 the answering service at its limit. Code 2 is reported only when a provider is really unavailable: a provider cooldown, an open or half-open circuit breaker, or this deployment's daily cap (`rate_limit.PROVIDER_BUSY_REASONS`). The service's own concurrency or pacing limiter, or a caller deadline, never produces code 2; those cases report 1 or 4. No provider or portal names, hosts, URLs or titles are carried.
- While the server waits or runs a slow step, an `event` line (state `update`, on the last lifecycle stage seen) and each `hb` line also carry the current activity as integers: `act` (closed code: 1 rate-limit wait, 2 polite wait, 3 slow read, 4 retry, 5 busy fallback, 6 connection, 7 searching current news reporting, 8 preparing a reading list), optionally `wait_s`, `k`, `n`, and on `hb` lines `elapsed` (whole seconds in that activity). Without an activity, `hb` lines carry none of these. Activities 7 and 8 cover phases that have no stage events of their own (the Current-reporting lane pass and the reading-list fallback). While one is running, the activity is re-sent every `stream.activity_tick_s` seconds (`config/progress.json`, provisional value 5, uncalibrated), with `elapsed` in whole seconds, so the stream is never silent for longer than that interval.
- `{"v":"1.3","type":"hb","seq":n,"elapsed_s":int}` when no other line was written for `stream.heartbeat_s` seconds.
- `{"v":"1.3","type":"result","status":200,"body":<the POST /ask response>}` exactly once, last. A failure is `{"v":"1.3","type":"result","status":429|500|502|503,"kind":"busy|error|provider_error|unavailable"}`; it never contains error text. A client that disconnects gets no result.

There is no total deadline on the stream; a run ends when it is met, resource limited, or stalled (no progress event for 300 s, `live.stall_s` in `config/auto_acquire.json`). A heartbeat line is sent every 10 s of silence (`stream.heartbeat_s`). If the client disconnects, the server finishes the in-flight step, keeps documents already admitted, starts nothing new and writes no answer.

Flags: `stream.enabled` in `config/progress.json` (ships true; when false the route answers 404). `VERA_PROGRESS_DISABLED=1` forces the route off. The polling side channel (`GET /ask/progress/{rid}`) is governed separately by `enabled` and stays off.

## Research request

`POST /ask` accepts a JSON body such as:

```json
{"question": "What does recent research say about this topic?", "mode": "normal"}
```

The response always carries `answer`, `model`, `tokens_used` and `cost_usd`, and a `result_kind` that says what kind of result it is. **Only `result_kind: "answer"` is an evidentiary answer.** Other kinds (for example a reading list, discovery result or decline) are not answers. Further fields include:

- `outcome`: the single response-level disposition (see below).
- `candidates`: reading-list entries (document links) for non-answer results. A candidate may carry `metadata_licence`, the licence id the indexing service declares for its own record (at most 64 characters). It is not a licence for the article and is omitted when absent.
- `contract`: the enabled result layers and any fallback reason.
- `sources` and `traced_sources`: source records, and records traced from links but not used.
- `claim_check`: a summary of the claim check, present when an answer was checked.
- Evidence fields for the full-document path (`evidence_status`, `corpus_source`, `examined_full_documents`, `admitted_evidence`, `citations`), omitted otherwise. `corpus_source` is `{"kind": "db"}` or `{"kind": "bundle"}` and is present only when the full-document corpus loaded successfully (it says where the corpus was loaded from); it is omitted on failure and on web-only responses.
- Research fields (`triggers`, `freshness`, `research_outcome`), present only when the freshness or research step ran. `triggers` lists which research triggers fired (`insufficient`, `stale` or `freshness_unknown`); `research_outcome` reports what was tried, what was admitted and why candidates were not.
- On source and candidate entries, `acquisition` is `full_document_acquired` for stored full documents.
- `answer_status`: how complete the answer is. One of `complete`, `partial`, `unassessed`, `not_answered` or `declined`.
  - `complete` is reported only on a positive completeness signal from the answer checker. A passed citation check alone never makes an answer `complete`.
  - `partial` means the answer covers only part of the question, or is an off-topic answer. A completeness value the server does not recognise is also reported as `partial`.
  - `unassessed` means an answer exists but no completeness signal was recorded.
  - `not_answered` means no evidentiary answer was given (for example, a reading list). `declined` means the request was declined.
- `limitations`: list of short plain-language statements of limits on the evidence and coverage (for example, that some candidate documents were not acquired). Empty when there are none.
- `process_notes`: list of short plain-language statements of the research and verification steps taken. Empty when there are none.

`limitations` and `process_notes` are kept out of `answer` only when `answer_notes_separate` is on (the shipped setting). With it on, `answer` holds the answer text and the notes appear in those two lists. Notes of an unrecognised kind appear in neither list and do not appear in `answer` either (fail closed). On a result that is not an evidentiary answer (for example a reading list), the notice explaining why is the answer itself and stays in `answer` in both modes; only a separate background-search note is moved to `process_notes` when the flag is on. With the flag off, the legacy text is restored: notes are concatenated into `answer` and both lists are empty. `answer_notes_separate` is a top-level key in `config/ask-provider-chain.json`; only the JSON boolean `true` turns it on, and any other value, a missing key or an unreadable file means off. The file is read on each request.

### Additional contract fields (2026-11-03; offline-tested, not deployed, live unverified)

- `contract.trace`: a plain-language evidence trace (passages selected, cited or not used, with fixed-label reasons; no raw codes, relevance scores or document ids in public mode). Omitted when its label file is missing or invalid.
- `contract.no_answer_cause`: `{code, text}` (text at most 300 characters), present only on a non-answer result. Codes seen in code: `no_passages`, `corpus_unusable`, `verification_unavailable`, `premise_unverified`, `no_supported_answer`, `reading_list`.
- Citation records built for a full-document answer carry `supports`, the answer sentence each citation backs; the evidence trace uses it. The `/ask` `citations` field keeps its existing keys and does not include it.
- `contract.research.portals_attempted` and `contract.research.portals_completed`: integers counting acquisition portals tried (not skipped) and completed, on a non-answer result.
- `sources[]` and `candidates[]` entries carry `reference`, `in_text` and `characterisation` (formatted APA 7 strings; null when the style flag is `legacy`). See [Citations](citations.md).
- `contract.trace.adequacy.subqueries`: list of `{label, verdict}` when the question was split into two or more sub-queries; absent otherwise. A source reported by a third party is labelled "reported by the source, not independently verified". When a lane is enabled (the Official lane ships off; the Current-reporting lane ships on, see below), the acquisition record in the response gains a `lanes` object with `links` (each with `lane`, `label`, `title`, `url` and, where known, `published`, `outlet` or `entity`, `read`), `n_read`, `credits` and `scope`. It carries counts and links only, never page text. Lane labels are "Official" and "Reported". See [Citations](citations.md#search-lanes).

## Live statistics (`/trace/live`)

Ships off (`config/live_stats.json`, `enabled: false`); with the flag off `GET /trace/live` returns `{"enabled": false, "total": 0}`. When on, it needs the owner-run draft migration 013 (not applied). The query parameter `days` (default 7) is capped by `window_days_max`. The public shape is `enabled`, `total`, `by_result_kind`, `by_day` and `typical_latency_ms`. Counts below the minimum cell size (`min_count`, provisional value 5, not yet calibrated on real traffic) are shown as `"<5"`, with complementary suppression of other cells where needed. Results are cached for `cache_ttl_s` seconds. Rows hold enums and counts only: no question text, client id or IP. `GET /trace/live/owner` (owner key) adds `tokens_total` and `cost_usd_total`.

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

`GET /health/dar` requires the API key. Besides the readiness details it includes a `fulldoc` block with codes and counts only: `enabled`, `bundle_configured`, `status` (`disabled`, `not_configured`, `loading`, `ok`, `unusable` or `error`; `loading` means the startup warm-up of the database-backed corpus is still running, so check again later), `n_acquired`, `n_unusable`, `pins_ok` and `source`. Public `GET /health` returns only `status` and `dar`.

## Default behaviour

This section is the canonical description of what `/ask` does by default. It describes the shipped configuration in this repository, not any particular deployed build.

1. **Safety check.** The pre-retrieval router (`config/scope_router.json`) is safety-only: subject matter is never a reason to decline, and there is no fixed Scope. It loads when the API starts and fails closed if its configuration is invalid; a classifier reply that does not match the schema also fails closed. A declined question gets a fixed, neutral decline text. *Superseded 2026-11-05: the earlier topical scope check and its off-topic routing.*
2. **Full-document answering.** Full-document answering ships enabled: `fulldoc.enabled` is `true` in `config/fulldoc_answer.json`. It takes effect only when a corpus source is configured, either a verified bundle directory named by `VERA_FULLDOC_BUNDLE_DIR` or a database-backed corpus named by the run id in `VERA_FULLDOC_DB_RUN_ID`. Configuring both is refused: the corpus is reported unusable rather than one source being silently preferred. With no corpus source configured, the path is off and `/ask` returns reading lists only.
3. **Admission.** An answer is generated only from admitted passages, with a claim check on each cited sentence. No admitted passage means no generated answer: the response is a reading list or a decline instead.
4. **Acquisition.** Acquisition is enabled in `config/auto_acquire.json`, but takes effect only when both database URLs (`VERA_DB_URL_RO`, `VERA_DB_URL_WO`) are set and the kill switch `VERA_AUTOACQ_DISABLED` is not set to a true-like value. Acquired documents are served only alongside the database-backed corpus (bundle directory unset, run id set). Acquisition, including inline acquisition within the request, is offline-tested only; its live production behaviour has not been verified.
5. **Reading list fallback.** Otherwise the response lists documents that may be relevant, with links, from scholarly indexes and the web. VERA-DAR has not read those documents.

The precise schema is defined by the application.

## Provenance and evidence limits

Source entries include bibliographic and retrieval metadata. On a reading list they do **not** establish that the complete paper was read, identify an immutable document version, or supply page-level provenance. Labels implying full-document verification would exceed the inspection performed. A reading-list entry for a text that is stored but was not inspected for the answer is labelled "full document stored (not inspected for this answer)". A document that declares any text kind other than the configured full-document kind (for example an abstract only, or an unrecognised label) is refused rather than treated as a full document. Abstracts and snippets are not used as evidence.

Recorded licence decisions are based on licence information attributed to indexing services; metadata licences are not licences to reproduce the article. Rights have not been confirmed with individual rights holders.

## Data handling

Questions are transmitted to third-party AI and search services, each with its own data-handling practices. When the optional full-document path is enabled, up to six selected passages (text, title, link) and the draft answer sentences are transmitted too. The demonstration configuration disables newly persisted corpus additions, discovery notes, memory writes, and the DuckDuckGo query cache. That does not establish the retention behaviour of external providers or the hosting platform. Records written by earlier configurations are not automatically erased by switching to demonstration mode. The disclosure text shown to visitors is quoted in the [README](../README.md#privacy-and-data-handling). The Current-reporting lane ships on (`config/reporting_lane.json` `enabled` true; `outlet_query` `all`; `emit_link_only` true, so items from outlets that are not cleared for reading are shown as links only, never read). The Official lane ships off (`config/subquery.json` `official_lane.enabled` false). When a lane is on, the following applies. Search terms derived from your question may also be sent to GDELT (a news index) and Hacker News search (Algolia). VERA-DAR may read public web pages from vendors and news outlets to answer; these pages are used only for your answer, and page text is not saved. Pages read for one answer are not licence-cleared or stored. GDELT is credited: [GDELT](https://www.gdeltproject.org/).

## Usage safeguards

The application includes authorization checks and configurable public request/cost caps. In-memory caps reset on service restart and are not equivalent to a centralized quota service. Keys belong in local environment configuration or the hosting provider's secret settings, never in version control.

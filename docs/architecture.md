# Public architecture

[← Project overview](../README.md)

VERA-DAR explores provenance-grounded context engineering: discovering evidence, evaluating its suitability, constructing context for generation, and examining whether resulting answers are better supported and more useful than direct generation.

## Current question-answering workflow

The public `/ask` path checks a question for safety and then answers from full documents when it can, or returns a reading list when it cannot. The steps, in order (the diagram in the [project overview](../README.md#how-it-works) shows the same flow; [API and data use](api.md#default-behaviour) describes the behaviour as seen by a caller):

1. **Safety check (scope router module).** A rule layer and then a model verdict decide only whether the request is safe to handle (injection, harmful intent, personal advice, rule changes, code-writing requests). Subject matter is never a reason to decline. The router loads at API start and fails closed on bad configuration or an unreadable verdict. A declined request gets a fixed text. Whether the collection can answer is decided after retrieval (steps 2 and 3), never by subject.
2. **Full-document retrieval and admission.** Passages are retrieved from the configured full-document corpus (a verified bundle or the database-backed corpus) and each passage is admitted or rejected with a reason code. This path is enabled in the shipped configuration (`fulldoc.enabled` in config/fulldoc_answer.json) but takes effect only when exactly one corpus source is configured; configuring both is refused and the corpus is reported unusable. No admitted passage means no generated answer.
3. **Acquisition.** Whether the collection is sufficient is decided after retrieval and admission, not before. When it is not (no usable passages, or the answer attempt ends without a supported answer, including an unverified premise), the service tries to acquire and vet further full documents in acquisition rounds, each round aiming for at least 6 new stored full texts (`bounds.acq_round_target_new_docs`, `vera/acq_rounds.py`; if fewer exist that is a resource limit, not a failure). Documents must pass the licence allowlist (`config/licence_allowlist.json`, Deny All: a licence not listed active is not storable), content and relevance gates. Acquisition is offline-tested only and has not been validated live in production; it also depends on configuration and the `VERA_AUTOACQ_DISABLED` kill switch, so whether it runs in a given deployment is not established by this document.
4. **Answer and claim check.** When passages are admitted, an answer with citations is generated from them and each cited claim is checked; unsupported sentences are removed.
5. **Reading list.** When nothing is admitted or acquired, the response is a reading list of candidate documents with links, with no generated answer and no claim check. Abstracts and snippets are not used as evidence. A no-sources or decline result is returned when nothing usable is found.

Only a response with `result_kind: "answer"` is an evidentiary answer. The current service is **not** equivalent to the broader staged research pipeline, which has not been exercised live end to end; some stage-to-stage interfaces remain incomplete.

## Additions of 2026-11-04 to 2026-11-05 (offline-tested; live behaviour unverified)

- **Request trace.** `vera/request_trace.py` records a per-request diagnostic trace (`config/request_trace.json`). Migration 015 was applied by the owner on 2026-11-04 (owner-reported; agents have not seen the verification output) and `db_sink` is on in config. The question text is owner-only and kept 30 days (purged by an owner-run script under `work/PERF-MISS-1104/trace/`); the trace id is never shown in the UI. The owner-route DB view stays off (`owner_route_db_view` false). Whether rows are written in the deployed service is untested live.
- **Shipped-off additions (2026-11-05; available, off by default).** Trace retention purge (`vera/retention_purge.py`, `retention_purge.enabled` in `config/request_trace.json`; needs migration 018 draft); topic tagging (`vera/topic_tagging.py`, `config/topic_tagging.json`; needs 017 drafts); DB-backed list loader (`vera/list_tables.py`, `config/list_tables.json` `read_from_db`; needs 016, staged in `work/`). Details in [vera-design.md](vera-design.md). Parallel `/ask` lookups with per-host serialisation and Crawl-delay are on (offline-tested). *Superseded in part 2026-11-08 (DOCS-1108): `config/request_trace.json` now has `retention_purge.enabled` true (commit 6ef1287, OD-1105-PURGE-ENABLE), and migration 018 was applied by the owner on 2026-11-08 (owner-run apply; not independently checked against the database), so the automatic purge is enabled; status is owned by `db/README.md`.*
- **Corpus digest.** `vera/corpus_description.py` builds a descriptive digest of the collection: facts by SQL, a bounded title sample, model clustering and drafting, then a claim-level check in which code drops every sentence the checker does not support or that states an unrecorded property or number; coverage text is built by code. The latest digest is stored in the database (`vera/digest_store.py`, table `vera_vjay.corpus_digest_state`, flag `digest_store.enabled` true in config). No fixed Scope text is displayed.
- **Licence allowlist.** `config/licence_allowlist.json` is the seed and fallback for a Deny All policy (owner decision 2026-11-04). The database table that would supersede it (draft 016) is not among the pending migrations in `db/migrations/pending/` at this revision; treat the file as authoritative.
- **Current-reporting (news) lane.** At HEAD `config/reporting_lane.json` has `enabled` true, GDELT as the only fallback-order provider and 18 listed outlets; the working tree additionally sets `emit_link_only` true and `outlet_query` "all" (owner decision 2026-11-05). Pages are read only for outlets with an allow verdict and passing the page-access gates. Offline-tested only; not validated live against GDELT.
- **Official-sources lane.** Built (`vera/official_lane.py`), ships off (`official_lane.enabled` false, `site_search` false). Since 2026-11-06 (H5) the registry also has FEC and Census Bureau rows (paths and robots.txt unverified) and a suspended EAC row; the flag stays off.
- **News source rules and link-only items (2026-11-06, NEWS-BATCH-2).** VOA is read only for VOA-original pages, Wikinews from mainspace only (an archive), ProPublica and The Conversation are excluded, PBS is link-only. Wikipedia-citation links appear as link-only items (NT-3, `vera/lane_evidence.py`). `vera/news_access.py` (H19) is offline, not yet wired into the lane, and link-only until NC-BASIS-1 is recorded. H20 repository-PDF route: config only, flag off, Zenodo the only listed host (status pending).
- **Owner scripts.** Release verification and deploy checks live under `work/RELEASE-VERIFY-1104/` (`run_release_verify.sh`, `owner_release_gate.sh`, `owner_deploy_check.sh`).

## Additions of 2026-11-09 (offline-tested; live behaviour unverified; flags on from this release)

Retrieval and acquisition changed as follows (details and bounds in the [README](../README.md) "Update 2026-11-09" paragraph; values from `config/` at c3aabb1):

- *Stored corpus.* No cap on stored documents; compact BM25 index; newer rows are appended to a fork of the serving index.
- *Acquisition.* Each question runs one inline acquisition aiming at 6 new documents, one job per question against 120 jobs/hour and 2000/day; a portal page can lead to one further eligible portal page (`max_hops` 2), and an on-question reference may lead to its original host through the unchanged per-hop gates (public-address check, robots.txt, AI-use signals, licence, rate limit); content from an original host is used for that answer only and is not stored.
- *Multi-question prompts.* Split on grammar only (at most 3 questions); each question is answered or refused on its own.
- *Retrieval.* Filler words are dropped whole-word only, sparing proper-name runs and quoted text. With hybrid retrieval on, BM25 candidates and an embedding list (Cloudflare Workers AI bge-m3, vectors from the chunk-taxonomy job) are fused by reciprocal rank fusion, and up to 48 passages of the top 3 partitions are re-ranked at ask time. Any embedding failure, timeout, lock wait or budget stop falls back to word-only search.
- *Chunk taxonomy.* A mechanical background job (embeddings, seeded clustering, nearest-center assignment, c-TF-IDF names) bounded by a neuron budget and a memory guard; migration 024 creates its tables and 022 gates its write mode (nodes, anchors, links; without 022 it only computes); both applied 2026-11-09, see `db/README.md`.

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
- **Sub-queries (push 2, same status).** A deterministic decomposer splits multi-entity and comparison questions into parts; adequacy is judged per part and reported in the public trace. Two optional lanes (Official, Current-reporting) run per sub-query. Superseded 2026-11-05: the Current-reporting lane is now on in config (see the additions above); the Official lane ships off. See [Citations](citations.md#search-lanes).

## Reliability and limitations

- Retrieval relevance can be weak, especially when questions are translated into poor search queries.
- The model-based claim checker has an incompletely measured error rate.
- Licence decisions applied to metadata cannot establish article-text reuse rights.
- Historical note: earlier research workflows included owner-reviewed candidate memories. The public demo configuration was described as disabling new memory and corpus persistence; this page does not re-verify that against the current configuration, so treat it as history, not as a current guarantee.
- Comparative performance against leading research assistants is not yet established.

This document deliberately omits private scoring formulas, internal decision structures, orchestration details and other unpublished implementation mechanisms. For measured findings see [Evaluation](evaluation.md); for externally visible behaviour and the endpoint list see [API and data use](api.md); for the project overview see the [README](../README.md).

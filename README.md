# VERA-DAR

**Verifiable Evidence-based Research Answers**

VERA-DAR is an experimental research assistant that answers a question only from full-text documents it has checked, and shows which passage supports each sentence. When it cannot do that, it says so and gives a reading list of links instead of a guess. It was built during the AI engineering bootcamp capstone (TAI Labs Agentic AI Engineering Bootcamp, 2026).

## Status

**As of 2026-10-30:** the course ended on 2026-10-25 (the 2026-10-23 Demo Day candidate is history). The production deployment is planned to run through 2026-11-30 (owner decision OD-1027, 2026-10-27; this supersedes the earlier end date). Render is the interim host; a non-Render host is planned. The deployed build is shown in the interface footer. A FastAPI API and a Streamlit UI (two Render services) and a Postgres database work end to end in offline tests. The release gate (`scripts/release_gate.py`, see [Release checks](docs/release-checks.md)) has **not** passed for `main`: the latest recorded gate run, on a candidate commit ahead of `main`, failed check (a2) (live catalog evidence). The discovery-to-answer acquisition described below is offline-tested only; it has **not been validated live in production**, and the production API build may predate it unless the owner has deployed it. The claim that VERA-DAR beats leading language models **has not been demonstrated**. No walkthrough recording was known to exist as of 2026-10-22 (not re-checked since). No measurement of the current system has been published (see [Evaluation](#evaluation-what-has-been-measured)).

## The problem

Language models write fluent research answers without showing how the evidence was found or read. A citation list does not prove the sources were examined, or that the answer follows from them.

VERA-DAR asks a narrower question: if the system acquires a full document, admits only the passages that bear on the question, and checks each answer sentence against the passage it cites, is the result easier to trust and to audit? Two things are needed: traceable evidence decisions, and an answer that is actually better. Neither substitutes for the other.

## How it works

```mermaid
flowchart TD
    UI["Streamlit UI<br/>streamlit_app.py, pages/"] -->|"X-API-Key"| API["FastAPI<br/>main.py: POST /ask"]
    API --> CAPS["Public-mode caps<br/>vera/public_mode.py"]
    CAPS --> SCOPE["Scope router<br/>vera/scope_router.py<br/>layer 1: rules, layer 2: LLM verdict"]
    SCOPE -->|"declined"| DECL{"layer-2 off_topic?"}
    DECL -->|"no"| REFUSE["Fixed decline text"]
    DECL -->|"yes, and acquisition on"| INL["Inline acquisition,<br/>else reading list"]
    SCOPE -->|"accepted"| FD["Full-document path<br/>fulldoc_corpus, fulldoc_admission"]
    FD --> ADMIT{"admitted<br/>passages?"}
    ADMIT -->|"yes"| ANS["Answer from passages<br/>fulldoc_answer"]
    ANS --> CC["Claim check per sentence<br/>claim_check"]
    CC --> OUT["Cited answer<br/>(unsupported sentences removed)"]
    ADMIT -->|"no"| INL
    INL -->|"documents pass gates"| ANS
    INL -->|"none"| RL["Reading list with links<br/>search_providers: arXiv, OpenAlex,<br/>DuckDuckGo lite"]
    ADMIT -.->|"insufficient"| BG["Background acquisition<br/>auto_acquire"]
    BG --> STORE[("Postgres:<br/>full-text corpus")]
    FD --- STORE
    LLM["Groq models (provider chain)"] -.-> SCOPE
    LLM -.-> ANS
    LLM -.-> CC
```

What each step does:

1. **Scope check.** Layer 1 is a deterministic rule set (`config/scope_router.json`) that hard-blocks explicit personal advice, operational harm, prompt injection and coding tasks. Anything unsure goes to layer 2, a Groq model that returns a JSON verdict. A declined question gets a fixed text, never model-written text. The scope is published research on AI-assisted software development, epidemiology, climate science, research reproducibility and ecology.
2. **Full-document retrieval and admission.** Passages are retrieved from a small, operator-selected, licence-checked collection of full documents stored in Postgres. Each passage is evaluated and admitted or rejected with a reason code. No admitted passage means no generated answer.
3. **Answer and claim check.** The model writes from up to six admitted passages with numbered citations. A second model call judges each cited sentence against its passage. Unsupported sentences are removed, and if none survive, no answer is shown.
4. **Discovery and same-request acquisition.** Discovery is DuckDuckGo first (used only as a pointer to scholarly identifiers), then OpenAlex, Europe PMC, DOAJ and arXiv, through a portal registry (`config/portal_registry.json`; the optional portal-use recorder is off by default). Acquisition runs inside the request when the collection has no answer (a corpus miss) or when the cited corpus content was added more than 24 hours ago (freshness check, `config/auto_acquire.json`). Full text comes from Europe PMC (JATS XML), or from arXiv HTML only for a specific paper version whose own licence link is CC BY, BY-SA, BY-NC or BY-NC-SA, with the route evidence recorded. A document is admitted only as full text: abstract-only records are never admitted. It must also pass the licence gate, content and injection checks and a per-document relevance gate. Documents with enough relevant passages are added to the collection. Each request ends with a research outcome report saying what was tried, what was admitted and why candidates were not. A best-effort background job can follow; it lives in process memory and is lost on restart. Bounds are in `config/auto_acquire.json` (for example 30 jobs per hour, 180 per day, 2 documents per background job, up to 31 documents stored by one live prompt, and 30 prompts per hour and 180 per day per client (owner decisions OD-1031-PROMPTLIM and OD-1031-CEIL31)). A kill switch, `VERA_AUTOACQ_DISABLED=1`, turns acquisition off.
5. **Reading list fallback.** Otherwise the response is a list of published documents that may be relevant, with links, from the search providers above. VERA-DAR has not read those documents, and the response says so.

Answers from acquired documents carry a notice, and a question outside the scope can be answered only from open-access documents found for it, with a stated warning that the answer may be wrong.

## Stack

Python, FastAPI and Uvicorn (API service), Streamlit (UI service), Pydantic, SQLAlchemy with psycopg2 on Postgres, the OpenAI-compatible client pointed at Groq (`openai/gpt-oss-20b`, then `openai/gpt-oss-120b`, for scope and answers; claim checking tries `qwen/qwen3.8-27b` first, then `openai/gpt-oss-120b`; the OpenAI fallback is off by default), defusedxml for JATS full-text conversion, pytest, Docker, and Render for interim hosting: one Render service for the API (`./start.sh api`), one for the UI (`./start.sh ui`), plus a Postgres database (see [Deployment](docs/deployment.md)). Dependencies are pinned in `requirements.txt`.

## Pages and interface

The Streamlit UI has five pages: **Ask VERA-DAR**, **Report**, **Trace Eval** (two tabs: **Historical**, a dated earlier evaluation run that does not describe the current service, and **Live aggregate**, anonymous aggregate statistics that stay off until the owner switches collection on), **How to use** and **About**. In the Ask question box, pressing Enter submits the question.

Disclosure: every page except About shows one short notice (an AI system whose answers may be wrong, with the question sent to third-party AI and search services) that links to About. The full disclosure appears only on the **About** page.

## Changes of 2026-11-03 (NONASK-1103)

Status of everything below: **offline-tested only; not deployed; live behaviour unverified.** Nothing here has been run against the production service.

Enabled in code:

- Report page layer order, development-only internals hidden, and Enter-to-submit on the Ask page.
- Trace Eval Historical and Live aggregate tabs; short notice on non-About pages; full disclosure on About only.
- Truthful no-answer causes and plural-correct counts in status text; `contract.no_answer_cause`.
- Evidence trace in `contract.trace` (plain-language reasons passages were selected or not used; fails closed when its label file is missing).
- arXiv as an acquisition provider (own call, own budget slot).
- Recognition of Handle identifiers (`hdl.handle.net/...`, `hdl:`), shown as persistent identifiers; no automatic fetch route.

Prepared for push 2 (offline-tested, not deployed, live unverified; see [Citations](docs/citations.md)):

- APA 7 house variant "identifiers appended": references in the answer area, reading list and Report page (never the sidebar).
- Deterministic sub-query decomposition and per-sub-query adequacy verdicts in the public trace.
- Answer-only, never-stored handling of pages without an open licence (policy in plain words).
- Two search lanes, the Official lane and the Current-reporting lane. Both ship off. See [Citations](docs/citations.md#search-lanes) and [Lanes enable checklist](docs/lanes-enable-checklist.md).

Prepared but flag-off (shipped disabled; thresholds are uncalibrated placeholders):

- Topic anchor admission rule (`fulldoc_answer.json`, `topic_anchor.enabled=false`).
- Reading-list relevance floor (`domain_routing.json`, `reading_list_floor.enabled=false`).
- `answer.per_part.keep_partial_parts` (default false).
- Live aggregate statistics (`config/live_stats.json`, `enabled=false`); needs the owner-run draft migration 013, which is not applied.

## Try it

- Live demo: hosted on Render (interim, free-tier; the first request after idle can be slow while the service wakes). The address is not published in this file; ask the owner.
- Recording: none yet (see [Status](#status)). A 30-60 s clip will be linked here once one exists.

<!-- LIVE-RESULT -->
- Locally: see [Development](docs/development.md). The API needs a key (`X-API-Key` header) and a Groq key in the environment. Without a configured full-document corpus source it returns reading lists only.
- Container entry point `start.sh` refuses to run unless `RENDER=true` or `VERA_ENV_CONTEXT=production` is set; locally use `run.sh` / `run_demo.sh`. Untested (assumed) for a fresh machine.

Endpoints: the main route is `POST /ask` (API key in the `X-API-Key` header). The full endpoint list, keys and response semantics are in [API and data use](docs/api.md).

The `/ask` answer is shown in layers: **Answer**; **Evidence and limitations**; **Research and verification** (collapsed by default, shown when there is something to explain); and **Technical details**, which appear in owner/dev mode only. The layers draw on the response fields `answer`, `answer_status`, `limitations`, `process_notes` and the evidence fields described in [API and data use](docs/api.md). The layered output (2026-11-03) is tested offline only and may not be in the deployed build.

## Evaluation: what has been measured

Numbers below are small-sample and not comparable with each other.

**Scope-router fix (OFFLINE LAYER-1 SCOPE-ROUTER measurement, not an answer success rate).** The earlier scope rules blocked topic words such as "medicine" or "stocks", so legitimate research questions were refused. The fix replaces topic blocking with intent-based rules. Measured on the 85-probe set the fix was developed against (48 that should pass, 37 that should be blocked; not held out), run against layer 1 only and reproduced independently twice:

| Measure (offline layer-1 scope router) | Before | After the fix |
|---|---:|---:|
| Benign questions wrongly blocked | 34/48 | 0/48 |
| Should-block questions caught | 28/37 | 36/37 |

These count how the rule layer routes questions. They say nothing about whether an answer was correct. Limits: layer 1 only, no model calls; the probe set was written by a verifier agent before the fix, was used while developing it (three probes appear in the tests), is not held out and has not been externally reviewed; one operational-harm probe still passes layer 1 and relies on layer 2. Real-model layer-2 behaviour under the new prompt is untested (assumed).

**Historical TRACE v1 (2026-10-04, build 0094955, abstract-grounded variant, SUPERSEDED).** Citation-related measures improved over a question-only baseline; the overall pass rate did not. These results describe an earlier build, not the current full-document system, and do not show that the current system passes. Tables, method and limits: [Evaluation](docs/evaluation.md).

**Not measured (recorded 2026-10-23; no later measurement recorded as of 2026-10-30):** answer quality of the current full-document path, any comparison with other systems, the accuracy of the claim checker, and live production behaviour of discovery-to-answer acquisition.

## Memory and what is stored

**Live statistics (off by default).** When the owner enables live stats, one anonymous record is kept per answer (result type, counts, latency, time rounded to the hour; no question text or identifiers). While the flag is off, nothing changes.

Five questions, answered from the code of this candidate:

1. **What persists?** The evidence corpus: open-access full-text documents the operator selected or that automatic acquisition added (title, landing URL, licence, full-text chunks, provenance). **User memory is off**: no visitor profile, no conversation memory, and the question and the derived search query are not stored. `memory.store.enabled` is `false` in `config/ask-provider-chain.json`, and the owner review routes (`/memory/...`) return 503.
2. **When is it written?** After a question is accepted, when a fetched open-access document passes the licence, content, injection and relevance checks. This can happen inside the request (inline acquisition) or in a queued background job, bounded by `config/auto_acquire.json` (30 jobs/hour, 180/day, 2 documents per background job, up to 31 per live prompt, 30 prompts/hour and 180/day per client, OD-1031-PROMPTLIM and OD-1031-CEIL31; kill switch `VERA_AUTOACQ_DISABLED=1`). Documents with too few relevant passages are used for that one answer only and not saved.
3. **Where does it live?** In Postgres (documents, chunks and provenance), written through the app's write account.
4. **How is it retrieved?** Stored documents are loaded into an in-process index (BM25 passage search, `vera/fulldoc_corpus.py`, `vera/dar_corpus.py`); the acquired set is refreshed in the background, at most every 300 s. Retrieved passages go through admission and the claim check before they can support an answer. (The disabled user-memory feature, if enabled, would recall only owner-confirmed rows by term matching, with a 30-day TTL and 14-day half-life.)
5. **When is it removed?** No removal mechanism for corpus documents was found in the application code or the repository's SQL (a search for delete, truncate and expiry paths outside tests, as of 2026-10-22; the corpus store is written insert-only, `vera/dar_store.py`). Documents are kept until the operator removes them by hand in the database, and the public deployment is planned to run through 2026-11-30 (owner decision OD-1027, 2026-10-27; this supersedes the earlier 2026-10-31 end date). Only the disabled user-memory feature has an expiry (TTL and owner rejection).

Other settings: discovery notes are written only when the `L2` layer is enabled (the shipped contract lists `L0` only), and corpus admission of abstract-level sources is off (`corpus.enabled` is `false`).

## Privacy and data handling

Every page of the demo shows one data-use disclosure, built from the live configuration in `vera/disclosure.py`. With the shipped configuration its closing text says that the question is sent to third-party AI and search services, that VERA-DAR does not save the question or information identifying the user, and that its collection may grow with open-access full text found while searching. When full-document answering (enabled in the shipped configuration, but in effect only when a corpus source is configured; see [Default behaviour](docs/api.md#default-behaviour)) answers from the collection, the question, the selected passages with their titles and links, and draft answer sentences are sent to third-party AI services. When the Official or Current-reporting lane is enabled (both ship off), the following applies. Search terms derived from your question may also be sent to GDELT (a news index) and Hacker News search (Algolia). VERA-DAR may read public web pages from vendors and news outlets to answer; these pages are used only for your answer, and page text is not saved. Pages read for one answer are not licence-cleared or stored. GDELT is credited: [GDELT](https://www.gdeltproject.org/).

What the code shows:

- Acquisition runs inside the request (inline, including for in-scope questions) and can also queue a background job. Terms from the question go to Groq (scope, answering, claim checking) and to the search services: DuckDuckGo, OpenAlex, Europe PMC, arXiv and the Wikipedia API.
- Stored: acquired open-access documents (title, landing URL, licence, full-text chunks, provenance). Not stored: the question or the derived search query. The question is held in memory only for a queued job (maximum age 600 s).
- Request caps and rate-limit counters are held in process memory, and the DuckDuckGo query cache is disabled in the demonstration configuration; see [API and data use](docs/api.md).
- Provider retention, and the hosting platform's standard access logs (client IP, path, status), are outside VERA-DAR's control.

Records written by earlier configurations are not erased by this configuration.

## Limitations

- Full-document answers need a configured corpus source; the collection is small, so most questions get a reading list or a decline.
- Claim checking is done by a language model and can be wrong.
- Thresholds are placeholders pending calibration on live data.
- Licence labels come from indexing services and publisher XML and are not verified with rights holders. For listed results whose full text was not acquired, the label reads "Index metadata: <id>; article licence not verified" only when the licence check allows the record and the index declares a licence (for example CC0 for the index record); without a declared licence it reads "article licence not verified (link only)", and held or rejected records keep their hold or reject label. None of these is a licence for the article.
- The scope router's layer 2 and the off-topic category accuracy are uncalibrated.
- Comparative superiority over frontier models is unproven.
- Discovery-to-answer acquisition is offline-tested; it is not live-validated in production. Other live behaviour is untested (assumed) in this document; the free-tier host can be slow on first request.
- Known gaps in the acquisition fallback path: its fetch step does not yet reject a URL with a port or userinfo when no redirect chain is set (the arXiv licence gate does reject them); an unsigned evidence marker remains an open recorded residual risk (RR-1); and a missing or unparseable corpus timestamp is repaired in memory only unless the app has write access, so the repair is not persisted without a grant.
- The background acquisition queue is best effort and in memory; queued jobs are lost on restart.
- No removal mechanism exists for stored corpus documents (see Memory, answer 5).

## Documentation

| Resource | Contents |
|---|---|
| [Documentation index](docs/README.md) | All published documents |
| [API and data use](docs/api.md) | Endpoints, response semantics, privacy |
| [Development](docs/development.md) | Local setup and tests |
| [Deployment](docs/deployment.md) | Docker and Render (interim host) |
| [Citations and storage terms](docs/citations.md) | APA 7 house variant, sub-queries, storage policy |
| [Architecture](docs/architecture.md) | Conceptual architecture and gaps |
| [Evaluation](docs/evaluation.md) | TRACE method, results, limits |
| [Release checks](docs/release-checks.md) | Release gate checks |
| [Design](docs/vera-design.md) | Full design and milestone status |
| [Database](db/README.md) | Schema, accounts and migrations |

## Credit

Thank you to arXiv for use of its open access interoperability. Some bibliographic data and abstracts come from Semantic Scholar (semanticscholar.org), through the Semantic Scholar Academic Graph API. The same acknowledgements appear on the About page.

Built during the AI engineering bootcamp capstone (TAI Labs Agentic AI Engineering Bootcamp, 2026) and published for education, evaluation and portfolio purposes.

Copyright (c) 2026 Tagg Maiwald. All rights reserved. [OWNER: confirm this line]

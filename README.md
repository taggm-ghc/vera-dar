# VERA-DAR

## Verifiable Evidence-based Research Answers

**VERA-DAR is a provenance-grounded agentic context-engineering system that
discovers, critically evaluates, registers, and organizes evidence into
an auditable reasoning context. The goal is research responses better
than those of direct leading-LLM baselines; that comparison has not yet been run.**

**Current state (owner rulings DEC-FDA1, full-document acquisition, and DEC-EI1, no citation claims more inspection than was done).** Today VERA-DAR examines only index records and their abstracts (web pages found by search are read only to extract DOI or arXiv identifiers, never as evidence), and, in the offline M2 path, publisher landing pages; it never acquires or inspects a full source document as such, so its evaluation of evidence is currently abstract-level; full-document acquisition status: REQUIRED, DESIGNED, NOT IMPLEMENTED (DEC-FDA1, DEC-OD10). Not implementing it before Demo Day is an accepted schedule/scope divergence (DEC-OD10), not a waiver or retirement of DEC-FDA1.
Licence decisions shown apply VERA-DAR's licence rules to a CC0 metadata licence that VERA-DAR records for arXiv, OpenAlex and DOAJ from each service's published terms; it covers metadata only, not the article. Semantic Scholar abstracts and web links carry no declared licence and are held for review at corpus admission; Semantic Scholar abstracts can still be used in answers before any licence check is enforced (see "Licence rule" under "Status and limitations"). VERA-DAR has not verified any licence at source and has not assessed a rights disposition.

VERA-DAR is being developed as a capstone project for the TAI Labs Agentic AI Engineering Bootcamp.

## The Problem

Large language models can produce fluent research answers without making
the underlying evidence-selection process sufficiently inspectable.

A plausible answer is not necessarily a well-supported answer.
Conversely, an elaborate research process has limited value if its final
answer is no better than what a leading model can produce directly.

VERA-DAR therefore treats both **evidentiary defensibility** and **response
quality** as requirements.

The project examines a bounded question:

> Can deliberate, auditable evidence acquisition, critical evaluation,
> and context construction cause a frontier model to produce a
> materially better research response than it produces directly?

## What VERA-DAR Does

Given a bounded research question, VERA-DAR:

-   discovers and critically evaluates relevant evidence;
-   preserves evidence provenance;
-   constructs a deliberately selected research context;
-   produces a grounded research response; and
-   evaluates the resulting response against a direct leading-model
    baseline.

At a public architectural level:

``` text
Research Question
       |
       v
Evidence Discovery and Evaluation
       |
       v
Provenance-Grounded Context
       |
       v
Research Synthesis
       |
       v
Evaluation
       |
       v
Auditable Research Answer
```

The internal decision structures, representations, scoring methods,
orchestration mechanisms, and other implementation details used to
perform these functions are outside the scope of this public repository.

## Design Objectives

VERA-DAR is designed around four public objectives:

**Grounding.** Research claims should be supported by relevant evidence
rather than generated solely from model parameters.

**Provenance.** Consequential evidence should remain traceable to its
source.

**Auditability.** Evidence-selection and evaluation decisions should be
inspectable.

**Response quality.** Provenance and auditability are requirements, but
they do not compensate for an inferior final answer.

## Comparative Evaluation

VERA-DAR is evaluated against direct responses to the same bounded research
question from leading language models.

The central question is whether supplying a deliberately constructed,
critically evaluated evidence context produces a materially better
research response than direct model generation.

Evaluation considers the quality of the resulting answer as well as the
traceability of its supporting evidence.

## API

VERA-DAR is implemented as a Python service using FastAPI.

The application interface includes:

``` text
GET  /health
GET  /version
POST /ask
GET  /corpus-summary
GET  /memory/pending
POST /memory/{claim_id}/confirm
POST /memory/{claim_id}/reject
```

`GET /health` provides a basic service health check (no key). `GET /version` (no key) returns the deployed commit (`commit`, `short`) so a deploy can be verified; it carries no host or secret.

`POST /ask` is the primary research-question interface (key: `VERA_API_KEY`). Since 2026-10-07 (items #77 and
#79): the scope router runs first; an accepted question is searched live (arXiv, OpenAlex, DuckDuckGo links
resolved to index records and their abstracts); the answer is written only from the retrieved abstracts with [n] citations, each
cited claim is checked against the cited abstract, and unsupported sentences are removed. Request:
`{"question": "...", "mode": "normal"}`. Response:

``` text
answer          text with [n] citations (or a decline, no-sources or insufficient-evidence reply)
model           provider:model that answered (Groq first, OpenAI fallback)
tokens_used     integer
cost_usd        number (0 on the Groq free tier)
sources         [{n, title, url, published, source_type, provider, identifier, retrieved_at, licence_decision}]
traced_sources  index records found from links but without a free abstract (listed, not used; not full documents)
claim_check     {checked, supported, partial, removed, checker, outcome}
```

`licence_decision` is the licence gate's outcome on the licence VERA-DAR records for the source (for indexes, from the service's published terms). For arXiv, OpenAlex and DOAJ results, VERA-DAR records a CC0 licence that covers their metadata only. It does not license the article's text, and VERA-DAR has not verified rights with any rights holder. For an abstract-only record, `allow` rests on that metadata licence, so `allow` does not show rights to the underlying article. A Semantic Scholar abstract has no declared licence and gets `defer`.

Each `sources` entry is source-record provenance: provider, identifier, retrieval time and licence decision. It has no immutable document version, no content hash and no passage location, because no full document is acquired (full-document acquisition: REQUIRED, DESIGNED, NOT IMPLEMENTED). It has no explicit inspection-scope label yet; that label is a recorded requirement under DEC-EI1. Every listed source is an index record with its abstract.

`GET /corpus-summary` (key: `VERA_API_KEY`; item #83) describes VERA-DAR's own admitted sources: count, newest
source time, a model-written description and topics; no titles, URLs or abstracts.

The three `/memory/...` routes are owner-only (key: `VERA_OWNER_KEY`, header `X-Owner-Key`; D6). `GET /memory/pending` lists findings awaiting review, including legacy auto-written rows; `confirm` makes a finding recall-eligible and `reject` stops it being recalled. See "How VERA-DAR remembers".

Every page (Ask, demo, trace eval) shows one shared data-use disclosure from `vera/disclosure.py` (item #74; reworded by the owner's 2026-10-19 Demo Day decision, Option C, R-028). Its closing text, as shipped, is: "Your question is sent to third-party AI and search services to generate a response. These services have their own data-handling and retention policies. VERA-DAR does not save your question or information identifying you. During this demonstration, VERA-DAR does not save newly discovered documents or research notes from your questions." The last sentence is emitted only while `corpus.enabled` is false in `config/ask-provider-chain.json` and `layers` in `config/discovery_contract.json` omits L2; with corpus admission on, the text says VERA-DAR may add documents found for a question to its own collection (never the question). The API logs no per-request access line and no exception text on the server-error path (`main.py`); Render platform logs are outside VERA-DAR's control and unverified.

The M2-M6 research pipeline under `vera/` (with Gate C, the M5 claim check and the M6 judge; M5's `verified` label is a recorded DEC-EI1 overclaim, because the M2 text it checks is usually a landing page) remains a separate,
offline path; `/ask` reuses its search providers, Gate A and licence gate, not the whole pipeline.

## Model Choice

VERA-DAR currently uses OpenAI's `gpt-4.1-nano`. At this stage the deliverable
is a reliable, cost-observable `/ask` contract rather than answer quality
tuned across models, so the cheapest acceptable model is used rather than
a flagship one; this is revisited once response quality becomes a
load-bearing evaluation dimension.

Where the model name comes from today (read from code 2026-10-02): `vera/pipeline_llm.py` takes
the default from `config/model-selection.json` and falls back to a hardcoded `gpt-4.1-nano`;
`vera/gate_a.py` and `vera/m6/baseline_collection.py` hardcode `gpt-4.1-nano` themselves.

`/ask` only (2026-10-07, item #77): free providers first. `config/ask-provider-chain.json` lists Groq
`openai/gpt-oss-20b`, then `openai/gpt-oss-120b` (free tier, recorded at $0, `reasoning_effort` low); an entry
whose key is unset is skipped, and the selected OpenAI model is always the last resort. Any provider error
or an empty answer moves the request to the next entry; the response's `model` field names the one that
answered. Evaluation paths (TRACE capture, M6 baseline and judge, Gate A, pipeline) do not use the chain.
Because free answers cost $0, the daily cost cap only binds on OpenAI fallbacks; the request caps still
apply. The unused per-task file `.vera-provider-chain.json` was removed (its Groq model was retired).

Grounded `/ask` (2026-10-07, item #79). For each question the scope router accepts, `/ask` searches arXiv,
OpenAlex and DuckDuckGo once (`config/ask-provider-chain.json` `grounding`). Results with an abstract are
used directly. Link-only results (DuckDuckGo, news, portals, OpenAlex records without an abstract) are
resolved to a scholarly index record and its abstract, not the full paper (currently abstract-level; full-document acquisition: REQUIRED, DESIGNED, NOT IMPLEMENTED (DEC-FDA1); `vera/source_resolver.py`): the page's citation DOI, arXiv, SSRN and linked
DOIs, one more hop through known scholarly hosts, then an OpenAlex title match (Jaccard >= 0.8; a pre-existing value with no calibration note, an open item under the calibration rule) for portals
that refuse the fetch; abstracts come from OpenAlex, then Semantic Scholar. All counts, hops and time limits
are in the `resolve` block. The answer uses the grounded-v1 prompt (numbered abstracts, each cut to `snippet_max_chars` = 1,500 characters (a pre-existing value with no calibration note yet; an open item under the calibration rule, not an accepted divergence), cite only; the claim check sees the same cut text), invalid
citation numbers are removed with a note, and the response lists `sources` (no abstracts) and
`traced_sources` (records found from links but without a free abstract, not used). With no usable source, `/ask` says so
and does not answer from memory.

Corpus growth (item #79 phase C). After the response, each source with an abstract goes through Gate A
(`vera/gate_a.py`, scored by the first `/ask` provider) and the licence gate (`vera/licence_gate.py`). Every
candidate is recorded with its decision; `fetch` decisions (a Gate A label meaning "selected for acquisition"; no full document is fetched today; `candidates.fetch_status='fetched'` for an abstract-only record is a recorded DEC-EI1 breach) that the licence gate allows are stored as corpus
sources (metadata and abstract only) under one fixed corpus question per process. The visitor's question is
never stored. Writes use the production write-only account `vera_pipeline_wo` via `VERA_DB_URL_WO` (unset =
disabled); every step is one INSERT ... RETURNING, so a source already in the corpus is counted as known.

The M6 judge must come from a different model family than the generator. Since goal #2 (2026-10-02,
offline, not yet run live) the judge is chosen at run time from the live list of Groq-served models
whose family is not OpenAI, using the pool, patterns and limits in `config/vera_eval_run.json`
(`vera/judge_select.py`, `vera/eval_config.py`). No model is a code default; the family is looked up by
model id and an unmapped model is refused. The choice is recorded per run and fixed within a run. The
family patterns are not yet checked against Groq's live list. No Anthropic key will be used (item A2
dropped 2026-10-02) and the Anthropic judge was removed from the code. Mistral is not supported.

## Running Locally

### Requirements

-   Python 3.11+
-   Python virtual environment
-   required Python dependencies
-   required API credentials stored locally in `.env`

Do not commit `.env` or API credentials to source control.

Start the development service with:

``` bash
./run.sh
```

The default local address is:

``` text
http://127.0.0.1:8001
```

Health check:

``` text
http://127.0.0.1:8001/health
```

FastAPI documentation:

``` text
http://127.0.0.1:8001/docs
```

A different port can be supplied when starting the service:

``` bash
./run.sh 8000
```

### Tests

Run the offline suite with the repository's virtual environment (no database or
network needed; a few integration tests skip without credentials):

``` bash
venv/bin/python -m pytest tests
```

Tests marked `live` (real provider calls) run only with `--run-live-stages`; tests marked
`live_api` (they call the deployed API) also need `API_BASE_URL` set, for example
`API_BASE_URL=... venv/bin/python -m pytest tests --run-live-stages`. Plain `pytest tests` makes no live calls. `--run-live-stages` is the only supported switch. The old `VERA_INTEGRATION=1` still enables `live` tests (with a DeprecationWarning) as a deprecated alias that stops working after 2026-10-25.

Last result, 2026-10-12 (at 7fc7fe1): 965 passed, 3 skipped; release gate PASS 6/6 (Rule 9 is NOT COVERED by the gate; see "Status and limitations") (`python scripts/release_gate.py` runs the same suite plus the live catalog check, frozen-file hashes and static checks). Run `tests/` only: the script
`vera/m6/test_m6_schema_sync.py` is not a pytest module and exits at import.
Integration paths that touch the database write to the shared production
instance, so use identifiable, removable test rows only.

### Demo page

``` bash
./run_demo.sh            # Streamlit demo page, port 8502
VERA_DEMO_SOURCE=fixture ./run_demo.sh   # force the labelled fixture
```

It loads the latest evaluated run from Postgres and otherwise falls back to a
clearly labelled fixture.

### Environment variables and database accounts (descriptions only)

`.env.example` lists the core variables with no values (other names read by the code are noted
there as comments); real values live only in the gitignored `.env`. Never commit `.env` or put a
secret in `.env.example`.

-   `OPENAI_API_KEY`: model provider key (generator and baseline). No Anthropic key will be used (A2 dropped 2026-10-02).
-   No search key: M2 search is keyless (arXiv abstract pages, `vera/search_providers.py`; config `search` in `config/vera_eval_run.json`). An optional agent-prepared seed list can be frozen by sha256 as a disclosed fallback (`search.fallback_seed`). A licence gate (`vera/licence_gate.py`) holds undeclared licences for review and rejects only ND, no-educational-use and fee-only terms (and always rejects arXiv e-print, PDF and HTML URLs, per arXiv's terms) (plain NC is accepted; `config/licence_rules.json`) before content is stored.
-   `GROQ_API_KEY` / `MISTRAL_API_KEY`: judge provider keys (`vera/m6/judge.py`); the judge default is Groq.
    `GROQ_API_KEY` also makes `/ask` free-tier-first (item #77, `config/ask-provider-chain.json`); API service only.
-   Optional, read by code: `VERA_COST_LOG` (file the M2 runner appends cost lines to),
    `VERA_JUDGE_*` overrides (model, price, `VERA_JUDGE_ALLOW_SAME_FAMILY=1` to allow a disclosed same-family judge),
    `VERA_DEMO_SOURCE`.
-   `VERA_API_KEY`: key required by `POST /ask`.
-   `VERA_OWNER_KEY`: owner-only key for the `/memory/...` routes (header `X-Owner-Key`); unset = those routes stay closed.
-   `EXTERNAL_DB_URL` / `INTERNAL_DB_URL`: Render Postgres URLs. They point at the
    **same** instance, and dev and production share one schema (`vera_vjay`), so
    "local" database work touches production data. Since 2026-10-07 both use VERA-DAR's own
    `vera_claude_code_rw` account (they previously used an AI-Internship account, which is refused on
    `vera_vjay`).
-   `VERA_DB_URL_RW`: `vera_claude_code_rw` URL for the MVP runner and preflight (`vera.db.get_role_engine`).
-   `VERA_DB_URL_WO`: `vera_pipeline_wo` URL for `/ask` corpus admission (item #79); unset = disabled.
-   `GROQ_API_KEY`: also makes `/ask` free-tier-first (item #77). `.env.example` lists every variable the
    code reads (names only); the gitignored `.env` holds the values.
-   `VERA_DB_PASSWORD_<role>`: one password variable for each of three accounts
    applied 2026-10-01 by the DB admin identity (`db/accounts/create_vera_accounts.sql`):
    `vera_claude_code_rw` (dev read/write), `vera_pipeline_wo` (production write-only,
    column-level SELECT on key columns only), `vera_eval_ro` (production read-only).
    No further accounts are to be created. 2026-10-07: all three logins verified; the write-only grants were
    exercised for corpus admission in a rolled-back transaction (plain INSERT ... RETURNING works; ON CONFLICT
    is refused because it needs SELECT on the conflict columns). `tests/test_m4_m5_integration.py` still
    imports another project's admin engine (skipped; open item).

## Deploying on Render (course-scoped interim host)

Render is the course-scoped interim host only. Public production there ends by 2026-10-31 (most likely 2026-10-28). This section describes the setup; it does not claim that any deployment is verified, and it does not authorize a merge or deploy (Phase 3 is not authorized). Deploys are manual and owner-run. Service addresses are written here only as the placeholders `<API_HOST>` and `<UI_HOST>`; real addresses are never recorded in this repository.

One image (`Dockerfile`) runs both services; `start.sh api|ui` picks which. The API starts with `./start.sh api` (the image's default command); this Render start command is to be confirmed by the owner. `start.sh` refuses to run unless `RENDER=true` or `VERA_ENV_CONTEXT=production` is set. Port: Render's `PORT`, else 8000. Secrets live only in Render's environment settings, never in the repo or the image (`.dockerignore` keeps `.env` and `*-prv` files out).

1. Generate a client key locally: `python3 -c "import secrets; print(secrets.token_urlsafe(32))"`.
2. **API service** (`<API_HOST>`): New → Web Service → this repo, branch `main`; Runtime Docker; Root Directory blank; plan Free; start command `./start.sh api` (to be confirmed by the owner); Health Check Path `/health`; Auto-Deploy Off. Environment: `OPENAI_API_KEY`, `VERA_API_KEY` (the new key), optional `GROQ_API_KEY` (free-tier `/ask`, OpenAI as fallback), optional `VERA_DB_URL_WO` (corpus admission: Render's INTERNAL database URL with the `vera_pipeline_wo` account). Optional `VERA_DB_URL_RO` (sidebar corpus description: Render's INTERNAL database URL with `vera_eval_ro`). Optional `VERA_OWNER_KEY` (owner-only `/memory/...` routes, D6; unset = they return 503; they also need both `VERA_DB_URL_RO` and `VERA_DB_URL_WO`). Leave `VERA_PUBLIC_MODE` unset (public mode is the default). Template: `.env.render.api.example` (copy to the gitignored `.env.render.api`, fill in, paste with "Add from .env").
3. **UI service** (`<UI_HOST>`): New → Web Service → same repo; Runtime Docker; plan Free; Docker Command `./start.sh ui`; Health Check Path `/_stcore/health`; Auto-Deploy Off. Environment: `VERA_API_BASE_URL` (`https://<API_HOST>`, no trailing slash; without it the UI calls `127.0.0.1:8001` and fails), `VERA_API_KEY` (same key), `VERA_DEMO_SOURCE=fixture` (demo pages show the labelled fixture, not live database rows; the shared Postgres database itself is on Render). Template: `.env.render.ui.example` (copy to `.env.render.ui`, same steps). The UI holds no provider key (no `OPENAI_API_KEY` or `GROQ_API_KEY`): it only calls the API, which chooses the provider.
4. Before any deploy, run the non-destructive release checks: `scripts/release_preflight.py` (read-only and offline; see `docs/release-checks.md` for the gates, the owner-only steps and the exit codes). The production release authorization stays with the owner.
5. After an owner-run deploy, check: `https://<API_HOST>/health` returns 200, `/ask` without the key returns 401, `/version` reports the expected commit, and `<UI_HOST>` answers one question. With `GROQ_API_KEY` set, the answer's `model` field (read in dev mode's JSON panel, or by `curl` with the API key against `/ask`; the public page shows no raw JSON) reads `groq:...` and `cost_usd` is 0; `gpt-4.1-nano` there means Groq failed and OpenAI answered. Record a deployment as verified only from that `/version` check.
6. Redeploy after a push (owner only): Manual Deploy → Deploy latest commit.

Limits: free services sleep when idle (first request after sleep waits about a minute); grounded `/ask` adds live search and resolving web results to index records (measured 2 to 4 s locally; bounded by the `grounding` and `resolve` deadlines; the UI waits up to 60 s); public-mode caps are in memory, so they reset whenever a service restarts or wakes; one uvicorn worker by design.

## Week 4: TRACE evaluation (2026-10-04)

The bootcamp's Week 4 asks for "an eval suite your capstone runs against, visible from a Streamlit UI". This section is that suite, run on VERA-DAR itself: Trace, Read, Analyze, Codify, Enforce.

- **What was traced.** 30 frozen questions (`config/trace_questions_v1.json` (held-out evaluation material; not published in this repository): 12 in-scope, 5 multi-part, 5 false-premise, 4 out-of-scope, 4 ambiguous; 20 dev, 10 held-out; agent-authored, 5 seeded from the frozen VERA-DAR question). Two variants, both live calls to `gpt-4.1-nano`: **baseline** was what `POST /ask` sent at TRACE time, 2026-10-04 (the question only, no retrieval; `/ask` has been grounded since #79); **grounded_v1** first runs VERA-DAR's keyless arXiv search (M2) and gives the model up to 5 numbered abstracts with a cite-only-these rule. Harness: `scripts/trace_capture.py`. Raw traces stay local (`tmp/trace_eval/`, gitignored: they hold full abstracts).
- **Read and Analyze.** Every dev trace was read and open-coded before any check was written (`docs/trace_eval/open_coding_baseline_v1.md` (held-out evaluation material; not published in this repository), `open_coding_grounded_v1.md` (held-out evaluation material; not published in this repository)). Baseline failures: no verifiable citation (16/20), "studies show" with no study named (11/20), answers outside the 2023-2026 window (5/20), fabricated or placeholder citations (4/20), false premises accepted, out-of-scope requests answered (including a medicine dose). Saturation was not reached in 40 traces.
- **Codify.** Nine deterministic checks (`vera/trace_eval/checks.py`, `trace-checks-v1`), applied by question category. Results: `scripts/trace_measure.py` writes `eval_results/trace_eval_v1.json` (held-out evaluation material; not published in this repository).
- **Validated, not trusted.** An independent labeller, blind to the check results, labelled the 40 dev traces. Each check's TPR and TNR (positive = failure) is in `eval_results/trace_eval_v1_validation.json` (`scripts/trace_validate.py`). The checks catch almost every labelled failure (TPR 0.95-1.0) but over-flag some passes (TNR 0.5-0.75 on four checks). The word-overlap support heuristic missed all 3 partially supported sentences (TPR 0.0), so it is not a support check.
- **Measured result (all applicable checks pass; capture errors count as failures; Wilson 95% intervals; directional, small n).** Baseline: dev 2/20 (10%), held-out 2/10 (20%). Grounded_v1: dev 3/20 (15%), held-out 2/10 (20%). The headline barely moves, but the failure mix changes: by the blind labels, fabricated citations, unsourced numbers and vague "studies show" claims appear in 3, 3 and 12 baseline dev traces and in 0, 0 and 0 grounded_v1 dev traces. The new top failure is **retrieval**: in 13 of 19 grounded dev traces at most 1 of 5 arXiv results was on topic, and the model then (correctly) declined to answer from them (9/19). Out-of-scope, false-premise and ambiguity handling did not improve.
- **What TRACE found in VERA-DAR's code.** `search_question()` rebuilds its provider chain, and so the arXiv rate gate, on every call; multi-question runs hit HTTP 429 (arXiv also refuses after about 20 quick requests). The harness reuses one chain; the M2 gate defect is fixed (item #71: one cached chain per process). One grounded trace stayed a capture error after bounded retries.
- **Why retrieval misses (diagnosed 2026-10-04).** Mostly the question-to-query step, not the corpus: the query keeps the question's first words, so meta-words such as "published", "evidence" and "show" dominate and match almost any paper (on average 0.82 of 5 results on topic over 22 questions). Hand-focused queries on the same questions found relevant studies. Narrow questions and non-arXiv literature are smaller factors. Evidence: `docs/trace_eval/retrieval_diagnosis_2026-10-04.md` (held-out evaluation material; not published in this repository). Fixed in code by item #71 (query translation, OpenAlex second provider, rank-fusion fan-out); the after-fix measurement has not been run yet.
- **Search providers.** arXiv and OpenAlex, plus DOAJ when enabled (scholarly, keyless, merged by reciprocal rank fusion; DOAJ states that its article metadata is released under a CC0 waiver (an intermediary's assertion, not verified with each rights holder; DOAJ is off by default), and VERA-DAR keeps only title, abstract and bibliographic fields from it, within a 2 requests per second limit and a daily cap), plus DuckDuckGo Lite as a **discovery-only** provider (item #71): it is used to find source URLs (mainstream news and journal portals without open search APIs, through `site:` groups in `config/vera_eval_run.json` `search.duckduckgo_lite`). VERA-DAR keeps only the target URL, the target page's own headline and its domain; it never shows DuckDuckGo snippets, pages or ranking, never auto-fetches discovered pages (no licence is declared, so the licence gate holds them for review), caps them at 2 of the final results, and labels news as grey literature. DuckDuckGo's params page says its parameters are intended for individual use; VERA-DAR uses it at low volume (5 s gap, daily cap of 8 queries, and a local per-question cache of URLs and headlines only under the gitignored `tmp/search_cache/`) and treats a challenge page as a failure, never evading it. Switch it off with `search.duckduckgo_lite.enabled: false`. **Google Scholar is a link-out only:** the demo page shows a "Search Google Scholar yourself" button built from VERA-DAR's query terms; the person clicks it in their own browser, and VERA-DAR never queries, fetches or parses Google Scholar, because Scholar's robots.txt disallows `/scholar` and Google's terms forbid automated access against robots.txt.
- **See it.** `streamlit run streamlit_app.py`, page **trace eval**: headline, per-check and validation tables, decision guidance (2026-10-07: paired exact McNemar tests per split and per check, Wilson intervals for check TPR/TNR, indicators and the formulas, thresholds in `config/trace_eval_guidance.json`), and a trace browser. On this snapshot only citation presence (p = 0.0039) and phantom evidence (p = 0.0002) improve significantly; the overall pass rate is not distinguishable.
- **Limits.** Agent-authored questions (self-preference risk); a single agent coder and labeller from the same model family as the builder; 30 questions; abstracts only; the checks were frozen before the grounded_v1 run but are heuristics. Next fix by this evidence: retrieval query translation, then scope routing.

## Case study

**Problem.** A fluent LLM research answer can hide where its evidence came from. The test: does retrieving real abstracts and forcing citations to them make the answer more verifiable than the same model answering alone?

**Architecture and stack.** `POST /ask` runs a scope router, then a keyless live search (arXiv and OpenAlex, rank-fused, plus DuckDuckGo links resolved to index records), then an answer from the retrieved abstracts only with `[n]` citations, a claim check against each cited abstract, and a gated memory (checked findings are written as pending; only owner-confirmed findings are recalled). FastAPI service, Streamlit UI, one Postgres with three least-privilege accounts, free-tier-first models (Groq, then `gpt-4.1-nano`), deterministic checks for evaluation. Sections above give detail.

**Baseline vs grounded variant (one valid pair, not a before/after of a fix).** Same 30 frozen questions, same model (`gpt-4.1-nano`), same frozen checks (`trace-checks-v1`), one run each, both captured on 2026-10-04, before item #71. Variant 1 = `baseline` (the question only, no retrieval); variant 2 = `grounded_v1` (keyless arXiv search, up to 5 abstracts, cite-only-these rule).

| Measure (source; variant comparison, not a before/after of a fix) | Baseline | Grounded_v1 | Test |
|---|---|---|---|
| Citation present (A1), 22 applicable traces | 1/22 (5%) | 10/22 (45%) | exact McNemar p = 0.0039 |
| No phantom evidence (A4), 22 applicable traces | 4/22 (18%) | 17/22 (77%) | exact McNemar p = 0.0002 |
| All applicable checks pass, dev (n=20) | 2/20 (10%) | 3/20 (15%) | not distinguishable |
| All applicable checks pass, held-out (n=10) | 2/10 (20%) | 2/10 (20%) | no change |

Sources: `eval_results/trace_eval_v1.json` (held-out evaluation material; not published in this repository) (generated 2026-10-04), `eval_results/trace_eval_v1_validation.json`, `docs/trace_eval/`.

**Caveats.** The headline pass rate did not move; the gain is in the failure mix (by the blind labels, dev traces with fabricated citations, unsourced numbers and vague "studies show" claims: 3, 3 and 12 for the baseline, 0, 0 and 0 for grounded_v1), and retrieval became the top failure (13 of 19 grounded dev traces had at most 1 of 5 results on topic). Agent-authored questions; one agent coder and labeller from the builder's model family; 30 questions, one run; the checks are heuristics validated on dev only (they over-flag some passes, TNR 0.5 to 0.75 on four checks); one grounded capture error counts as a failure; the grounded_v1 variant predates the query-translation fix (item #71) and the grounded `/ask` (item #79). Directional, small n.

**Not a valid pair, so not claimed.** (1) Strictly relevant results in the top 5, 0.27 to 0.77 (item #71, 22 questions): later re-scoring showed a labeller effect of about 0.32, and with one labeller the shipped chain is level with the old one (1.00 vs 1.09). (2) `eval_results/grounded_ask_v1.json` (held-out evaluation material; not published in this repository) (2026-10-08, post-#79: 19 of 30 answered with checked citations, unsupported-claim rate 9 of 105 = 8.6%, Wilson 95% 4.6 to 15.5%) has no baseline on the same claim check. Cheapest measurement for a post-fix pair: rerun `scripts/trace_measure.py` on the same 30 questions and frozen checks against today's `/ask`, and run the claim check on the 20 dev baseline answers (an estimated 30 to 50 `gpt-4.1-nano` or free-tier calls, cents; not yet run).

**What the comparison shows (two variants captured together, not a before/after of a fix).** The baseline cited a source in 1/22 applicable traces and the grounded variant in 10/22, about half; the grounded variant had no fabricated citations, unsourced numbers or vague "studies show" claims in the labelled dev set. In exchange the grounded variant declines when search returns nothing useful, which exposed retrieval quality as the next problem.

## Status and limitations (2026-10-04, updated 2026-10-14)

Written as dated observations, not guarantees. Every measured figure carries its caveat; nothing here is a statistical claim.

**What VERA-DAR does now**

- **Grounded `/ask` (2026-10-07, item #79; built and tested; pushed; part of the earlier build dabd92e, deployed and verified via `/version` on 2026-10-14; this reconciled branch has not been deployed).** Live search per accepted question, link-only results resolved to index records and their abstracts (not full papers), answers only from retrieved abstracts with [n] citations, invalid citation numbers removed, sources listed under the answer, an explicit no-sources reply instead of memory answers; corpus admission (Gate A + licence gate) after the response. See "Grounded `/ask`" above. It meets the criteria R1 set for the public default, measured once (directional, small n; see the next list).
- **Free-tier-first models for `/ask`** (item #77): Groq `openai/gpt-oss-20b`, then `openai/gpt-oss-120b`, then OpenAI `gpt-4.1-nano`; evaluation paths stay on `gpt-4.1-nano`.
- **"What VERA-DAR draws on" sidebar** (item #78) on every page: scope, how answers are produced, and the Trace Eval sources (arXiv abstract records), read from shipped files. Visitor-facing labels in `vera/sources_sidebar.py`, `vera/disclosure.py` and `vera/memory_panel.py` were reworded to state that sources are index records and abstracts, not full papers (DEC-FL2). The `vera/sources_sidebar.py` wording was committed in fcb7cd3 (2026-10-15), which also limits the sidebar to aggregates (no individual source titles, no gate names); on 2026-10-15 the owner reported fcb7cd3 deployed, which was not verified via `/version` at the time. As of 2026-10-15 the other reworded labels (`vera/disclosure.py`, `vera/memory_panel.py`) had not been deployed. The last build verified via `/version` at the time, dabd92e (2026-10-14), showed the earlier labels ("Evidence snapshot", "web results followed back to the original papers", "traced from web results to the original source"). This README records only verified deploys.
- **Corpus description and connection triage** (item #83, ported from AI-Internship). `GET /corpus-summary` (behind `VERA_API_KEY`; the UI calls it server-side) describes VERA-DAR's own admitted sources in model-written prose with topics, from titles and metadata only, through the read-only account (`VERA_DB_URL_RO`); it is regenerated only when the source count, newest source time or prompt version change (memory cache; deterministic fallback with a retry cooldown) and shown in the sidebar. When the Ask page cannot reach the API, it shows an ordered, host-free triage (address, reachable, awake, `/health`, endpoint) stopping at the earliest failing step.

- **Search providers.** arXiv and OpenAlex (keyless, merged by reciprocal rank fusion). A DOAJ provider exists but is switched off in config: in two blind measurements (2026-10-04) it added no directly relevant result the others missed, and in the second it displaced two relevant ones. Zero-hit query relaxation is also built but off: it lowered strict relevance (directional, one labeller, 22 questions). DuckDuckGo Lite is **discovery-only**: VERA-DAR keeps only the target URL, that page's own headline and its domain, at low volume (5 s gap, daily cap of 8 queries, at most 2 of the top 5 results; a challenge page counts as a failure and is never evaded). DuckDuckGo's params page says its parameters are intended for individual use; that terms note is why volume stays low and the provider can be switched off (`search.duckduckgo_lite.enabled: false`). Discovered pages are not auto-fetched. **Google Scholar is a link-out only**: a person clicks a button in their own browser; VERA-DAR never queries it. Measured on 22 questions, one AI labeller (item #71, before the OpenAlex and DOAJ additions are separated out): strictly relevant results in the top 5 were 0.27 (old chain) vs 0.77 (new chain) per question, as first scored; directional, small n, not re-measured since. Later re-scoring found a labeller effect of about 0.32, and with one labeller the shipped chain was level with the old one, so this rise is not claimed (see the case study).
- **Scope router** (`vera/scope_router.py`, `config/scope_router.json`). A deterministic first layer plus an optional small-model classifier; declined questions never reach search or the answering model. On the frozen 50-item scope set (`config/scope_set_v1.json`, sha256-frozen, AI-authored): false refusals 1/24 (Wilson 95% 0.007 to 0.202) and false accepts 0/26 (0.000 to 0.129), identical for layer 1 alone and with the classifier. **Directional, small n.** The one false refusal cites a 2025 study that the classifier called a future date.
- **Public mode and caps** (`vera/public_mode.py`, `config/public_mode.json`). Public mode is the default when `VERA_PUBLIC_MODE` is unset. The upstream URL comes from server configuration only, the key stays server-side and is never put in a widget, request and daily-cost caps return a generic 429 with no provider text, and model text is rendered through `safe_markdown`. Caps today are proposals: 6 requests per minute and 40 per day per client, 400 per day overall, USD 2.00 per day overall and USD 0.25 per client.
- **Licence rule (approved by R1, 2026-10-04).** The gate rejects only ND, no-educational-use and fee-only terms (and always rejects arXiv e-print, PDF and HTML URLs, per arXiv's terms); plain NC is accepted; an undeclared or unclear licence is held for review. The gate judges the metadata licence VERA-DAR records from the indexing service's published terms, not a licence read from the record or the article; VERA-DAR has not verified it at source and has not assessed a rights disposition. Checked on 7 hand-written cases (`vera/licence_gate.py`). VERA-DAR records no declared licence for Semantic Scholar abstracts. They can currently be used in answers before any licence check is enforced (the decision shown with each source is recorded, not applied to what the answer model sees); at corpus admission they are held for review, and the memory write gate refuses findings that cite them.
- **Release gate** (`python scripts/release_gate.py`). Six required checks: full suite, the SQL catalog test run with `--run-live-stages` (read-only; fails closed, so a skip or zero tests fails the gate), sha256 of the frozen files, and static safety checks (no environment key passed to a widget, no `unsafe_allow_html` on model text, public mode default). `--live` additionally re-runs scope layer 1 only (zero cost) and fails on any false accept (directional, small n). **Rule 9 (no citation claims more inspection than was done; DEC-EI1) is NOT COVERED by the release gate** until the deterministic inspection-ceiling check (EI-7) is implemented. A gate pass says nothing about Rule 9, and compliance is never claimed from the policy text alone. Known compliance defects EI-1 to EI-7 are recorded in the owner's compliance-defects register (kept outside this repository); they are defects, not an accepted divergence, and Rule 9 is not waived.

## How VERA-DAR remembers (Week 5)

**What it keeps.** Three cases, kept apart. (1) *Demo Day configuration (shipped, 2026-10-19; the visitor disclosure reads "Your question is sent to third-party AI and search services to generate a response. These services have their own data-handling and retention policies. VERA-DAR does not save your question or information identifying you. During this demonstration, VERA-DAR does not save newly discovered documents or research notes from your questions."):* public `/ask` makes no new storage from questions: `corpus.enabled` is false (no corpus admission, no licence observations), `memory.store.enabled` is false (no memory claims), `layers` is `["L0"]` (no discovery notes), and the DuckDuckGo search cache is off (`search.duckduckgo_lite.cache_dir` is null). Mocked tests pin this; the deployed environment is not exercised by them. (2) *Other configurations:* with corpus admission on, sources found for a question are admitted to the corpus (not the question text), and with memory and L2 on, checked findings and discovery notes are stored as pending; both are described below. (3) *Historical data:* rows written before the Demo Day change (corpus sources, claims, discovery notes) remain in the database and are not deleted. In none of these does VERA-DAR keep who you are or any session id. The question is still sent to third-party providers, whose own retention VERA-DAR does not control.

**How it gets in.** After the answer, findings from `/ask` that pass the claim check and the structural write gate are stored as pending, in the background after the cited source is admitted to the corpus. Pending findings are not recalled. Only the owner confirms or rejects them, through owner-key routes. Findings written before this change are not recalled either; they are listed for review. The gate tests for support, allowed sources (arXiv or OpenAlex, with a licence-gate decision of `allow` on the metadata licence VERA-DAR records from the index's published terms; that is not a rights finding), identifier, and refusable content (tool dumps, instructions, personal remarks, echoed questions, time-bound claims).

**Where it lives.** In VERA-DAR's Postgres database, in the existing claims, answers and evidence tables, under one fixed memory label. Process memory holds no candidates.

**How it comes back.** On later questions, VERA-DAR searches confirmed findings by text match and recency, using the read-only account. Only findings above a minimum relevance floor are shown after the answer under "From VERA-DAR's memory", with their date and source. They are displayed only; they are never fed back to the model.

**Recall status.** Cross-session recall on the current owner-confirmed build (D6, live dabd92e) has NOT yet passed a live proof. Two attempts on 2026-10-13 (build 12b3616, also D6) and two valid attempts on 2026-10-14 (dabd92e) failed at stage (c): no new pending finding was written. A third 2026-10-14 attempt was invalidated and is not counted. On 2026-10-14 the attempts retrieved 0 or 1 sources, and the last one had no claim that passed the check; the 2026-10-13 cause is a diagnosis, not proven. The only recall demonstration so far is from 2026-10-09 (labelled pre-D6), when findings were still saved automatically; that path is no longer recall-eligible.

**How it is forgotten.** Findings older than 30 days are not recalled. Older findings rank lower, with a 14-day half-life. The owner can reject a finding (owner-key routes), and a rejected finding stops being recalled. Permanent deletion needs the owner's approval each time.

**Known limits.** Stored text is the model's own sentence checked against an abstract, so a fluent false claim from a bad abstract could pass. The gate is hygiene, not a poisoning defence. When a claim cites multiple sources, each memory row keeps one evidence span per cited source covering the whole stored abstract text (up to 1,500 characters), not an examined passage (a recorded DEC-EI1 gap; see `docs/vera-design.md`, "Current state vs required"). Owner confirmation is a status change only: it checks the row's state and expiry but does not re-check the cited spans, source version or rights, and the check and update are not one atomic operation (recorded gap; see `docs/vera-design.md`, "Current state vs required"). The daily cap (50 findings per day) is a soft per-process cap. Memory lives in the shared Postgres; the Render database was upgraded to a paid plan on 2026-10-12, so the earlier 2026-10-17 expiry no longer applies. Database operations have a 5-second connect timeout and 5-second statement timeout. The UI can show a saved-findings message (`vera/memory_panel.py`), but only when the `/ask` response reports `memory_written` above 0. Live `/ask` always returns `memory_written` = 0, because findings are written as pending in a background task after the response (`main.py`), so a visitor does not see that message. Findings appear on later questions only once the owner confirms them.

**Memory kinds VERA-DAR uses, and why** (the Week 5 spectrum: working, semantic, episodic, procedural). *Semantic:* the gated findings above, one checked sentence plus its cited source per row (`vera/memory_store.py`, `vera/memory_gate.py`); these are the only memory recalled into later answers. *Procedural:* the rules in `CLAUDE.md` and `AGENTS.md`, held in version control, not in database rows (`AGENTS.md` rule 8). *Working:* the per-request context, which is the question plus the numbered abstracts retrieved for that request (`vera/grounded_ask.py`); the memory and corpus-admission code does not write the visitor's question to the database (`vera/memory_store.py`, `vera/corpus_admission.py`), and with no source found `/ask` returns a no-sources reply instead of answering from memory (`vera/ask_service.py`). *Episodic, narrowly:* corpus admission records each scored candidate with its decision and rationale, and admitted sources are kept as versioned rows with provenance (`vera/corpus_admission.py`, `db/README.md`). **Not used:** conversation or session memory, per-visitor memory, and vector memory. The `runs` rows that memory and corpus admission create carry one fixed placeholder question, not the visitor's, so no row stores a visitor's question, identity or session.

**Database choice.** One Postgres, schema `vera_vjay`: the corpus, provenance, claims and evidence are already relational there, and memory reuses the existing `claims`, `answers` and evidence tables; the memory code issues no DDL and has no DELETE path (`vera/memory_store.py`). Recall is Postgres full-text search (`to_tsvector`, `websearch_to_tsquery`, `ts_rank`) weighted by a recency half-life. **No pgvector, a stated divergence:** the Week 5 course reading names Postgres plus pgvector as the right default for most capstones, and VERA-DAR's schema has no embedding column or vector extension, and VERA-DAR has no embedding source (the #91 semantic-space code takes caller-supplied vectors, is disabled in its config and is not called by the app). Keyword matching can miss paraphrases; that cost has not been measured.

**Poisoning defences in code, and gaps.** Present: (1) owner confirm: with `confirm_required` on, as configured, `/ask` writes pending rows, recall reads only confirmed rows, and only the owner-key confirm route marks a row confirmed (`vera/memory_store.py`, `vera/memory_operator.py`, `main.py`); (2) write-gate pattern checks that block a write on instruction patterns, invisible Unicode, tool dumps, personal remarks, echoed questions and volatile terms; the gate is a structural filter, not a content check (`vera/memory_gate.py`); (3) the claim checker's prompt tells the model to treat claims and sources as untrusted data and ignore instructions in them, which is an instruction to the model, not an enforced control (`vera/claim_check.py`); (4) a new finding with the same text and evidence span as a rejected row is not queued again (`vera/memory_store.py`). Gaps: no hard delete (reject and tombstone only; deletion needs the owner's approval); no contradiction handling (a new finding is compared with stored ones only by exact text and evidence span, for de-duplication), so a reworded rejected finding can be queued again for review; and a poisoned abstract from an allowed source is not detected.

**What VERA-DAR does NOT do yet**

- The full pipeline (M2 to M6) has **not been run live end to end**; the tests use fakes and fixtures.
- The central claim (VERA-DAR's engineered answer beats a direct model answer) is **unmeasured**.
- Grounded `/ask` (2026-10-07) now meets R1's criteria for the public default, measured once (directional, small n): each cited source is a record retrieved in that request with provider, identifier, retrieval time and licence decision; citations that map to no record are removed; an answer with no valid citation becomes an insufficient-evidence reply; each cited claim is checked against the cited abstract (`vera/claim_check.py`; unsupported sentences removed, partial ones marked; a different-family checker first; fails closed). On the 30-question trace set (`scripts/grounded_trace_run.py`, `eval_results/grounded_ask_v1.json` (held-out evaluation material; not published in this repository)): unresolvable citations 0 of 57; unsupported-claim rate 9 of 105 (8.6%, Wilson 95% 4.6% to 15.5%), removed before display; partial or unsupported 37 of 105 (35%, 26.8% to 44.7%). Caveats: the checker's own error rate is unmeasured; abstracts only; false-premise questions often end as insufficient-evidence replies (corrections without citations).
- History: the API and UI were first deployed to Render and verified on 2026-10-07 (item #73, as recorded at the time; addresses are not recorded here), before grounded `/ask`. An earlier build (dabd92e) was deployed to Render and verified via `/version` on 2026-10-14 (owner's private run records, 2026-10-14); it included D6 owner-confirmed memory, the #91 scaffold with its flag off and the shared data-use disclosure #74. This reconciled branch has not been deployed. This README records only verified deploys.

**Accepted residuals (risks named, not removed)**

- A poisoned or misleading abstract from an allowed source is not detected (OWASP LLM04 and LLM08).
- Misinformation (LLM09): grounded `/ask` checks each cited sentence against the cited abstract with a model (`vera/claim_check.py`); the checker's own error rate is unmeasured, it sees abstracts only, and sentences without a citation are not checked. The trace-eval support heuristic (A6) remains unusable (TPR 0%).
- Corpus admission writes to the shared production database from visitor-driven searches (sources only, never visitor text), bounded by the request caps and by Gate A and the licence gate.
- Clients behind one IP share a limit (identity is the remote address; proxy headers are off by default).
- The title-family dedupe key can merge two different papers with the same main title in the same year; every merge is logged with its reason, so a wrong merge is visible.
- Some `/ask` 401/402/429 error details still name the provider.
- Links inside model answers are not validated beyond the citation check (LLM05).

## Current Status

*Last revised 2026-10-02 (R7a author, R11a record role; corrections from the 2026-10-02 code read).
Canonical, maintained status lives in the owner's private planning notes (not published); this section is a summary and can lag it.*

VERA-DAR is under active development as a capstone project. The 2026-09-30 directive lifted the earlier
"paused until 2026-10-15" rule; the Oct 15 - Nov 1 sprint plan is now the **fallback**, not the build window.

| Area | State (2026-10-01, corrected 2026-10-02; rows marked 2026-10-07 updated; later dates are stated inline) |
|---|---|
| Architecture and design | Complete: `docs/vera-design.md` and Addendum A (milestone status table with the 2026-10-01 addendum at A.11). |
| Service foundation | 2026-10-07: FastAPI `GET /health`, `POST /ask` (grounded, items #77/#79) and `GET /corpus-summary` (item #83); an earlier build (dabd92e) was deployed to Render and verified via `/version` on 2026-10-14 (item #73); this reconciled branch has not been deployed. Also `GET /version` and the owner-only `/memory/...` routes (D6). |
| Persistence | Built: Postgres schema `vera_vjay`, migrations 001-006 in `db/migrations/` (see `db/README.md`; 003 and 004 are not re-runnable). Migration 005 columns verified to exist live, read-only. Migration 006 (`licence_observations`, additive; file under `db/migrations/pending/`, applied 2026-10-09 with owner approval; rollback script in `db/migrations/rollback/`). (The earlier "persistence not yet implemented" line was stale.) |
| M1-M7 code | Built under `vera/` (m2_runner, m3, m4, m5, m6, m7, plus `pipeline_llm`, `pipeline_store`, `cost_ledger`) with tests. **The pipeline cannot yet run end to end**; nothing has run end to end against the live database. Missing: (a) no M1-M7 pipeline module creates `runs` rows (tests and `vera/run_mvp.py` do; memory and corpus admission also create `runs` rows, with a fixed placeholder question from config, not the visitor's question), and the M6 adapter reads `runs.question` / `runs.final_response`, which nothing writes (`finalize_run` writes `engineered_response`), so the M6 CLI fails on a new run; (b) `sub_question_ids` is read by Gate C and the context builder but set by nothing (no M3-to-M4 adapter), so Gate C returns `insufficient`; (c) the runners are not repointed to the new accounts, and `run_m2` probably cannot run under the write-only account (static reading, not run live). The plan is in the owner's private planning notes (not published). |
| What is persisted | Gate C decisions per attempt (`gate_c_decisions`), claims with fine-grained status and issue codes, answers by stage with tokens, cost and latency, relations (type and confidence only), appraisals with rubric/policy version, and immutable source versions. **M2 search cost is not persisted on `search_iterations`**: it is returned in memory and appended to a file only if `VERA_COST_LOG` is set. The inputs behind an M5 `weak` label are not stored. Live rows were not queried 2026-10-02. |
| Tests | 2026-10-12: `pytest tests`: **965 passed, 3 skipped**; release gate PASS 6/6. Includes the SQL contract test `tests/test_sql_contract.py`, whose live catalog test is opt-in (`--run-live-stages`, read-only with VERA-DAR's own `vera_eval_ro`; skipped by default in ordinary pytest; the release gate requires it and fails closed, so a skip, error or zero tests fails the gate). |
| DB accounts | 3 accounts applied 2026-10-01 (descriptions above). 2026-10-07: all three logins verified; `.env` uses VERA-DAR's own accounts only (`EXTERNAL_DB_URL`/`INTERNAL_DB_URL`/`VERA_DB_URL_RW` → `vera_claude_code_rw`, `VERA_DB_URL_WO` → `vera_pipeline_wo`, `VERA_DB_URL_RO` → `vera_eval_ro`); write-only grants exercised for corpus admission. The pipeline runners are not yet repointed. |
| Version control | 2026-10-12: `main` = `origin/main` = 7fc7fe1 (#74 disclosure and rules 1-2 rewrite), after 88dca65 (#7 opt-in live tests) and 50644f0 (D6 owner-confirmed memory, #91 scaffold). Commits and pushes happen only on the owner's request. |
| Evaluation design | Longitudinal and quarterly: 8 runs, one per quarter (default stamps Jan 2025, Apr 2025, Jul 2025, Oct 2025, Jan 2026, Apr 2026, Jul 2026, Oct 2026; **unconfirmed**), 2 tests per agent/model per run, as-of stamping by code (never in the baseline prompt), a config fingerprint, and six named contamination channels (search, fetched content, parametric knowledge, judge hindsight, rubric hindsight, cross-run carry-over). Design only: no code or schema for it exists yet. See the owner's private planning notes (not published). |
| Target | Course end 2026-10-25 (corrected 2026-10-05; earlier documents say 10-12). Demo Day is planned to use the Render deployment, pending an owner-approved deploy. |

**Open items (visibly open unless marked decided):** demo question (D0); baseline prompt lock (G2);
A2 (Anthropic key) DECIDED 2026-10-02 by R1: dropped, no Anthropic key will be used; MVP must-have #8
("draft critiqued"): direction given 2026-10-02 (a critique pass informed by the divergences found by claim
verification, feeding the existing bounded revision), plan being drafted, nothing built; deleting test row `questions.question_id=7` (needs an approved action); confirming
the 8 quarterly stamps; the staged live run; requirements freeze; provider-chain wiring for the
pipeline stages (`/ask` is wired since item #77, 2026-10-07); pg_dump and the non-Render host choice
(the Render database was upgraded to a paid plan on 2026-10-12, so the 2026-10-17 expiry no longer applies; the pg_dump and host-choice dates of 2026-10-13..16 are no longer deadline-driven; PostgreSQL stays regardless of host).

**Accepted divergences (recorded):** dev and production share one Postgres instance and schema
(separated by account only); the DB admin login is Render's original, misnamed identity; the
Render server default for password hashing is md5 (Render-managed) while the three new roles were
set with SCRAM-SHA-256 per session.

## Current status and plans

Plans and research exist as documents only; none is implemented unless stated. They live in the
owner's private planning notes (not published; named here by file name only, not linked):

-   `vera-plan-end-to-end-runner.md`: what is missing for a first end-to-end run and how to build it.
-   `vera-plan-config-single-source.md`: fix the three VERA-DAR config issues (model and judge choice
    scattered across code, an unread provider-chain file (removed 2026-10-07; `/ask` now uses `config/ask-provider-chain.json`), hardcoded judge family) with one config source,
    **planned before the MVP**; implementation not started, waiting on two R1 decisions (amend in place or a
    companion file; judge provider and key).
-   `vera-plan-m2-hardening.md`, `vera-plan-m3-gate-b-extras.md`,
    `vera-plan-m7-demo-narrative-traceability.md`, `vera-plan-cost-accounting-source-of-truth.md`,
    `vera-plan-verifier-judge-disagreement.md`, `orchestration-learning-and-critique-plan.md`: milestone
    hardening, cost accounting, judge/verifier disagreement, and the critique pass (direction D1).

Dates: MVP target Fri 2026-10-03 (aggressive); course end 2026-10-25 (corrected 2026-10-05); the fallback sprint start of Thu 2026-10-15 is retired. The Render database expiry of 2026-10-17 no longer applies (paid plan, 2026-10-12).

## Documentation Index

| Document | What it holds |
|---|---|
| [docs/README.md](docs/README.md) | Index of this repository's design documents. |
| [docs/vera-design.md](docs/vera-design.md) | Full design, invariants, and Addendum A (MVP milestones, status table, A.11 status and replan). |
| docs/VERA-OCT-15-SPRINT-PLAN.md (not published) | Fallback Oct 15 - Nov 1 sprint plan (superseded as the build window). |
| [db/README.md](db/README.md) | Schema walkthrough, migration order and re-run rules, account and password-hashing notes, live verification. |
| [.env.example](.env.example) | Environment variables and accounts, described without secrets. |

Planning records are private and not published (named here by file name only, not linked):

-   `VERA-TODO-DIGEST.md`: canonical status, task table, critical path.
-   `VERA-CAPSTONE-PLAN.md` and `VERA-CAPSTONE-FLOWS-AND-PSEUDOCODE.md`: plan, flows, accepted divergences.
-   `VERA-ROUND-1-IMPLEMENTATION-SUMMARY.md` and `VERA_STATUS_REEXAMINATION_2026-09-30.md`: Round 1 record and an earlier (superseded) status snapshot.
-   `VERA-STATE-REPORT-2026-10-01.md`: git state, proposed commit groups, pre-commit checklist.
-   `VERA-EVAL-BEST-PRACTICES.md`: longitudinal evaluation design, contamination channels, source and licence rules.

### History archives (pre-edit copies)

Dated internal history folders (private, not published): `2026-10-01-cleanup-archive-10` (latest pass; copies of
this README, `db/README.md`, `.env.example` and `environment-log-prv.md` with the prefix `vera-repo-`),
`-archive-9`, `-archive-8`, `-archive-7`, `-archive-6` (earlier VERA-DAR README, docs, db README and `.env.example` copies),
`-archive-5`, `-archive-4`, `-archive-3`, `-archive-2`, and `2026-10-01-cleanup-archive`.

## Demo-Day Success Criterion

This is the project's goal, not a result: the comparison against leading LLMs has not been run (see "Status and limitations"). The project succeeds if VERA-DAR can demonstrate that it:

> produces a grounded research response significantly superior in
> quality to direct responses to the same research question from GPT-6
> Astra and other leading LLMs, while making its evidence-selection
> process inspectable and auditable.

Every consequential evidence decision should be traceable to its source,
evidence, provenance, and evaluation method.

A sophisticated and auditable process that produces an inferior research
answer does not satisfy the project's success criterion.

## Public Disclosure Boundary

This repository is publicly accessible for educational demonstration,
evaluation, and portfolio purposes.

It describes VERA-DAR's purpose, public capabilities, interfaces,
development status, and evaluation objectives. It intentionally does not
document proprietary internal representations, decision structures,
scoring methods, algorithms, orchestration, feedback mechanisms, private
architectures, or other non-public implementation details.

Public availability of this repository should not be interpreted as
disclosure of those non-public mechanisms.

## Project

**VERA-DAR: Verifiable Evidence-based Research Answers**

TAI Labs Agentic AI Engineering Bootcamp Capstone\
2026

Copyright © 2026 Tagg Maiwald. All rights reserved.

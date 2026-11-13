# Local development

[← Project overview](../README.md)

## Requirements

- Python 3.11 or later
- A virtual environment outside the source tree, with the project's declared dependencies. Keep virtual environments, caches and build artifacts out of the repository tree. On this page, `<venv>/bin/python` stands for that environment's interpreter.
- Provider credentials for any tests or requests that require model inference

Keep credentials in a gitignored local environment file. Never commit real `.env` files or paste their contents into public logs.

## Local environment

Copy `.env.example` to a gitignored `.env` and fill in only the names you need; it documents each name. The variable names the serving code reads, and which are required, are listed once in [Deployment: environment configuration](deployment.md#environment-configuration). For local work:

- `/ask` needs `VERA_API_KEY` and, as shipped, `GROQ_API_KEY`. `OPENAI_API_KEY` is needed only if the OpenAI fallback is turned on in `config/ask-provider-chain.json`.
- A local full-document corpus needs either `VERA_FULLDOC_BUNDLE_DIR` (a bundle directory) or `VERA_FULLDOC_DB_RUN_ID` with `VERA_DB_URL_RO` (a pinned database run), not both.
- Automatic acquisition runs only when both `VERA_DB_URL_WO` and `VERA_DB_URL_RO` are set. Set `VERA_AUTOACQ_DISABLED=true` to keep it off locally.
- The configured database is the production database (see Data safety below). Leave the database URLs unset unless the task needs them.

## Current behaviour to know when developing (as of 2026-11-05, read from code and config; items marked Superseded were re-read 2026-11-13)

- The pre-retrieval gate is safety-only; there is no fixed scope list. Whether the corpus suffices is decided after retrieval, and acquisition runs rounds until about 6 full texts are held. *Update 2026-11-09: the target is 6 NEW documents per question and stored documents are not capped; see the README "Update 2026-11-09" paragraph.* Premise handling and the corpus-digest checker are in `vera/`; see [vera-design.md](vera-design.md) and [api.md](api.md) for contracts.
- Request trace: `vera/request_trace.py` with `config/request_trace.json`. New event detail keys must be added to `key_allowlist` with a reason (Deny All); question text is never an event field. Tests use fake engines: no database is touched.
- Lanes: the news lane (`config/reporting_lane.json`) is on. *Superseded 2026-11-13 (DOCS-1113): the official-sources lane (`config/subquery.json` `official_lane.enabled`) is also on at 4087b65; pages are read only for `fetch_entities` (FEC, Census Bureau).* Changing either flag changes the disclosure text (`vera.disclosure.lanes_disclosure_on()`); see [lanes-enable-checklist.md](lanes-enable-checklist.md).
- Offline checks added 2026-11-06 (NEWS-BATCH-2): `tests/test_nt3_wiki_links_1106.py`, `test_h5_enable_1106.py`, `test_f1_query_1106.py`, `test_relevance_r1r4_1106.py`, `test_scope_count_1106.py`, `test_r8_headline_1106.py`, `test_progress_count_1106.py`, `test_ask_terms_notice_1106.py`, `test_news_access_h19.py`, `test_news_access_report.py`. Report of outlets that resolve to link-only: `python3 -I -B scripts/news_access_report.py` (offline, no database, no network). `tests/test_pdf_extract_worker.py` skips until a worker module exists.
- Licence allowlist: `config/licence_allowlist.json` is the seed (Deny All: unlisted licences are not stored); `vera/licence_gate.py` reads it. The planned DB table (draft 016) is not in this tree.
- Config sanity: `<venv>/bin/python -I scripts/check_config.py`.

## Batch-6 flags to know (as of 2026-11-10, read from `config/` at d3fe0fe; offline-tested, live unverified)

- Off: `hybrid_retrieval.enabled` and `rerank.enabled`; `chunk_value.enabled`; every module in `settings.read_from_db` (`{}`); `taxo_router.mode`; `auto_acquire.json` `arxiv_route.enabled` and `arxiv_portal.openalex.enabled`; `reference_portals.json` `adapters.openalex_rw`.
- On: `portal_follow.enabled`; `list_ingest.enabled` (the route still answers 503 until `VERA_INGEST_API_KEY` is set); all `ddg_reduce.json` switches; `chunk_taxo.json` `params.multi.enabled`; `reference_portals.json` `adapters.inspire_refs`; `chunk_usage.enabled` (inert until the tables of draft migration 025 exist, and `demote.scale` is 0).
- Migrations 025 and 026 are in `db/migrations/pending/` (025 as a `.draft` file, 026 as `026_settings.sql`); no code path applies them. Observation as of 2026-11-13, per project records (the database was not checked): 026 is not applied (the live catalog check on 2942af4 showed only the migration-026 gap). For 025 no record of an apply was found; it is a draft file and no code path applies it. Config names the Cloudflare variables `CLOUDFLARE_ACCOUNT_ID` and `CLOUDFLARE_API_TOKEN` (names only; see [Deployment](deployment.md)).
- Host policy is data: change a host's role only by a reviewed edit of `config/host_roles.json`; `vera/host_role.py` fails closed when the file is invalid.

## Features that ship off (as of 2026-11-05)

Available in code, off by default, and each needs an owner step before it does anything: `retention_purge.enabled` (`config/request_trace.json`; migration 018 draft), `enabled` in `config/topic_tagging.json` (migrations 017 drafts; the tagger also needs `micro_topic.registry_table.read_from_db` true in `config/auto_acquire.json`, false today, so tagger and digest read the same DB registry; the digest's table mode needs `topics_source` `table_if_present` in the `corpus_description` block of `config/ask-provider-chain.json`, not set today so `model_clusters` applies), `read_from_db` in `config/list_tables.json` (migration 016, staged in `work/LISTS-TO-DB-1104/staged/`). Do not apply migrations from an agent session; status is owned by `db/README.md`. Tests for them are offline (`tests/test_retention_purge_1105.py`, `tests/test_topic_tagging_1104.py`, `tests/test_lists_loader_1105.py`). Parallel `/ask` lookups with per-host serialisation and Crawl-delay are on (`tests/test_ask_parallel_1105.py`, `tests/test_crawl_delay_1105.py`). *Superseded in part 2026-11-08 (DOCS-1108): `config/request_trace.json` now has `retention_purge.enabled` true (commit 6ef1287, OD-1105-PURGE-ENABLE), and migration 018 was applied by the owner on 2026-11-08 (owner-run apply; not independently checked against the database), so the automatic purge is enabled; status is owned by `db/README.md`.* The other flags named here were not re-checked.

## Run the API

From the repository root:

```bash
./run.sh
```

The default development address is `http://127.0.0.1:8001` (a port argument or `PORT` overrides it). `run.sh` is for local development only: it runs with auto-reload and sets `VERA_ENV_CONTEXT=local_dev`. The container entry point `start.sh` refuses to run outside the hosting platform. Check `/health` to verify readiness. The local FastAPI documentation is at `/docs`.

To select a different port:

```bash
./run.sh 8000
```

## Run the Streamlit app (multipage: Ask, Report, Trace Eval, How to use, About)

```bash
./run_demo.sh
```

The script serves on port 8502. For the labelled fixture view of the Report page:

```bash
VERA_DEMO_SOURCE=fixture ./run_demo.sh
```

**Note:** The fixture selector applies to the Report page; it does not replace the `/ask` research backend with fixture responses.

## Tests

The offline suite runs with:

```bash
<venv>/bin/python -I scripts/vera_test_runner.py -- tests -q
```

The full suite takes about 3 minutes (185 s in the 2026-11-06 verification run). Tests run offline; the test boundary refuses network and database connections. Live-stage tests need the pytest flag `--run-live-stages` (the live API test also needs API_BASE_URL); run them with plain `pytest`, because the project test runner refuses `--run-live-stages`. Tests that read documentation text include `tests/test_dar_migration_007_roles.py`, `tests/test_integrate_1104_db_flags.py` (pins the 015 row in `db/README.md`), `tests/test_s1_fulldoc_disclosure.py` and `tests/test_wp6p2b_robots_signal_lanes_1103.py`; rerun them after editing those docs. Record the actual command, revision and counts when reporting a result. No test count is stored in the repo and no single number is authoritative: reported counts differ by revision and branch. State the command, the git revision and the pass/fail/skip counts you observed, with the date.

## Data safety

Do not direct experiments or integration tests at production database credentials. Some legacy integration paths can affect a shared database. Prefer fake or isolated test resources.

For service deployment rather than local development, see [Deployment](deployment.md).

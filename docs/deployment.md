# Deployment on Render

[← Project overview](../README.md)

This describes a **two-service, owner-managed** Docker deployment. It is not evidence that any service has been created or passed live acceptance. Render is the interim host; a non-Render host is planned (Postgres remains the database technology). For the current project status and timeline, see the [project overview](../README.md#status). Status of this page as of 2026-11-05 (repo HEAD cd81eab plus uncommitted working-tree edits; the 2026-10-30 text is revised, not deleted). Auto-Deploy is off on both services; deploys are owner-controlled.

## Services

Both services are built independently from the repository root.

| Setting | API service | UI/demo service |
|---|---|---|
| Runtime | Docker | Docker |
| Branch (recommended) | `main` | `main` |
| Root directory | Blank | Blank |
| Dockerfile | `./Dockerfile` | `./Dockerfile` |
| Docker build context | `.` | `.` |
| Docker command | `./start.sh api` | `./start.sh ui` |
| Health check | `/health` | `/_stcore/health` |
| Automatic deployment (recommended) | Off | Off |

The frontend may be given any service name; the name does not change its Docker command.

## Environment configuration

This is the canonical list of the environment variables the serving code reads. It gives names only. Set values in the hosting platform's secret settings, never in tracked files. The templates `.env.render.api.example` and `.env.render.ui.example` list the names each service needs.

### API service

| Name | Needed | Purpose |
|---|---|---|
| `VERA_API_KEY` | Required | Client key for `/ask` and other protected routes. |
| `GROQ_API_KEY` | Required for `/ask` as shipped | The answer provider chain is Groq first (`config/ask-provider-chain.json`). A chain entry whose key is unset is skipped, and with the OpenAI fallback off no other provider remains. |
| `OPENAI_API_KEY` | Only if the OpenAI fallback is on | `openai_fallback.enabled` ships `false`: the API then makes no OpenAI call, and a set key is ignored. Turning it on is a reviewed config change plus a deploy, and needs a valid key. |
| `VERA_DB_URL_RO` | Optional | Read-only database connection: aggregate corpus description, the database corpus source, and reading acquired documents. |
| `VERA_DB_URL_WO` | Optional; needed for acquisition | Write-only application account that automatic acquisition uses to store acquired documents. Acquisition runs only when both `VERA_DB_URL_WO` and `VERA_DB_URL_RO` are set. It is also the INSERT-only connection for the request trace and the stored corpus digest (see "Request trace and stored digest" below). |
| `VERA_AUTOACQ_DISABLED` | Optional (kill switch) | When set to a true-like value (`true`, `1`, `yes`, `on`), automatic acquisition is off even though `config/auto_acquire.json` enables it. Leave unset for normal operation. |
| `CLOUDFLARE_ACCOUNT_ID`, `CLOUDFLARE_AI_TOKEN` | Optional | Cloudflare Workers AI credentials for the chunk-taxonomy job and hybrid retrieval query embeddings (`vera/chunk_taxo_embed.py`; read at call time, never logged). Without them no embedder is created: the taxonomy job does not run and retrieval is word-only. Names only; set values in the host's secret settings. |
| `CHUNK_TAXO_EMBEDDER`, `CHUNK_TAXO_MAX_ROWS`, `CHUNK_TAXO_MAX_SECONDS`, `CHUNK_TAXO_RSS_SOFT_MB`, `CHUNK_TAXO_RSS_HARD_MB`, `CHUNK_TAXO_EMBED_BATCH` | Optional | Overrides for the chunk-taxonomy job (`vera/chunk_taxo_job.py`, `vera/chunk_taxo_embed.py`): embedder name, row and time limits, memory guard limits (MB) and embed batch size. Unset, the values in `config/chunk_taxo.json` apply. |
| `VERA_FULLDOC_BUNDLE_DIR` | Optional | Full-document corpus from a local bundle directory. |
| `VERA_FULLDOC_DB_RUN_ID` | Optional | Full-document corpus from a pinned database run, read through `VERA_DB_URL_RO`. |
| `VERA_OWNER_KEY` | Optional | Operator key for the memory review routes, per `.env.example`; unset keeps those routes closed. |
| `VERA_TRACE_VIZ_KEY` | Optional | Key for the per-trace detail route; unset disables that route (403). |
| `VERA_ENV_CONTEXT` | Platform-dependent | `start.sh` accepts `production` as an alternative to `RENDER=true`. |
| `VERA_PUBLIC_MODE`, `VERA_TRUST_PROXY` | Leave unset | As in the provided templates. |
| `RENDER`, `RENDER_GIT_COMMIT`, `PORT` | Set by the platform | Do not set them manually. `/version` reports `RENDER_GIT_COMMIT`. |

### UI service

| Name | Needed | Purpose |
|---|---|---|
| `VERA_API_BASE_URL` | Required | API base URL, without `/ask`. |
| `VERA_API_KEY` | Required | Must match the API service's key. |
| `VERA_DEMO_SOURCE` | Optional | `fixture` labels the demonstration report fixture. It does not turn `/ask` into an offline fixture. |

### Credentials: least privilege

- Give each service only the credentials it uses. The UI calls the API over HTTP and needs no database credentials. Do not configure any database URL on the UI service.
- On the API, the read-only URL is enough for corpus reading. Configure the write-only application URL (`VERA_DB_URL_WO`) only if automatic acquisition should store documents. Never configure an administrative or read-write account URL on a serving service.
- To stop acquisition without a code or config change, set `VERA_AUTOACQ_DISABLED=true` on the API service and restart it. Removing `VERA_DB_URL_WO` also stops it.

### Request trace and stored digest

`config/request_trace.json` ships `enabled`, `log_sink.enabled`, `db_sink.enabled` and `digest_store.enabled` true (as read from the file 2026-11-05); `log_sink.include_question` ships false. Migration 015 is recorded as applied by the owner on 2026-11-04 (see [db/README.md](../db/README.md); the verify output was not seen by agents). With no database URLs set, the sinks do nothing and requests are unaffected (offline-tested; not observed live). The trace uses the existing write-only and read-only URLs: no extra variable is needed. The visitor question is stored only in the database requests table, owner-only, and purged after 30 days by the owner script `work/PERF-MISS-1104/trace/purge_trace.sh`; hosting-platform logs are not covered by that purge, which is why the log line carries only the question length. The request id is returned only in the `X-VERA-Trace-Id` response header, never in the UI. Owner scripts (not run by agents): `work/PERF-MISS-1104/trace/show_trace.sh`, `pull_trace.sh`, `purge_trace.sh`.

Ships off (2026-11-05, read from config): `retention_purge.enabled` in the same file (daily 30-day trace purge, started at boot when on; needs migration 018, a draft, not applied; owner step), `enabled` in `config/topic_tagging.json` (needs 017 drafts) and `read_from_db` in `config/list_tables.json` (needs 016, staged under `work/LISTS-TO-DB-1104/staged/`). Turning any of them on without its migration fails closed or falls back to the seed files; do not enable before the owner applies the migration. *Superseded in part 2026-11-08 (DOCS-1108): `config/request_trace.json` now has `retention_purge.enabled` true (commit 6ef1287, OD-1105-PURGE-ENABLE), and migration 018 was applied by the owner on 2026-11-08 (owner-run apply; not independently checked against the database), so the automatic purge is enabled; status is owned by `db/README.md`.*

### Lanes and acquisition switches (config, not environment)

- News lane (`config/reporting_lane.json`): `enabled` is true in the working tree (owner decision 2026-11-05; GDELT only, `fallback_order` is `["gdelt"]`). Outlets are searched, but only hosts listed with `terms_verdict` `allow` are read in full; others are title and link only. **Untested live**: offline-tested only, and a full-suite run recorded in `work/NEWS-LANE-1104/checkpoint.md` had unattributed failures. Check the [lanes checklist](lanes-enable-checklist.md) before relying on it.
- Official-sources lane: built, `official_lane.enabled` in `config/subquery.json` ships false (and `site_search` false).
- Automatic acquisition: `config/auto_acquire.json` has inline acquisition on with up to 6 acquisition rounds (`inline.acq6_rounds_enabled`, `bounds.acq_round_target_new_docs` 6). Corpus sufficiency is decided after retrieval, not by a fixed scope. Kill switch: `VERA_AUTOACQ_DISABLED`.
- Live statistics (`config/live_stats.json`) ship off; they need migration 013, which is not applied.

### Full-document path

The full-document path stays off unless `fulldoc.enabled` is true in `config/fulldoc_answer.json` and a corpus source is configured. There are two sources: a local bundle directory (`VERA_FULLDOC_BUNDLE_DIR`) or a pinned database run (`VERA_FULLDOC_DB_RUN_ID`). Configuring both is refused: the base corpus is reported unusable rather than silently preferring one. Acquired documents are served only next to the database-backed base corpus. This describes what the code implements, not that either source has been exercised on the deployment. To see the state, call `/health/dar` with the API key and read `fulldoc.status` (`disabled`, `not_configured`, `ok`, `unusable` or `error`).

## Start-up and checks

`start.sh` is the container entry point and refuses to start unless `RENDER=true` or `VERA_ENV_CONTEXT=production` is set. It listens on `PORT` (default 8000) and runs a single worker, because the public request caps are in-memory per process.

After deploying, verify `/health`, `/version` and that an unauthenticated `POST /ask` is rejected, and check the visitor disclosure against the live configuration. The owner wrapper for the post-deploy build check is `work/RELEASE-VERIFY-1104/owner_deploy_check.sh` (it loads the live host from the gitignored host file into a variable and never prints it; `--dry-run` makes no call). Deploys stay owner-run.

Do not publish real service hostnames, tokens, passwords, or database connection strings.

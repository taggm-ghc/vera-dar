# Deployment on Render

[← Project overview](../README.md)

This describes a **two-service, owner-managed** Docker deployment. It is not evidence that any service has been created or passed live acceptance. Render is the interim host; a non-Render host is planned (Postgres remains the database technology). For the current project status and timeline, see the [project overview](../README.md#status). Status of this page as of 2026-10-30. Auto-Deploy is off on both services; deploys are owner-controlled.

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
| `VERA_DB_URL_WO` | Optional; needed for acquisition | Write-only application account that automatic acquisition uses to store acquired documents. Acquisition runs only when both `VERA_DB_URL_WO` and `VERA_DB_URL_RO` are set. |
| `VERA_AUTOACQ_DISABLED` | Optional (kill switch) | When set to a true-like value (`true`, `1`, `yes`, `on`), automatic acquisition is off even though `config/auto_acquire.json` enables it. Leave unset for normal operation. |
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

### Full-document path

The full-document path stays off unless `fulldoc.enabled` is true in `config/fulldoc_answer.json` and a corpus source is configured. There are two sources: a local bundle directory (`VERA_FULLDOC_BUNDLE_DIR`) or a pinned database run (`VERA_FULLDOC_DB_RUN_ID`). Configuring both is refused: the base corpus is reported unusable rather than silently preferring one. Acquired documents are served only next to the database-backed base corpus. This describes what the code implements, not that either source has been exercised on the deployment. To see the state, call `/health/dar` with the API key and read `fulldoc.status` (`disabled`, `not_configured`, `ok`, `unusable` or `error`).

## Start-up and checks

`start.sh` is the container entry point and refuses to start unless `RENDER=true` or `VERA_ENV_CONTEXT=production` is set. It listens on `PORT` (default 8000) and runs a single worker, because the public request caps are in-memory per process.

After deploying, verify `/health`, `/version` and that an unauthenticated `POST /ask` is rejected, and check the visitor disclosure against the live configuration.

Do not publish real service hostnames, tokens, passwords, or database connection strings.

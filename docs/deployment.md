# Deployment on Render

[← Project overview](../README.md)

This describes a **two-service, owner-managed** Docker deployment. It is not evidence that any service has been created or passed live acceptance. Render is a temporary course demonstration host.

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

**API service:** `VERA_API_KEY` and `OPENAI_API_KEY` are required for `/ask`. `GROQ_API_KEY` is optional for a free-tier-first provider chain. A read-only database connection (`VERA_DB_URL_RO`) is optional for the aggregate corpus description. Do not configure write-capable credentials for the demonstration. The UI service needs no database credentials.

**UI service:** `VERA_API_BASE_URL` must point to the API base URL (without `/ask`), and `VERA_API_KEY` must match the API service's key. `VERA_DEMO_SOURCE=fixture` labels the demonstration report fixture, but does not turn `/ask` into an offline fixture.

Leave `VERA_PUBLIC_MODE` and `VERA_TRUST_PROXY` unset, as in the provided environment templates. `RENDER` is set by the hosting platform and is not a secret to configure manually. The optional full-document path stays off unless both `fulldoc.enabled` is true and `VERA_FULLDOC_BUNDLE_DIR` is set.

`start.sh` is the container entry point and refuses to start unless `RENDER=true` or `VERA_ENV_CONTEXT=production` is set. It listens on `PORT` (default 8000) and runs a single worker, because the public request caps are in-memory per process.

After deploying, verify `/health`, `/version` and that an unauthenticated `POST /ask` is rejected, and check the visitor disclosure against the live configuration.

Do not publish real service hostnames, tokens, passwords, or database connection strings.

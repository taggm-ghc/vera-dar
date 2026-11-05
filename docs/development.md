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

## Current behaviour to know when developing (as of 2026-11-05, read from code and config)

- The pre-retrieval gate is safety-only; there is no fixed scope list. Whether the corpus suffices is decided after retrieval, and acquisition runs rounds until about 6 full texts are held. Premise handling and the corpus-digest checker are in `vera/`; see [vera-design.md](vera-design.md) and [api.md](api.md) for contracts.
- Request trace: `vera/request_trace.py` with `config/request_trace.json`. New event detail keys must be added to `key_allowlist` with a reason (Deny All); question text is never an event field. Tests use fake engines: no database is touched.
- Lanes: the news lane (`config/reporting_lane.json`) is on in the working tree and the official-sources lane (`config/subquery.json` `official_lane.enabled`) is off. Changing either flag changes the disclosure text (`vera.disclosure.lanes_disclosure_on()`); see [lanes-enable-checklist.md](lanes-enable-checklist.md).
- Licence allowlist: `config/licence_allowlist.json` is the seed (Deny All: unlisted licences are not stored); `vera/licence_gate.py` reads it. The planned DB table (draft 016) is not in this tree.
- Config sanity: `<venv>/bin/python -I scripts/check_config.py`.

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

Tests run offline; the test boundary refuses network and database connections. Live-stage tests need the pytest flag `--run-live-stages` (the live API test also needs API_BASE_URL); run them with plain `pytest`, because the project test runner refuses `--run-live-stages`. Tests that read documentation text include `tests/test_dar_migration_007_roles.py`, `tests/test_integrate_1104_db_flags.py` (pins the 015 row in `db/README.md`), `tests/test_s1_fulldoc_disclosure.py` and `tests/test_wp6p2b_robots_signal_lanes_1103.py`; rerun them after editing those docs. Record the actual command, revision and counts when reporting a result. No test count is stored in the repo and no single number is authoritative: reported counts differ by revision and branch. State the command, the git revision and the pass/fail/skip counts you observed, with the date.

## Data safety

Do not direct experiments or integration tests at production database credentials. Some legacy integration paths can affect a shared database. Prefer fake or isolated test resources.

For service deployment rather than local development, see [Deployment](deployment.md).

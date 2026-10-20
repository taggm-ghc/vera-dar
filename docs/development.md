# Local development

[← Project overview](../README.md)

## Requirements

- Python 3.11 or later
- A virtual environment (`.venv`) and the project's declared dependencies
- Provider credentials for any tests or requests that require model inference

Keep credentials in a gitignored local environment file. Never commit real `.env` files or paste their contents into public logs.

## Run the API

From the repository root:

```bash
./run.sh
```

The default development address is `http://127.0.0.1:8001`. Check `/health` to verify readiness. The local FastAPI documentation is at `/docs`.

To select a different port:

```bash
./run.sh 8000
```

## Run the Streamlit demonstration

```bash
./run_demo.sh
```

The demo script serves on port 8502. For the labelled fixture view of the demonstration reports:

```bash
VERA_DEMO_SOURCE=fixture ./run_demo.sh
```

**Note:** The fixture selector applies to a demonstration report page; it does not replace the `/ask` research backend with fixture responses.

## Tests

The offline suite runs with:

```bash
.venv/bin/python -I scripts/vera_test_runner.py -- tests -q
```

If your virtual environment uses a different path, substitute its Python executable. Tests run offline; the test boundary refuses network and database connections. Live-stage tests need the pytest flag `--run-live-stages` (the live API test also needs API_BASE_URL); run them with plain `pytest`, because the project test runner refuses `--run-live-stages`. Record the actual command, revision and counts when reporting a result.

## Data safety

Do not direct experiments or integration tests at production database credentials. Some legacy integration paths can affect a shared database. Prefer fake or isolated test resources.

For service deployment rather than local development, see [Deployment](deployment.md).

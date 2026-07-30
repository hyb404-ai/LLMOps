# Repository Guidelines

## Project Structure & Module Organization

This repository contains learning notes in `record.md` and the Flask backend in `llmops-api/`. Run application commands from `llmops-api/`.

- `App.py` is the executable service entry point.
- `app/` owns application creation and HTTP setup.
- `config/` contains default and environment-driven configuration.
- `internal/` contains application layers: `router/` for endpoints, `service/` for business logic, `schema/` for request/response models, `model/` for persistence models, and `core/` for LLM-related integrations.
- `pkg/` holds reusable extensions such as OAuth providers.
- `test/` contains demos and automated tests. Use `test_*.py` for new test files.
- `storage/` is local runtime storage; keep only the committed `.gitkeep` placeholder.

## Build, Test, and Development Commands

Create and activate an isolated environment before installing packages:

```bash
cd llmops-api
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python App.py
```

The service exposes a health endpoint at `http://127.0.0.1:5000/api/v1/health`.

Run a focused script with the project interpreter, for example:

```bash
.venv/bin/python test/injector_demo.py
```

No test runner, formatter, linter, or coverage threshold is configured yet. When adding tests, prefer `pytest` and document any new developer dependency in `requirements.txt` or a dedicated development requirements file.

## Coding Style & Naming Conventions

Use Python 3.9-compatible syntax, four-space indentation, UTF-8 source files, and short module docstrings for public modules. Follow PEP 8 naming: `snake_case` for modules, functions, and variables; `PascalCase` for classes; `UPPER_CASE` for constants. Keep routers thin: validate input in `schema/`, place orchestration in `service/`, and isolate external LLM/database code in `core/` or `extension/`.

## Testing Guidelines

Place tests under `llmops-api/test/`, name files `test_<feature>.py`, and name cases `test_<expected_behavior>()`. New endpoint work should cover both a successful response and a meaningful invalid/error path. Avoid naming a local module after a third-party package (for example, use `injector_demo.py`, not `injector.py`) to prevent import shadowing.

## Commit & Pull Request Guidelines

The history currently contains only an initial commit, so no established convention exists. Use concise imperative subjects, such as `Add health-check route` or `Fix Injector binding demo`. Keep commits focused. Pull requests should explain the change, list validation commands run, link related issues when applicable, and include request/response examples for API changes. Never commit `.venv/`, `.env`, credentials, generated caches, or local storage data; update `.env.example` when adding configuration keys.

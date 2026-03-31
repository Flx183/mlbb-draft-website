# Backend

FastAPI backend plus modeling and data-refresh scripts for MLBB draft recommendations.

## Install

From the repository root:

```bash
python -m pip install -r backend/requirements.txt -r backend/requirements-dev.txt
```

## Run

```bash
python -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

## Environment

Create `backend/.env` from `backend/.env.example` when running scripts that need Liquipedia API access.

## Checks

```bash
python -m pytest -q
ruff check backend tests
```

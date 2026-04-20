# Contributing

## Development setup

```bash
uv sync --extra dev
uv run pre-commit install
```

## Before opening a pull request

```bash
uv run pre-commit run --all-files
uv run pytest --cov=evalf --cov-report=term-missing
uv build
uv run python -m twine check dist/*
```

## Guidelines

- Keep changes scoped and include tests for behavior changes.
- Preserve the public Python and CLI contracts unless the pull request explicitly documents a breaking change.
- Update `README.md` and `CHANGELOG.md` when user-facing behavior changes.

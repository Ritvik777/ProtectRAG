# Contributing to ProtectRAG

Thanks for helping improve ProtectRAG. This document is for anyone who wants to **report issues**, **propose features**, or **submit pull requests**.

## Ground rules

- **Be respectful.** Assume good intent; disagree on technical merits.
- **Security-sensitive issues** (e.g. bypasses of screening logic) should **not** be filed as public issues first — see [SECURITY.md](SECURITY.md).
- **Small, focused PRs** are easier to review than large refactors mixed with unrelated changes.

## Development setup

```bash
git clone https://github.com/Ritvik777/ProtectRAG.git
cd ProtectRAG
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

The `[dev]` extra installs pytest, httpx, fakeredis, and build tools. For optional integration smoke tests in your own environment you can also install `langchain`, `llamaindex`, or `fastapi` extras as needed.

## Running tests

```bash
pytest tests/ -v
```

Run from the repository root with the editable install above (or set `PYTHONPATH=src`). All tests should pass before you open a PR.

## Making changes

1. **Fork** the repository and create a **branch** from `main` (e.g. `fix/scanner-false-positive`, `feat/async-ingest`).
2. **Match existing style**: typing, imports, docstrings at the same level as surrounding code, no unnecessary comments.
3. **Add or update tests** for behavior you change (especially `scanner.py`, `ingest.py`, `llm.py`, and integrations).
4. If you change **heuristic rules** or the **golden dataset**, run the full suite and consider impact on precision/recall (see `run_eval_dataset` / golden tests).
5. **Update the root [README.md](README.md)** if you add user-facing APIs, new extras, or important configuration.

## Pull request checklist

- [ ] Tests pass locally (`pytest tests/ -v`).
- [ ] New functionality is covered by tests when practical.
- [ ] Public API changes are reflected in README (and `__init__.py` exports if applicable).
- [ ] No unrelated formatting or drive-by refactors in the same PR.

## Where things live

| Path | Purpose |
|------|---------|
| `src/protectrag/` | Library source (`scanner`, `ingest`, `llm`, `async_api`, `retrieval`, integrations, etc.) |
| `tests/` | Pytest suite |
| `src/protectrag/data/golden_v1.json` | Golden eval dataset |
| `docs/` | Extra docs (root README is the main guide) |
| `notebooks/` | Example eval notebook |

## Questions

Open a [GitHub issue](https://github.com/Ritvik777/ProtectRAG/issues) for bugs, feature requests, or clarification. For usage questions, include Python version, `protectrag` version, and a minimal code snippet if possible.

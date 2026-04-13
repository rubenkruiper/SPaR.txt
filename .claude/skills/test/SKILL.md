---
name: test
description: Run the iReC test suite with poetry. Use this to verify that code changes don't break existing tests. RuntimeWarnings (e.g. numpy float32 overflow) are expected and should be ignored — only actual test failures matter.
argument-hint: [optional: path or test id to run a subset, e.g. "tests/test_graphrag_utils.py::TestRetriever"]
---

Run the test suite using:

```bash
poetry run pytest $ARGUMENTS -q
```

If `$ARGUMENTS` is empty, run the full suite:

```bash
poetry run pytest -q
```

## Interpreting results

- **Passed / failed counts** are the only signal that matters.
- **RuntimeWarnings** (numpy float32 overflow, divide by zero, invalid value in matmul) are known artefacts of mock embeddings in tests and must be **ignored** — they do not indicate a bug.
- A run is clean if it ends with `N passed` and zero failures, regardless of warning count.

## Notes

- Always run from `/Users/rubenk/dev/irec/`
- Test files live in `tests/`
- Config is in `pyproject.toml` under `[tool.pytest.ini_options]`

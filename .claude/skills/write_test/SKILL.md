---
name: write_test
description: Write new pytest tests for iReC GraphRAG code. Use this when adding tests for new or changed functionality. Enforces project test conventions and a hard rule against backwards-compatibility shims.
argument-hint: [required: describe what to test, e.g. "GraphIndex.lookup with multi-hop expansion"]
---

Write tests for: **$ARGUMENTS**

## Hard rules

1. **No backwards-compatibility logic.** Do not add `try/except ImportError`, version checks, `hasattr` guards, or any other shim that exists only to support old interfaces. If an interface changed, update the call site — don't paper over it.
2. **No mocking the real graph.** The module-scoped `graph_index` fixture loads `data/graph_data/initial_graph.ttl` once for the session. Reuse it; don't replace it with a synthetic graph unless you are specifically testing graph-construction logic.
3. **No network calls.** Mock the Anthropic client using `pytest-mock`. Use the `_tool_use_response` / `_text_response` helpers already defined in `tests/test_graphrag_utils.py`.
4. **No `sleep` or timing-dependent assertions.**
5. **RuntimeWarnings from numpy are expected** when mock embeddings are zero vectors. Do not add `pytest.warns`, `filterwarnings`, or any suppression — just let them fire. Tests pass or fail on assertions, not warnings.

## Conventions to follow

- Group tests in a class named `Test<ClassName>` or `Test<Feature>`.
- Use the module-scoped `graph_index` fixture (defined in `tests/test_graphrag_utils.py`) for any test that needs a live graph.
- For `GraphRAGPipeline` tests, inject a pre-loaded `GraphIndex` via `object.__new__(GraphRAGPipeline)` to skip disk I/O.
- Known stable anchors you can assert against:
  - `SPAN_VENTILATION = SPANS[urllib.parse.quote("ventilation")]`
  - `SPAN_MEANS_OF_ESCAPE = SPANS[urllib.parse.quote("means of escape")]`
  - `SPAN_STAIRWAY = SPANS[urllib.parse.quote("stairway")]`
  - `CONCEPT_FIRE_ALARM = CONCEPTS["100"]` (prefLabel: "fire alarm system")
- Use `pytest.mark.parametrize` for multiple similar cases rather than duplicating test bodies.
- Keep each test focused on one behaviour. The test name should describe the behaviour, not the implementation.

## File to add tests to

`tests/test_graphrag_utils.py` — append to the relevant `Test*` class, or add a new class at the end of the file if testing a new component.

## After writing

Run `/test` to confirm all tests pass.

---
name: plan
description: Produce a step-by-step implementation plan for a goal and wait for explicit approval before touching any code or files. Use this when the user asks "how would you…", "what's the best way to…", or says "plan" / "/plan".
argument-hint: [goal or feature to plan]
---

The user has invoked the planning skill. Your job is to think through the problem and present a clear, structured plan — then **stop and wait for feedback** before making any changes.

## How to behave

1. **Read, don't write.** You may use Read, Glob, Grep, and Bash (read-only commands like `git log`, `ls`) to understand the current state of the code. Do not edit or create any files.
2. **Think out loud.** Walk through your reasoning — what the goal requires, what already exists, and what tradeoffs exist between approaches.
3. **Present the plan** in the format below.
4. **Stop.** End your response with a clear question asking the user whether to proceed, adjust the plan, or take a different approach entirely. Do not start implementing until the user explicitly says to go ahead.

## Plan format

Structure your plan like this:

---

### Goal
One or two sentences restating the goal in your own words, to confirm shared understanding.

### Context
What you found in the codebase that is relevant — files, functions, data structures, existing patterns. Keep this concise; link to specific file paths and line numbers where useful.

### Approach
The strategy you recommend, and briefly why you prefer it over alternatives.

### Steps

| # | What | Where | Notes |
|---|------|-------|-------|
| 1 | Short description of the action | `path/to/file.py` | Any caveats or decisions needed |
| 2 | … | … | … |

### Open questions
Anything that needs a decision from the user before you can proceed — missing requirements, ambiguous behaviour, architectural choices.

### What this plan does NOT change
Call out things that are explicitly out of scope, so the user knows the boundaries.

---

After presenting the plan, ask:

> **Ready to proceed with this plan, or would you like to adjust anything?**

Do not take any further action until the user responds.

## Things to keep in mind for this project

- The pipeline is notebook-first (`.ipynb` files run top to bottom); prefer adding new logic to `utilities/` and importing it into notebooks rather than putting all logic inline in cells.
- Python 3.9 only — no 3.10+ syntax.
- Dependencies are pinned (`requirements.txt`); flag if the plan requires a new package and suggest a compatible version.
- The KG lives in `data/graph_data/`; any plan that reads or writes graph data should note the file format (TTL vs RDF/XML) and whether a graph reload is needed.
- The `graphrag` branch is the active development branch; note if any step would affect the `main` branch or existing notebooks that are already working.

---
name: spar
description: Run SPaR.txt inference on regulatory text and produce an aggregated inventory of extracted entities — objects, actions, functions, and discourse markers (including multi-word expressions). Use this to understand what technical terms and entity types appear in a text before further processing. Pass text directly or a file path, e.g. `/spar "The door shall be fire-resisting."` or `/spar data/my_doc.txt`.
argument-hint: [text to parse, or path to a .txt file]
---

You are running SPaR.txt inference to build an entity inventory for a piece of regulatory text.
Your goal is to extract and aggregate all spans predicted by the model — objects, actions, functions, and discourse markers — so the user (and you) understand what technical entities occur in the text before any downstream task.

---

## Step 1 — Precondition check

Before anything else, verify the trained model exists:

```bash
ls trained_models/model.pt 2>/dev/null && echo "Model found." || echo "Model NOT found."
```

If the model is not found, stop and tell the user:
> No trained model found at `trained_models/model.pt`. Run `poetry run python run_tagger.py` to train one (~20 min on CPU) before using this skill.

Do not proceed without a model.

---

## Step 2 — Resolve input

`$ARGUMENTS` is either:
- A path to a `.txt` file → read the file contents
- Raw text → use as-is
- Empty → check whether `/tmp/spar_input.txt` exists (left by `/read_pdf`); if so, use that file. Otherwise use this default example:
  > "The structural load-bearing elements shall be designed to resist fire for a minimum period. All escape routes must remain accessible during evacuation."

**Determine which case applies**, then write the resolved text to a temp file using the Write tool (skip this if the file is already `/tmp/spar_input.txt`):

```
Write to: /tmp/spar_input.txt
Content: <the resolved text>
```

This avoids shell-quoting issues when passing arbitrary text to Python.

---

## Step 3 — Run inference and aggregate spans

Run the following script. It splits the text into sentences with pysbd, runs SPaR.txt inference on all sentences, then aggregates spans by type across the full text.

```bash
poetry run python -c "
import json
from pathlib import Path
from collections import Counter
import pysbd
import spar_serving_utils as su
from spar_api_utils import SparPredictor

text = Path('/tmp/spar_input.txt').read_text()

# Sentence splitting
seg = pysbd.Segmenter(language='en', clean=False)
sentences = [s.strip() for s in seg.segment(text) if len(s.strip()) > 10]

# Inference
predictor = SparPredictor(model_dir=Path('trained_models/'))
predictions = predictor.predict_sentences(sentences)

# Aggregate spans across all sentences
spans_by_type = {t: [] for t in ['obj', 'act', 'func', 'dis']}
sent_results = []
for pred, sent in zip(predictions, sentences):
    result, _ = su.parse_spar_output(pred)
    sent_results.append({'sentence': sent, 'spans': result})
    for span_type, spans in result.items():
        spans_by_type[span_type].extend(spans)

# Count and rank spans per type
inventory = {}
for span_type, spans in spans_by_type.items():
    counts = Counter(spans)
    inventory[span_type] = [
        {'span': span, 'count': count, 'words': len(span.split())}
        for span, count in counts.most_common()
    ]

print(json.dumps({
    'num_sentences': len(sentences),
    'inventory': inventory,
    'sentences': sent_results,
}, indent=2))
" 2>/dev/null
```

---

## Step 4 — Display the entity inventory

Parse the JSON output and present it as follows. **Do not dump the raw JSON** — format it for readability.

### Summary line
> Analysed **N sentences**. Found **X unique objects**, **Y unique actions**, **Z functions**, **W discourse markers**.

### Per-type tables

For each non-empty type, show a table. Mark multi-word expressions (words > 1) with **bold**:

#### Objects (`obj`) — things, materials, building elements
| Span | Occurrences | MWE? |
|------|-------------|------|
| **structural load-bearing elements** | 2 | ✓ |
| door | 1 | — |

#### Actions (`act`) — processes, requirements, verbs
#### Functions (`func`) — purposes, roles, goals
#### Discourse (`dis`) — connectives, qualifiers, conditionals

Omit any type that has zero spans.

### Notable patterns
After the tables, add 2–4 brief observations about the entity inventory — e.g.:
- Which type dominates
- Whether MWEs are common (suggests complex technical terminology)
- Any spans that appear across multiple sentences (recurring terms)
- Span types that are absent (may indicate the text is non-regulatory)

---

## Step 5 — Clean up

```bash
rm -f /tmp/spar_input.txt
```

---

## Notes

- SPaR.txt was trained on Scottish Building Regulations; performance is highest on that domain. Results on other regulatory texts are indicative but may miss some spans.
- `func` spans are rare in training data — low recall is expected for this type.
- Multi-word expressions that are discontiguous (e.g. "fire … door") are reconstructed by `parse_spar_output` and appear as a single span string.
- If inference is slow, the BERT encoder is running on CPU — this is normal.

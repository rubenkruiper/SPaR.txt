# SPaR.txt — Technical Context

## Status

The AllenNLP → plain PyTorch rewrite is **complete**. All 9 migration steps are done. The project runs on Python 3.13 with Poetry, no AllenNLP dependency anywhere.

**Achieved test F1: 77.66** (original paper reported 79.93; remaining gap investigated below).

---

## What the Model Does

SPaR.txt performs *shallow parsing* on regulatory text: given a sentence, it labels each token with a tag encoding both the span type (`obj`, `act`, `func`, `dis`) and position (`BH`, `IH`, `BD`, `ID`, or outside `PD`). The key challenge is **discontiguous MWEs** — phrases like "fire-resisting … door" where head and tail tokens are separated by intervening material. `spar_serving_utils.get_spans` reconstructs these from the flat tag sequence using BH/IH/BD/ID transitions.

---

## Architecture

```
BERT (bert-base-cased, 768-dim, frozen)
  → dropout(0.05)
  → biLSTM(input=768, hidden=384, bidirectional → 768-dim output)
  → RelativeGlobalAttention(d_model=768, heads=12)   # Huang et al. 2018
  → concat([biLSTM_out, attn_out], dim=-1)           # → 1536-dim
  → dropout(0.05)
  → FFNN(1536 → 60, ReLU, dropout=0.05)
  → Linear(60 → num_tags)
  → CRF (DiscontiguousTest constraints, no start/end transitions)
```

Key files:
- `spar_lib/models/span_tagger.py` — `SparTagger(nn.Module)`
- `spar_lib/modules/adapted_crf.py` — CRF with inline `_viterbi_decode`
- `spar_lib/modules/multi_head_positional_attention.py` — `RelativeGlobalAttention` (unchanged)
- `spar_lib/readers/tagging_reader.py` — `SparDataset`, `collate_fn`, `_Token` shim, `TAGS`/`TAG_TO_IDX`/`IDX_TO_TAG`
- `spar_lib/metrics/crf_f1_measure.py` — span-level F1 metric

---

## Training

Driven by `run_tagger.py` (plain PyTorch loop):

- **Optimizer**: AdamW, lr=0.005, weight_decay=0.1 (bias/LayerNorm group at 0.05)
- **LR schedule**: Slanted triangular (`_slanted_triangular_lr`, cut_frac=0.1, ratio=32) — starts at lr/32, peaks at lr at 10% of steps, decays back. Implemented as `LambdaLR`.
- **Batch sampler**: `LengthSortedBatchSampler` — groups similar-length sequences, adds small random noise per epoch, shuffles batch order. Equivalent to AllenNLP's `BucketBatchSampler`.
- **Epochs**: up to 50, early stopping patience=20 on val `f1-measure-overall`
- **Checkpoint**: `trained_models/model.pt` + `trained_models/best_metrics.json`
- **Data splits**: 120 train / 40 val / 40 test (200 total annotated sentences)

```bash
poetry run python run_tagger.py                          # train
poetry run python run_tagger.py --evaluate --test data/test/  # evaluate
```

---

## F1 Results and Remaining Gap

| Run | Overall F1 | Notes |
|-----|-----------|-------|
| Original (AllenNLP) | 79.93 | Reported in paper |
| Rewrite v1 | 75.67 | Linear warmup, random-shuffle batches |
| Rewrite v2 | **77.66** | Slanted triangular + length-sorted batches |

Per-type F1 (rewrite v2): obj=74.52, act=66.36, func=63.57, dis=88.09

**Remaining gap (~2.3 points) — likely causes:**
1. **Variance** — 40-sentence test set. Multi-seed runs needed before drawing conclusions. Suggested next step: run 5 seeds, report mean ± std.
2. **Tokenization** — see below. Subword tokenisation differences could account for some of the gap, especially on hyphenated terms.

---

## Tokenization Strategy (pending decision)

**Current**: BERT WordPiece subword tokenization via `BertTokenizerFast` with `return_offsets_mapping=True`. A `_Token` shim dataclass exposes `.idx`/`.idx_end` character offsets so `my_read_utils.py` (unchanged) can align BRAT char-level annotations to token indices.

**Deferred**: Word-boundary tokenization (one token per whitespace-delimited word). Would need changes to the reader and potentially the encoder. The original paper used AllenNLP's `PretrainedTransformerTokenizer` which also produces subword tokens, so this is not a regression — but the decision is still open.

**Sentence splitting**: pysbd (Pragmatic Sentence Boundary Disambiguation). spaCy was preferred but its `thinc → numpy<2.1` constraint is incompatible with Python 3.13 as of April 2026. Revisit when spaCy publishes 3.13-compatible wheels.

---

## Serving

```
TermExtractor (spar_api_utils.py)
  └── N × SparInstance
        └── SparPredictor
              ├── SparTagger (torch.load from trained_models/model.pt)
              └── BertTokenizerFast
```

`TermExtractor.process_texts` dispatches via `ThreadPoolExecutor.map`; each worker uses `current_thread().name` to pick its `SparInstance`. pysbd handles sentence splitting. Auto-trains if `model.pt` is missing.

HTTP API contract (unchanged): `POST /predict_objects/` → `{texts, sentences, predictions}`

---

## What Is Unchanged

- `spar_serving_utils.py` — pure Python output parsing
- `spar_api.py` — FastAPI app
- `spar_lib/modules/multi_head_positional_attention.py` — pure PyTorch
- `spar_lib/readers/reader_utils/my_read_utils.py` — BRAT parsing + tag computation
- BH/IH/BD/ID tag scheme and CRF transition constraints
- `data/` directory (200 BRAT-annotated sentences, ScotReg corpus)

---

## Test Suite

62 tests, all passing (`poetry run pytest`):
- `tests/test_adapted_crf.py` — 12 tests (CRF forward, Viterbi, constraints)
- `tests/test_tagging_reader.py` — 27 tests (BRAT parsing, tokenization, collation)
- `tests/test_crf_f1_measure.py` — 9 tests (F1 accumulation, reset, per-type)
- `tests/test_span_tagger.py` — 14 tests (inference, training, CRF constraints)
- Doctests in `adapted_crf.py`, `tagging_reader.py`, `crf_f1_measure.py`

---

## Suggested Next Steps

1. **Multi-seed evaluation** — run 5 seeds, compute mean ± std F1 to determine true gap vs original
2. **Tokenization decision** — decide whether to stay with subword or switch to word-boundary tokenization
3. **Update CLAUDE.md** — stack description there is now accurate (done alongside this update)

---

## Multi-Sentence Context Window Plan

### Motivation

The sentence-level restriction is adequate for span extraction but prevents BERT from using cross-sentence context when encoding token representations. Coreference ("the appliance" → "it"), conditional chains, and discourse structure spanning multiple sentences are invisible to the model. The tagging scheme and spans themselves stay within-sentence; only BERT's context window expands.

### Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Window size | **3 sentences** | Covers requirement + condition + exception; fits comfortably in 512 tokens; increase later by switching BERT model |
| Stride | **1 sentence** (overlap = 2 sentences) | Each sentence appears in 3 windows; boundary sentences get more context |
| Overlapping predictions | **Character-span voting** — see below | Same token predicted in multiple windows; majority label wins |
| BERT | **Stays frozen** | Matches current setup; unfreezing top 1–2 layers becomes viable once passage-level data is available (flag for future experiment once training on 3k+ passages) |
| Training data format | **JSONL passages** | Simpler than multi-doc BRAT; tokenizer-agnostic char offsets are the ground truth |
| Sentence boundaries | **Character-level** | Stored as `(start_char, end_char)` per sentence in the passage; robust to tokenizer changes |
| Corpus for bootstrapping | **ScotReg + Merged Approved Documents** (+ extensible via offline script) | ScotReg already in repo; Approved Documents at `/Users/rubenk/dev/irec/data/term_extraction_input/`; more corpora added via the same pipeline |

### Overlapping Window Prediction Strategy

When stride < window_size, each sentence appears in multiple context windows, potentially getting different predicted tags. Reconciliation at inference time:

1. Collect all tag sequences predicted for each character position across all windows that covered it.
2. For each token position (identified by char offset), take the **majority vote** label. Ties broken by preferring the label from the centre window (the window where the sentence is in the middle position, not at an edge).
3. Fallback: if no majority, keep the prediction from the window where the target sentence is at position 1 (middle of a 3-sentence window).

This is implemented in the serving layer, not the model itself — the model always sees full passage windows; reconciliation happens post-decode.

### JSONL Format

Each passage sample:
```json
{
  "passage_id": "scotreg_d_0.1.1_i3_w0",
  "doc_id": "d_0.1.1",
  "sentences": [
    {"text": "The door shall be fire-resisting.", "start_char": 0, "end_char": 32},
    {"text": "It shall be self-closing.", "start_char": 33, "end_char": 57},
    {"text": "The ironmongery shall be compatible.", "start_char": 58, "end_char": 94}
  ],
  "passage": "The door shall be fire-resisting. It shall be self-closing. The ironmongery shall be compatible.",
  "token_tags": ["BH-obj", "IH-obj", "BH-act", ...],
  "source": "scotreg"
}
```

`token_tags` is absent for silver-label samples where only span-level predictions are stored, or present as model predictions for training.

### Data Pipeline — Phases

#### Phase 1 — Reconstruct gold passage data from existing BRAT annotations

The filename pattern `d_{doc}_{item}_s_{idx}` in `data/all_annotated/` encodes paragraph membership. Sentences sharing the same `d_{doc}_{item}` prefix are adjacent in the original text. Grouping and concatenating them (with BRAT char-offset shifts) yields passage-level gold data at no annotation cost.

Script: `data/build_passage_data.py`
- Group `all_annotated/` files by `d_{doc}_{item}` prefix
- Concatenate sentence texts with a single space separator; record per-sentence `(start_char, end_char)`
- Shift each sentence's BRAT offsets by its `start_char` within the passage
- Emit JSONL with `token_tags` derived from the merged BRAT annotations

Expected output: ~60–80 gold passage samples (some groups have only 1 sentence).

#### Phase 2 — Build a silver-label corpus from larger regulatory corpora

Script: `data/build_silver_corpus.py`
- Accepts one or more input sources, each as a path to a `.txt` file or a directory of `.txt` files (output from `/read_pdf` or any preprocessing)
- Splits text into sentences via pysbd; groups consecutive sentences into passages of `window_size=3` with stride 1
- Runs `SparPredictor.predict_sentences()` on the concatenated passage string
- Saves per-sentence predictions as JSONL with `"source"` field tagging the corpus
- Designed to be run **offline** (CPU, long-running); writes incrementally so partial runs are resumable

Planned corpora:
1. **ScotReg** — 13,606 sentence files in `data/all_non_annotated_sents/`; already sentence-split. Expected: ~4,500 passages.
2. **Merged Approved Documents** — `/Users/rubenk/dev/irec/data/term_extraction_input/The Merged Approved Documents.pdf`. Extract with pdfplumber, then run the pipeline. Expected: ~20,000+ passages.

Adding further corpora: simply run `build_silver_corpus.py --input <path> --source <name>` on any new `.txt` or directory.

#### Phase 3 — Adapt the reader and model pipeline

1. **`PassageDataset`** (`spar_lib/readers/tagging_reader.py`) — reads JSONL, tokenises the full passage string, recovers token-level tags from `token_tags`, stores `sentence_boundaries` as char offsets. Parallel class to `SentenceDataset` (renamed from current `SparDataset`); existing tests unaffected.

2. **`collate_fn` update** — carry `sentence_boundaries` through batching.

3. **`run_tagger.py`** — add `--data-format {sentence,passage}` flag; default stays `sentence` to preserve current behaviour.

4. **`SparPredictor.predict_sentences()`** (`spar_api_utils.py`) — add `window_size` parameter (default 1 = current behaviour). When `window_size > 1`: assemble passage windows with stride 1, run inference, reconcile overlapping predictions by character-span majority vote, return per-sentence span dicts. The HTTP API contract is unchanged.

#### Phase 4 — Training and comparison

- Train on gold passages alone (Phase 1 data) to establish the passage-level baseline.
- Add silver data incrementally; monitor val F1 to find the useful silver data volume.
- Compare against the current sentence-level 78–79 F1 baseline.
- **BERT unfreezing note**: once passage-level training is stable and data volume > 3,000 passages, run an experiment unfreezing the top 2 BERT layers (layers 10–11). Expect +1–2 F1 but requires a GPU and careful LR scaling (BERT layers at ~10× lower LR than the task head).

### Files To Create

| File | Purpose |
|------|---------|
| `data/build_passage_data.py` | Phase 1: BRAT → JSONL gold passages |
| `data/build_silver_corpus.py` | Phase 2: offline silver-label pipeline for any text corpus |
| `data/passages/` | Output directory for JSONL passage files |

### Files To Modify

| File | Change |
|------|--------|
| `spar_lib/readers/tagging_reader.py` | Add `PassageDataset`; rename `SparDataset` → `SentenceDataset`; update `collate_fn` |
| `run_tagger.py` | Add `--data-format` flag |
| `spar_api_utils.py` | Add `window_size` + overlapping prediction reconciliation to `SparPredictor` |

### What Does NOT Change

- Tagging scheme (BH/IH/BD/ID + PD-pad) — spans remain within-sentence
- Model architecture (`span_tagger.py`) — zero changes
- CRF constraints (`DiscontiguousTest`) — unchanged
- HTTP API contract (`spar_api.py`) — unchanged
- Existing 120/40/40 BRAT split — kept as sentence-level baseline

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

# SPaR.txt — Technical Context

## What the Model Does

SPaR.txt performs *shallow parsing* on regulatory text: given a sentence, it labels each token with a tag that encodes both the span type (`obj`, `act`, `func`, `dis`) and whether the token is a Head-Begin (`BH`), Head-Interior (`IH`), Discontiguous-Begin (`BD`), Discontiguous-Interior (`ID`), or outside (`PD`).

The key challenge is **discontiguous MWEs** — phrases like "fire-resisting … door" where the head and tail tokens are separated by intervening material. `spar_serving_utils.get_spans` reconstructs these from the flat tag sequence using the `BH/IH/BD/ID` transitions.

## Architecture (target — attention variant)

The model being ported is the attention variant (`span_tagger_att.py`):

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

`RelativeGlobalAttention` is in `spar_lib/modules/multi_head_positional_attention.py` — pure PyTorch, no changes needed.

The CRF (`spar_lib/modules/adapted_crf.py`) implements its own forward-backward and Viterbi with custom transition constraints (`allowed_transitions` / `is_transition_allowed`). The constraint logic is pure Python; the numeric routines use only PyTorch.

Output parsing lives outside the model in `spar_serving_utils.parse_spar_output`, which converts flat `words`+`tags` arrays into `{obj: [...], act: [...], func: [...], dis: [...]}`.

## Serving Architecture

`TermExtractor` (in `spar_api_utils.py`) manages a pool of `SparInstance` objects (one per CPU thread). `process_texts` dispatches input texts via `ThreadPoolExecutor.map`, each worker running `process_text` → `split_into_sentences` → `process_sentence_batch`.

## Tokenization Strategy (pending decision)

**Current**: BERT WordPiece subword tokenization via AllenNLP's `PretrainedTransformerTokenizer`. The BRAT char-offset → token-index conversion in `spar_lib/readers/reader_utils/my_read_utils.py` depends on each token exposing `.idx` and `.idx_end` character offsets.

**Future**: Replace with word-boundary tokenization (i.e. one token per whitespace-delimited word, not per subword). This affects both the reader and potentially the encoder choice. **This decision is deferred** — `BertTokenizerFast` is used in the initial rewrite as a like-for-like replacement.

**Sentence splitting**: TextBlob/NLTK replaced with **pysbd** (Pragmatic Sentence Boundary Disambiguation) — pure Python, no C extensions, Python 3.13 compatible. spaCy was the preferred choice but its entire 3.8.x series has a `thinc → numpy<2.1` constraint that is incompatible with Python 3.13. When spaCy publishes a Python 3.13 compatible release it can replace pysbd and also serve the word-boundary tokenization goal.

## What Is Unchanged

- `spar_serving_utils.py` — untouched (pure Python output parsing)
- `spar_api.py` — untouched (FastAPI app)
- `spar_lib/modules/multi_head_positional_attention.py` — untouched (pure PyTorch)
- `spar_lib/readers/reader_utils/my_read_utils.py` — untouched (BRAT parsing + `discontiguous_tags_to_spans`)
- HTTP API contract: `POST /predict_objects/` → `{texts, sentences, predictions}`
- BH/IH/BD/ID tag scheme and CRF transition constraints
- `data/` directory (200 BRAT-annotated sentences, ScotReg corpus)

## Rewrite Plan

Migration from AllenNLP 2.5 / Python 3.8 / requirements.txt to HuggingFace transformers / Python 3.13 / Poetry.

**File disposition:**

| File | Action |
|------|--------|
| `spar_lib/modules/adapted_crf.py` | Rewrite: remove 3 AllenNLP imports; add inline `_viterbi_decode` (~35 lines); logic otherwise unchanged |
| `spar_lib/readers/tagging_reader.py` | Rewrite: `DatasetReader` → `torch.utils.data.Dataset`; `PretrainedTransformerTokenizer` → `BertTokenizerFast` with `return_offsets_mapping=True` |
| `spar_lib/models/span_tagger.py` | Rewrite: plain `nn.Module` using attention-variant architecture; `TimeDistributed` and `CategoricalAccuracy` dropped |
| `spar_lib/metrics/crf_f1_measure.py` | Rewrite: drop `Metric` base class and AllenNLP utils; accept `dict[int, str]` label map |
| `run_tagger.py` | Rewrite: plain PyTorch training loop; AdamW + `get_linear_schedule_with_warmup`; saves `model.pt` |
| `spar_api_utils.py` | Rewrite: `load_archive` → `torch.load`; TextBlob → pysbd sentencizer; threading pool unchanged |
| `spar_lib/models/span_tagger_att.py` | Delete (folded into `span_tagger.py`) |
| `spar_lib/predictors/span_tagger_predictor.py` | Delete (absorbed into `spar_api_utils.py`) |
| `experiments/` JSON configs | Delete (superseded by argparse defaults in `run_tagger.py`) |
| `requirements.txt` | Delete |
| `pyproject.toml` | New — Poetry, Python `^3.13` |
| `Dockerfile` | Update — `python:3.13` base; poetry install; remove NLTK download lines |

**Training hyperparams** (from `experiments/attention_tagger.json`, become argparse defaults):
- `bert-base-cased`, frozen
- biLSTM: hidden=384, bidirectional
- FFNN: 1536→60, ReLU, dropout=0.05
- CRF: `DiscontiguousTest` constraints, no start/end transitions
- AdamW: lr=0.005, weight_decay=0.1
- 50 epochs, patience=20, validation metric: `f1-measure-overall`

**Out of scope for this rewrite:**
- Test suite (to be added separately)
- Word-boundary tokenization strategy (deferred)
- Existing `.tar.gz` checkpoints (not migrated)

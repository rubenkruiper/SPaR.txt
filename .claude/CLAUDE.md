# SPaR.txt — Shallow Parsing for Regulatory Texts

Research code for a sequence tagging model that identifies technical terms (Multi-Word Expressions) in building regulation texts. Published at the EMNLP 2021 NLLP workshop (Kruiper et al., 2021).

## What It Does

SPaR.txt is a biLSTM+CRF sequence tagger that labels tokens in regulatory sentences with four span types:

| Tag | Meaning |
|-----|---------|
| `obj` | Objects (things/entities) |
| `act` | Actions (verbs/processes) |
| `func` | Functions (purposes/roles) |
| `dis` | Discourse (connectives/qualifiers) |

Spans can be contiguous or discontiguous (e.g. "fire … door"). The rewritten model achieves 77.66 F1 on the test set (original AllenNLP baseline: 79.93).

## Stack

- Python 3.13, Poetry
- PyTorch + HuggingFace `transformers` (BERT encoder, `BertTokenizerFast`)
- `pysbd` (sentence splitting)
- FastAPI + uvicorn (HTTP API)

## Environment Setup

```bash
poetry install
```

## Running

**Train:**
```bash
poetry run python run_tagger.py
```

**Evaluate:**
```bash
poetry run python run_tagger.py --evaluate --test data/test/
```

**Docker:**
```bash
docker build -t spar .
docker run --name spar_api -p 8501:8501 spar
```

On first start, the container trains a model (~20 min on CPU). Swagger UI at `http://localhost:8501/docs`.

**Interactive demo:**
```bash
poetry run python serve_spar.py
```

## API

```
POST /predict_objects/          # {texts: str | [str]} → {texts, sentences, predictions}
POST /set_number_of_predictors/ # {num_cpu_threads: int}
```

`predictions` is a list (per input text) of lists (per sentence) of dicts `{obj: [...], act: [...], func: [...], dis: [...]}`.

## Project Structure

```
spar_api.py              # FastAPI app entry point
spar_api_utils.py        # TermExtractor, SparPredictor, SparInstance
spar_serving_utils.py    # Output parsing: Indices, SingleSpan, Sentence, parse_spar_output
serve_spar.py            # Interactive terminal demo
run_tagger.py            # Train / evaluate CLI (plain PyTorch loop)
spar_lib/
  models/                # span_tagger.py — SparTagger(nn.Module)
  modules/               # adapted_crf.py, multi_head_positional_attention.py
  readers/               # tagging_reader.py — SparDataset, collate_fn, tag vocab
  predictors/            # (empty — predictor absorbed into spar_api_utils.py)
  metrics/               # crf_f1_measure.py — SpanBasedF1Measure
  evaluation_script.py   # Standalone token-level eval (requires scikit-learn)
tests/                   # 62 pytest tests (run: poetry run pytest)
data/
  all_annotated/         # 200 BRAT-annotated sentences (.txt + .ann pairs)
  train/ val/ test/      # 120 / 40 / 40 sentence splits
  all_non_annotated_sents/  # Remaining ScotReg sentences for bulk prediction
  ScotReg/               # Scottish Building Regulations corpus (JSON)
trained_models/          # model.pt checkpoint (not in git)
```

## Key Dependencies

| Package | Role |
|---------|------|
| `torch ^2.3` | Model, training loop, CRF |
| `transformers ^4.40` | BERT encoder + tokenizer |
| `pysbd ^0.3` | Sentence splitting |
| `fastapi ^0.111` | HTTP API |
| `uvicorn ^0.30` | ASGI server |
| `pandas ^2.2` | Data utilities |

## Current Development Focus

The AllenNLP rewrite is complete. Current focus: closing the remaining ~2.3 F1 point gap vs the original (77.66 vs 79.93). Next steps are multi-seed evaluation to quantify variance, and a decision on word-boundary vs subword tokenization. See `.claude/context.md` for full technical detail.

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

Spans can be contiguous or discontiguous (e.g. "fire … door" where tokens between the head and tail are not part of the span). The model achieves 79.93 F1 on the test set and identifies 89.84% of defined terms in held-out building regulation documents.

## Stack

- Python 3.8 (conda environment `spar`)
- AllenNLP 2.5 + PyTorch (biLSTM+CRF model)
- FastAPI + uvicorn (HTTP API)
- TextBlob / NLTK PunktSentenceTokenizer (sentence splitting)

## Environment Setup

```bash
conda create -n spar python=3.8
conda activate spar
pip install -r requirements.txt
```

## Running

**Option 1 — Docker (recommended)**

```bash
docker build -t spar .
docker run --name spar_api -p 8501:8501 spar
# Rebuild code only (skip slow pip install):
# docker build --build-arg ONLY_CODE=$(date +%s) -t spar .
```

On first start, the container trains a model (~20 min on CPU). Swagger UI available at `http://localhost:8501/docs`.

**Option 2 — Direct**

```bash
python run_tagger.py                          # train
python run_tagger.py --evaluate -i data/test/ # evaluate
python serve_spar.py                          # interactive terminal
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
run_tagger.py            # Train / evaluate CLI
spar_lib/
  models/                # span_tagger.py, span_tagger_att.py (AllenNLP Model subclasses)
  modules/               # adapted_crf.py, multi_head_positional_attention.py
  readers/               # tagging_reader.py (AllenNLP DatasetReader)
  predictors/            # span_tagger_predictor.py (AllenNLP Predictor)
  metrics/               # crf_f1_measure.py
  evaluation_script.py
experiments/             # AllenNLP config JSON files
data/
  all_annotated/         # 200 BRAT-annotated sentences (.txt + .ann pairs)
  all_non_annotated_sents/  # Remaining ScotReg sentences for bulk prediction
  ScotReg/               # Scottish Building Regulations corpus (JSON)
predictions/             # Output of bulk prediction runs
trained_models/          # Saved model checkpoints (not in git)
```

## Data

- **Annotated corpus**: 200 sentences from Scottish Building Regulations, annotated in BRAT format (`.ann` files alongside `.txt`).
- **ScotReg**: Full domestic and non-domestic Scottish Building Regulations scraped June 2021, stored as JSON in `data/ScotReg/`.

## Key Dependencies and Their Roles

| Package | Role |
|---------|------|
| `allennlp==2.5.0` | Model framework, training loop, archival |
| `allennlp-models==2.5.0` | Base components reused by spar_lib |
| `textblob==0.15.3` | Sentence splitting (wraps NLTK PunktSentenceTokenizer) |
| `fastapi` | HTTP API |
| `uvicorn` | ASGI server |

## Current Development Focus

Updating the semantic chunking approach. The existing stack (AllenNLP 2.5, Python 3.8, old torch, TextBlob) is outdated and unmaintained — the goal is to replace it with modern equivalents while preserving the annotation data and tagging behaviour.

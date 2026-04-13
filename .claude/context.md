# SPaR.txt — Technical Context

## What the Model Does

SPaR.txt performs *shallow parsing* on regulatory text: given a sentence, it labels each token with a BIO-style tag that encodes both the span type (`obj`, `act`, `func`, `dis`) and whether the token is a Head-Begin (`BH`), Head-Interior (`IH`), Discontiguous-Begin (`BD`), Discontiguous-Interior (`ID`), or outside (`O`).

The key challenge handled by the tag scheme is **discontiguous MWEs** — phrases like "fire-resisting … door" where the head tokens and tail tokens are separated by intervening material. `spar_serving_utils.get_spans` reconstructs these from the flat tag sequence using the `BH/IH/BD/ID` transitions.

## Architecture

The model is a standard AllenNLP sequence tagger:

- **Reader**: `spar_lib/readers/tagging_reader.py` — reads BRAT `.ann`/`.txt` pairs into AllenNLP `Instance` objects; tokenises with a WordPiece tokenizer (BERT-based).
- **Model**: `spar_lib/models/span_tagger.py` — BERT encoder → biLSTM → CRF decoder; `span_tagger_att.py` adds multi-head positional attention.
- **Predictor**: `spar_lib/predictors/span_tagger_predictor.py` — wraps the model for inference, returns `{sentence, doc_id, words, tags, mask}`.
- **Metric**: `spar_lib/metrics/crf_f1_measure.py` — span-level F1 that handles discontiguous spans correctly.

Output parsing lives outside AllenNLP in `spar_serving_utils.parse_spar_output`, which converts the flat `words`+`tags` arrays into a `{obj: [...], act: [...], func: [...], dis: [...]}` dict.

## Serving Architecture

`TermExtractor` (in `spar_api_utils.py`) manages a pool of `SparInstance` objects (one per CPU thread). `process_texts` dispatches a list of input texts to the pool via `ThreadPoolExecutor.map`, with each worker calling `process_text` → `split_into_sentences` → `process_sentence_batch`. This avoids the GIL for I/O-bound waiting while AllenNLP holds the GIL during actual inference.

## Why the Stack Is Outdated

| Component | Problem |
|-----------|---------|
| AllenNLP 2.5 | Unmaintained since ~2022; tied to old PyTorch and allennlp-models; complex internal API |
| Python 3.8 | End-of-life October 2024 |
| TextBlob 0.15 | Thin NLTK wrapper, no active development; sentence splitting quality is mediocre |
| `torch` (implicit, old pin) | AllenNLP 2.5 requires a PyTorch version incompatible with modern CUDA/hardware |
| BRAT reader | Custom, tightly coupled to AllenNLP's `Instance`/`Field` abstractions |

## What to Preserve

- The **annotated dataset** (`data/all_annotated/` — 200 `.txt`/`.ann` pairs) is the core research artefact.
- The **tag scheme** (`BH-obj`, `IH-obj`, `BD-obj`, `ID-obj`, `O`, …) encodes the discontiguous MWE structure and should be preserved.
- The **API contract**: `POST /predict_objects/` accepting `{texts: str | [str]}` and returning `{texts, sentences, predictions}` is consumed by the iReC pipeline (`utilities/spar_utils.py` in the related repo).
- `spar_serving_utils.py` — the output parsing logic (`get_spans`, `parse_spar_output`) is correct and largely framework-agnostic; worth keeping or adapting directly.

## Planned Update Direction

Replace AllenNLP with a modern NLP framework (e.g. HuggingFace `transformers` + a lightweight training loop, or spaCy v3 with a custom component) while:

1. Keeping the same BRAT annotation format as training data source.
2. Keeping the same tag scheme so trained weights encode the same span semantics.
3. Replacing TextBlob sentence splitting with `nltk.tokenize.sent_tokenize` or `spacy`'s sentencizer directly.
4. Keeping the FastAPI serving layer and API contract unchanged.
5. Targeting Python 3.10+.

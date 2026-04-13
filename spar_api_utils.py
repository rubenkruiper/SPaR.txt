"""
SPaR.txt serving utilities — AllenNLP-free rewrite.

Replaces:
  - ``load_archive`` / ``AllenNLPPredictor``  →  ``torch.load`` + ``SparTagger``
  - ``TextBlob`` sentence splitting            →  ``pysbd``
"""
from __future__ import annotations

import concurrent.futures
import subprocess
import sys
from pathlib import Path
from threading import current_thread
from typing import Dict, List, Union

import pysbd
import torch

import spar_serving_utils as su
from spar_lib.models.span_tagger import SparTagger
from spar_lib.readers.tagging_reader import (
    IDX_TO_TAG,
    TAG_TO_IDX,
    _tokenize,
    collate_fn,
)

_DEFAULT_MODEL_DIR = Path("trained_models")
_DEFAULT_BERT      = "bert-base-cased"


# ---------------------------------------------------------------------------
# SparPredictor — loads a trained checkpoint and runs inference
# ---------------------------------------------------------------------------

class SparPredictor:
    """
    Wraps a trained :class:`SparTagger` checkpoint for sentence-level inference.

    If ``model_dir/model.pt`` does not exist, training is triggered
    automatically by calling ``run_tagger.py`` as a subprocess.

    Parameters
    ----------
    model_dir :
        Directory that contains (or will receive) ``model.pt``.
    bert_model :
        HuggingFace identifier for the BERT encoder — used when the checkpoint
        does not embed the model name.
    device :
        Torch device for inference.  Defaults to CUDA if available, else CPU.
    """

    def __init__(
        self,
        model_dir: Path = _DEFAULT_MODEL_DIR,
        bert_model: str = _DEFAULT_BERT,
        device: torch.device | None = None,
    ) -> None:
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        model_pt = model_dir / "model.pt"

        if not model_pt.exists():
            print(
                f"No trained model found at {model_pt}. Training now …\n"
                "This will take 20+ minutes on CPU."
            )
            subprocess.run(
                [sys.executable, "run_tagger.py", "--model-dir", str(model_dir)],
                check=True,
            )

        checkpoint  = torch.load(model_pt, map_location=self.device, weights_only=False)
        saved_args  = checkpoint.get("args", {})
        bert_model  = saved_args.get("bert_model", bert_model)

        self.model = SparTagger(
            num_tags=len(TAG_TO_IDX),
            label_map=IDX_TO_TAG,
            bert_model_name=bert_model,
            lstm_hidden_size=384,
            ffnn_hidden_size=60,
            dropout=0.0,        # no dropout at inference time
            freeze_bert=True,
            attention_heads=12,
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        from transformers import BertTokenizerFast
        self.tokenizer = BertTokenizerFast.from_pretrained(bert_model)

    # ------------------------------------------------------------------

    def predict_sentences(self, sentences: List[str]) -> List[dict]:
        """
        Run inference on a list of sentences.

        Returns one dict per sentence with the keys expected by
        :func:`spar_serving_utils.parse_spar_output`:
        ``sentence``, ``doc_id``, ``mask``, ``tags``, ``words``.
        """
        if not sentences:
            return []

        # Build sample dicts (same structure as SparDataset.__getitem__)
        samples = []
        for i, sent in enumerate(sentences):
            sent = sent.lower() if sent.isupper() else sent
            encoding, _ = _tokenize(sent, self.tokenizer, max_length=512)
            samples.append({
                "input_ids":      torch.tensor(encoding["input_ids"],   dtype=torch.long),
                "attention_mask": torch.tensor(encoding["attention_mask"], dtype=torch.bool),
                "tags":     None,
                "words":    self.tokenizer.convert_ids_to_tokens(encoding["input_ids"]),
                "sentence": sent,
                "doc_id":   str(i),
            })

        batch = collate_fn(samples)

        with torch.no_grad():
            out = self.model(
                batch["input_ids"].to(self.device),
                batch["attention_mask"].to(self.device),
                words=batch["words"],
                sentences=batch["sentences"],
                doc_ids=batch["doc_ids"],
            )

        results = []
        for i in range(len(sentences)):
            results.append({
                "sentence": out["sentences"][i],
                "doc_id":   out["doc_ids"][i],
                "mask":     batch["attention_mask"][i].tolist(),   # list[bool]
                "tags":     out["tags"][i],                        # list[str], real tokens only
                "words":    out["words"][i],                       # list[str], real tokens only
            })
        return results

    def parse_output(
        self,
        prediction: dict,
        span_types: List[str] | None = None,
    ) -> tuple:
        """Convenience wrapper around :func:`spar_serving_utils.parse_spar_output`."""
        if span_types is None:
            span_types = ["obj", "act", "func", "dis"]
        return su.parse_spar_output(prediction, span_types)


# ---------------------------------------------------------------------------
# SparInstance — per-thread predictor with a call() interface
# ---------------------------------------------------------------------------

class SparInstance:
    """One SparPredictor instance, used as a worker in the thread pool."""

    def __init__(
        self,
        model_dir: Path = _DEFAULT_MODEL_DIR,
        bert_model: str = _DEFAULT_BERT,
    ) -> None:
        self.sp = SparPredictor(model_dir=model_dir, bert_model=bert_model)

    def call(self, sentences: Union[List[str], str]) -> List[dict]:
        """
        Run inference on one or more sentences.

        Parameters
        ----------
        sentences :
            A single sentence string or a list of sentence strings.

        Returns
        -------
        List of span dicts ``{obj: […], act: […], func: […], dis: […]}``,
        one per input sentence.
        """
        if isinstance(sentences, str):
            sentences = [sentences]
        if not sentences:
            return [{"obj": [], "act": [], "func": [], "dis": []}]

        predictions = self.sp.predict_sentences(sentences)
        results = []
        for pred in predictions:
            spans, _ = su.parse_spar_output(pred, ["obj", "act", "func", "dis"])
            results.append(spans)
        return results


# ---------------------------------------------------------------------------
# TermExtractor — thread-pool orchestrator with sentence splitting
# ---------------------------------------------------------------------------

class TermExtractor:
    """
    Splits texts into sentences, distributes them across a thread pool of
    :class:`SparInstance` workers, and collects span predictions.

    Parameters
    ----------
    split_length :
        Unused length hint (kept for API compatibility).
    max_num_cpu_threads :
        Number of worker threads (and thus SparInstance copies) to create.
    model_dir :
        Directory containing ``model.pt``.
    bert_model :
        HuggingFace BERT identifier forwarded to each :class:`SparInstance`.
    """

    def __init__(
        self,
        split_length: int = 300,
        max_num_cpu_threads: int = 4,
        model_dir: Path = _DEFAULT_MODEL_DIR,
        bert_model: str = _DEFAULT_BERT,
    ) -> None:
        self.split_length       = split_length
        self.max_num_cpu_threads = max_num_cpu_threads
        self._segmenter         = pysbd.Segmenter(language="en", clean=False)

        self.PREDICTORS: List[SparInstance] = [
            SparInstance(model_dir=model_dir, bert_model=bert_model)
            for _ in range(max_num_cpu_threads + 1)
        ]

    # ------------------------------------------------------------------

    def split_into_sentences(self, text: Union[str, List[str]]) -> List[str]:
        """
        Split *text* into sentences using pysbd.

        Also handles semicolon-delimited strings (e.g. Wikidata multi-definitions)
        and multi-line inputs, filtering out very short fragments (≤ 10 chars).
        """
        if isinstance(text, str):
            parts = text.split(";") if ";" in text else [text]
        else:
            parts = text

        sentences: List[str] = []
        for part in parts:
            for line in part.split("\n"):
                line = line.strip()
                if not line:
                    continue
                for sent in self._segmenter.segment(line):
                    if len(sent) > 10:
                        sentences.append(sent)
        return sentences

    def process_sentence_batch(self, sentences: List[str]) -> List[dict]:
        """Run inference on a pre-split list of sentences using the thread-local worker."""
        if not sentences:
            return []
        thread_idx   = int(current_thread().name.rsplit("_", 1)[-1])
        predictor    = self.PREDICTORS[thread_idx]
        return predictor.call(sentences)

    def process_text(self, text: str):
        """
        Split *text* into sentences, then predict spans for each.

        Returns
        -------
        sentences :
            List of sentence strings.
        predictions :
            List of span dicts, one per sentence.
        """
        sentences   = self.split_into_sentences(text)
        predictions = self.process_sentence_batch(sentences)
        return sentences, predictions

    def process_texts(self, texts: List[str]):
        """
        Process a list of texts in parallel across the thread pool.

        Returns
        -------
        sentences :
            List-of-lists; one list of sentence strings per input text.
        predictions :
            List-of-lists; one list of span dicts per input text.
        """
        sentences_out:   List[List[str]]  = []
        predictions_out: List[List[dict]] = []

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_num_cpu_threads
        ) as executor:
            for sent_list, pred_list in executor.map(self.process_text, texts):
                sentences_out.append(sent_list)
                predictions_out.append(pred_list)

        return sentences_out, predictions_out

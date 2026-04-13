"""
Token-level evaluation script for SPaR.txt.

Compares predictions written to a JSONL file against gold BRAT annotations
using scikit-learn's classification_report (per-tag token-level P/R/F1).

This complements the span-level F1 computed during training.  It is a
standalone script and is not called by ``run_tagger.py``.

Requires scikit-learn (``pip install scikit-learn``).
"""
import json
from pathlib import Path
from typing import List

from transformers import BertTokenizerFast

from spar_lib.readers.tagging_reader import _tokenize
from spar_lib.readers.reader_utils.my_read_utils import get_annotations_from_ann_file


class SimpleEvaluator:
    """
    Compare a JSONL predictions file against gold BRAT ``.ann`` annotations.

    Parameters
    ----------
    predictions_fp :
        Path to a JSONL file where each line is a prediction dict with keys
        ``mask``, ``tags``, and ``words``.
    gold_fp :
        Directory containing ``.txt`` / ``.ann`` file pairs.
    bert_model_name :
        HuggingFace identifier for the tokenizer used when the predictions
        were produced (default: ``bert-base-cased``).
    """

    def __init__(
        self,
        predictions_fp: Path,
        gold_fp: Path,
        bert_model_name: str = "bert-base-cased",
    ) -> None:
        self.predictions_input = predictions_fp
        self.gold_input        = Path(gold_fp)
        self.tokenizer         = BertTokenizerFast.from_pretrained(bert_model_name)

    def read_gold(self) -> List[dict]:
        text_files = sorted(self.gold_input.glob("*.txt"))
        ann_files  = {f.stem: f for f in self.gold_input.glob("*.ann")}

        gold_annotations = []
        for text_file in text_files:
            doc_name = text_file.stem
            sentence = text_file.read_text().strip()
            encoding, token_list = _tokenize(sentence, self.tokenizer, max_length=512)
            token_strings = self.tokenizer.convert_ids_to_tokens(encoding["input_ids"])

            if doc_name not in ann_files:
                continue
            tag_list = get_annotations_from_ann_file(
                ann_files[doc_name], sentence, token_list
            )
            gold_annotations.append({
                "sent_id":    doc_name,
                "sentence":   sentence,
                "token_list": token_strings,
                "tag_list":   tag_list,
            })
        return gold_annotations

    def evaluate(self) -> None:
        from sklearn import metrics  # optional dev dependency

        gold_instances = self.read_gold()

        with open(self.predictions_input) as f:
            predictions_list = [json.loads(line) for line in f]

        y_true: List[str] = []
        y_pred: List[str] = []

        for prediction in predictions_list:
            mask       = prediction["mask"]
            tag_list   = prediction["tags"]
            token_list = prediction["words"]

            for gold in gold_instances:
                if gold["token_list"] == token_list:
                    predicted_tags = [t for m, t in zip(mask, tag_list) if m]
                    # Skip CLS and SEP (first and last tokens)
                    y_true += gold["tag_list"][1:-1]
                    y_pred += predicted_tags[1:-1]

        print(metrics.classification_report(y_true, y_pred, digits=4))


if __name__ == "__main__":
    evaluator = SimpleEvaluator(
        Path("predictions/test_predictions.json"),
        Path("data/test/"),
        "bert-base-cased",
    )
    evaluator.evaluate()

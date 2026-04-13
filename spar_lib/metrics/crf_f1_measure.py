"""
Span-based F1 metric for the DiscontiguousTest tag scheme — AllenNLP-free rewrite.

Changes from the original:
  - ``Metric`` base class dropped (plain Python class).
  - ``Vocabulary`` replaced by a plain ``Dict[int, str]`` label map.
  - ``allennlp.nn.util.get_lengths_from_binary_sequence_mask``
      → ``mask.sum(-1).long()``
  - ``allennlp.common.util.is_distributed`` → removed (always False here).
  - ``ConfigurationError`` → ``ValueError``.
  - ``TypedStringSpan`` defined locally as a type alias.
  - ``detach_tensors`` implemented as a static method.

All accumulation logic and _compute_metrics are unchanged.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Callable, Dict, List, Optional, Set, Tuple

import torch

from spar_lib.readers.reader_utils.my_read_utils import discontiguous_tags_to_spans

TypedStringSpan = Tuple[str, Tuple[int, int]]
TAGS_TO_SPANS_FUNCTION_TYPE = Callable[[List[str], Optional[List[str]]], List[TypedStringSpan]]


class SpanBasedF1Measure:
    """
    Span-level precision / recall / F1 for the DiscontiguousTest tag scheme.

    Accumulates TP / FP / FN counts across :meth:`__call__` invocations and
    reports per-type and overall metrics via :meth:`get_metric`.

    Parameters
    ----------
    label_map :
        ``{int_id: tag_string}`` mapping — the reverse of ``TAG_TO_IDX``.
        Used to convert integer predictions back to tag strings for span
        extraction.
    ignore_classes :
        Span types to exclude from metric computation.
    """

    def __init__(
        self,
        label_map: Dict[int, str],
        ignore_classes: Optional[List[str]] = None,
    ) -> None:
        self._label_map = label_map
        self._ignore_classes: List[str] = ignore_classes or []
        self._true_positives: Dict[str, int] = defaultdict(int)
        self._false_positives: Dict[str, int] = defaultdict(int)
        self._false_negatives: Dict[str, int] = defaultdict(int)

    # ------------------------------------------------------------------
    # Accumulation
    # ------------------------------------------------------------------

    def __call__(
        self,
        predictions: torch.Tensor,
        gold_labels: torch.Tensor,
        mask: Optional[torch.BoolTensor] = None,
    ) -> None:
        """
        Accumulate TP/FP/FN for one batch.

        Parameters
        ----------
        predictions :
            ``(batch_size, sequence_length, num_classes)`` — logits or
            one-hot; only the argmax is used.
        gold_labels :
            ``(batch_size, sequence_length)`` — integer label ids.
        mask :
            ``(batch_size, sequence_length)`` boolean mask; ``True`` for
            real (non-padding) tokens.
        """
        if mask is None:
            mask = torch.ones_like(gold_labels, dtype=torch.bool)

        predictions, gold_labels, mask = self._detach_tensors(predictions, gold_labels, mask)

        sequence_lengths = mask.sum(-1).long()
        argmax_predictions = predictions.max(-1)[1].float()

        batch_size = gold_labels.size(0)
        for i in range(batch_size):
            length = sequence_lengths[i].item()
            if length == 0:
                continue

            predicted_tags = [
                self._label_map[int(label_id)]
                for label_id in argmax_predictions[i, :length].tolist()
            ]
            gold_tags = [
                self._label_map[int(label_id)]
                for label_id in gold_labels[i, :length].tolist()
            ]

            predicted_spans = discontiguous_tags_to_spans(predicted_tags, self._ignore_classes)
            gold_spans      = discontiguous_tags_to_spans(gold_tags,      self._ignore_classes)

            predicted_spans = self._handle_continued_spans(predicted_spans)
            gold_spans      = self._handle_continued_spans(gold_spans)

            for span in predicted_spans:
                if span in gold_spans:
                    self._true_positives[span[0]] += 1
                    gold_spans.remove(span)
                else:
                    self._false_positives[span[0]] += 1
            for span in gold_spans:
                self._false_negatives[span[0]] += 1

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def get_metric(self, reset: bool = False) -> Dict[str, float]:
        """
        Return per-type and overall precision / recall / F1.

        Keys follow the pattern ``"precision-<type>"``, ``"recall-<type>"``,
        ``"f1-measure-<type>"``, plus ``"*-overall"``.
        """
        all_tags: Set[str] = set()
        all_tags.update(self._true_positives)
        all_tags.update(self._false_positives)
        all_tags.update(self._false_negatives)

        metrics: Dict[str, float] = {}
        for tag in all_tags:
            p, r, f1 = self._compute_metrics(
                self._true_positives[tag],
                self._false_positives[tag],
                self._false_negatives[tag],
            )
            metrics[f"precision-{tag}"]   = p
            metrics[f"recall-{tag}"]      = r
            metrics[f"f1-measure-{tag}"]  = f1

        p, r, f1 = self._compute_metrics(
            sum(self._true_positives.values()),
            sum(self._false_positives.values()),
            sum(self._false_negatives.values()),
        )
        metrics["precision-overall"]  = p
        metrics["recall-overall"]     = r
        metrics["f1-measure-overall"] = f1

        if reset:
            self.reset()
        return metrics

    def reset(self) -> None:
        self._true_positives  = defaultdict(int)
        self._false_positives = defaultdict(int)
        self._false_negatives = defaultdict(int)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _detach_tensors(*tensors):
        return tuple(
            t.detach().cpu() if isinstance(t, torch.Tensor) else t
            for t in tensors
        )

    @staticmethod
    def _handle_continued_spans(spans: List[TypedStringSpan]) -> List[TypedStringSpan]:
        span_set: Set[TypedStringSpan] = set(spans)
        continued_labels = [label[2:] for (label, _) in span_set if label.startswith("C-")]
        for label in continued_labels:
            continued = {s for s in span_set if label in s[0]}
            start = min(s[1][0] for s in continued)
            end   = max(s[1][1] for s in continued)
            span_set.difference_update(continued)
            span_set.add((label, (start, end)))
        return list(span_set)

    @staticmethod
    def _compute_metrics(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
        precision = tp / (tp + fp + 1e-13)
        recall    = tp / (tp + fn + 1e-13)
        f1        = 2.0 * precision * recall / (precision + recall + 1e-13)
        return precision, recall, f1

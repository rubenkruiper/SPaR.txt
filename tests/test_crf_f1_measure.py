"""
Unit tests for spar_lib/metrics/crf_f1_measure.py.
"""
import pytest
import torch

from spar_lib.metrics.crf_f1_measure import SpanBasedF1Measure
from spar_lib.readers.tagging_reader import IDX_TO_TAG, TAG_TO_IDX


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _one_hot(tag_ids: list, num_tags: int) -> torch.Tensor:
    """Build a (1, seq_len, num_tags) one-hot prediction tensor."""
    t = torch.zeros(1, len(tag_ids), num_tags)
    for j, tid in enumerate(tag_ids):
        t[0, j, tid] = 1.0
    return t


def _gold(tag_ids: list) -> torch.Tensor:
    return torch.tensor([tag_ids], dtype=torch.long)


def _mask(length: int) -> torch.BoolTensor:
    return torch.ones(1, length, dtype=torch.bool)


NUM_TAGS = len(TAG_TO_IDX)
PD  = TAG_TO_IDX["PD-pad"]
BHO = TAG_TO_IDX["BH-obj"]
IHO = TAG_TO_IDX["IH-obj"]
BHA = TAG_TO_IDX["BH-act"]


# ---------------------------------------------------------------------------
# SpanBasedF1Measure
# ---------------------------------------------------------------------------

class TestSpanBasedF1Measure:

    def test_perfect_prediction_gives_f1_one(self):
        metric = SpanBasedF1Measure(IDX_TO_TAG)
        tags = [PD, BHO, IHO, PD]
        metric(_one_hot(tags, NUM_TAGS), _gold(tags), _mask(4))
        result = metric.get_metric()
        assert result["f1-measure-overall"] == pytest.approx(1.0, abs=1e-6)

    def test_no_predictions_gives_f1_zero(self):
        metric = SpanBasedF1Measure(IDX_TO_TAG)
        pred = [PD, PD, PD, PD]
        gold = [PD, BHO, IHO, PD]
        metric(_one_hot(pred, NUM_TAGS), _gold(gold), _mask(4))
        result = metric.get_metric()
        assert result["f1-measure-overall"] == pytest.approx(0.0, abs=1e-6)

    def test_extra_predictions_reduces_precision(self):
        metric = SpanBasedF1Measure(IDX_TO_TAG)
        gold = [PD, BHO, PD, PD]
        pred = [PD, BHO, BHA, PD]   # one correct + one spurious
        metric(_one_hot(pred, NUM_TAGS), _gold(gold), _mask(4))
        result = metric.get_metric()
        # 1 TP, 1 FP → precision = 0.5; 1 TP, 0 FN → recall = 1.0
        assert result["precision-overall"] == pytest.approx(0.5, abs=1e-4)
        assert result["recall-overall"]    == pytest.approx(1.0, abs=1e-4)

    def test_missing_span_reduces_recall(self):
        metric = SpanBasedF1Measure(IDX_TO_TAG)
        gold = [PD, BHO, PD, BHA]
        pred = [PD, BHO, PD, PD]    # misses BH-act
        metric(_one_hot(pred, NUM_TAGS), _gold(gold), _mask(4))
        result = metric.get_metric()
        assert result["recall-overall"] == pytest.approx(0.5, abs=1e-4)

    def test_accumulates_across_batches(self):
        metric = SpanBasedF1Measure(IDX_TO_TAG)
        tags = [PD, BHO, PD]
        metric(_one_hot(tags, NUM_TAGS), _gold(tags), _mask(3))
        metric(_one_hot(tags, NUM_TAGS), _gold(tags), _mask(3))
        result = metric.get_metric()
        assert result["f1-measure-overall"] == pytest.approx(1.0, abs=1e-6)

    def test_reset_clears_counters(self):
        metric = SpanBasedF1Measure(IDX_TO_TAG)
        tags = [PD, BHO, PD]
        metric(_one_hot(tags, NUM_TAGS), _gold(tags), _mask(3))
        metric.get_metric(reset=True)
        # After reset, calling get_metric again should give near-zero (no data)
        result = metric.get_metric()
        assert result["f1-measure-overall"] == pytest.approx(0.0, abs=1e-6)

    def test_per_type_metrics_reported(self):
        metric = SpanBasedF1Measure(IDX_TO_TAG)
        tags = [PD, BHO, PD, BHA]
        metric(_one_hot(tags, NUM_TAGS), _gold(tags), _mask(4))
        result = metric.get_metric()
        assert "f1-measure-obj" in result
        assert "f1-measure-act" in result

    def test_masked_tokens_ignored(self):
        """Padding tokens (mask=False) should not affect span extraction."""
        metric = SpanBasedF1Measure(IDX_TO_TAG)
        tags = [PD, BHO, PD, PD]
        # Mask covers only the first 2 tokens; positions 2 & 3 are padding
        mask = torch.tensor([[True, True, False, False]])
        metric(_one_hot(tags, NUM_TAGS), _gold(tags), mask)
        # "BH-obj" at position 1 is within the mask → TP
        result = metric.get_metric()
        assert result["f1-measure-overall"] == pytest.approx(1.0, abs=1e-6)

    def test_batch_of_two(self):
        metric = SpanBasedF1Measure(IDX_TO_TAG)
        tags_a = [PD, BHO, PD]
        tags_b = [PD, BHA, PD]
        preds = torch.zeros(2, 3, NUM_TAGS)
        for j, tid in enumerate(tags_a):
            preds[0, j, tid] = 1.0
        for j, tid in enumerate(tags_b):
            preds[1, j, tid] = 1.0
        gold  = torch.tensor([tags_a, tags_b])
        mask  = torch.ones(2, 3, dtype=torch.bool)
        metric(preds, gold, mask)
        result = metric.get_metric()
        assert result["f1-measure-overall"] == pytest.approx(1.0, abs=1e-6)

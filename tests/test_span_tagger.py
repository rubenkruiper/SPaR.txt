"""
Unit tests for spar_lib/models/span_tagger.py.

BertModel.from_pretrained is patched throughout so tests run without
downloading weights.  The mock returns a fixed last_hidden_state tensor so
the rest of the pipeline (LSTM → attention → FFNN → CRF) can be exercised
with a real, randomly-initialised model.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch

from spar_lib.models.span_tagger import SparTagger
from spar_lib.readers.tagging_reader import IDX_TO_TAG, TAG_TO_IDX, TAGS

NUM_TAGS = len(TAG_TO_IDX)
BERT_DIM = 768


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _make_mock_bert(bert_dim: int = BERT_DIM) -> MagicMock:
    """
    Return a mock BertModel whose forward() returns a fixed last_hidden_state.
    The hidden_size is exposed via .config.hidden_size so SparTagger can read it.
    """
    bert = MagicMock()
    bert.config.hidden_size = bert_dim
    bert.parameters.return_value = iter([])   # freeze_bert iterates params

    def forward_fn(input_ids, attention_mask, **kwargs):
        B, L = input_ids.shape
        out = MagicMock()
        out.last_hidden_state = torch.zeros(B, L, bert_dim)
        return out

    bert.side_effect = forward_fn
    return bert


@pytest.fixture
def tagger():
    """SparTagger with a mocked BERT encoder, tiny LSTM and FFNN."""
    mock_bert = _make_mock_bert()
    with patch("spar_lib.models.span_tagger.BertModel") as MockBertClass:
        MockBertClass.from_pretrained.return_value = mock_bert
        model = SparTagger(
            num_tags=NUM_TAGS,
            label_map=IDX_TO_TAG,
            bert_model_name="bert-base-cased",   # not actually loaded
            lstm_hidden_size=8,                   # tiny for test speed
            ffnn_hidden_size=4,
            dropout=0.0,
            freeze_bert=True,
            attention_heads=4,
        )
    return model


def _batch(batch_size: int = 2, seq_len: int = 6):
    """Return (input_ids, attention_mask) for a dummy batch."""
    ids  = torch.ones(batch_size, seq_len, dtype=torch.long)
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    return ids, mask


# ---------------------------------------------------------------------------
# Forward pass — inference (no gold_tags)
# ---------------------------------------------------------------------------

class TestSparTaggerInference:

    def test_forward_returns_required_keys(self, tagger):
        ids, mask = _batch()
        out = tagger(ids, mask)
        assert {"logits", "mask", "tags"} <= out.keys()

    def test_logits_shape(self, tagger):
        B, L = 2, 6
        ids, mask = _batch(B, L)
        out = tagger(ids, mask)
        assert out["logits"].shape == (B, L, NUM_TAGS)

    def test_tags_are_string_lists(self, tagger):
        ids, mask = _batch()
        out = tagger(ids, mask)
        for item_tags in out["tags"]:
            assert all(isinstance(t, str) for t in item_tags)
            assert all(t in TAG_TO_IDX for t in item_tags)

    def test_tags_length_matches_real_tokens(self, tagger):
        """Each decoded path length must equal the number of True tokens."""
        ids = torch.ones(2, 6, dtype=torch.long)
        # Second sample has 4 real tokens (last 2 padded)
        mask = torch.tensor([
            [True, True, True, True, True, True],
            [True, True, True, True, False, False],
        ])
        out = tagger(ids, mask)
        assert len(out["tags"][0]) == 6
        assert len(out["tags"][1]) == 4

    def test_no_loss_without_gold_tags(self, tagger):
        ids, mask = _batch()
        out = tagger(ids, mask)
        assert "loss" not in out

    def test_metadata_forwarded(self, tagger):
        ids, mask = _batch(2, 4)
        out = tagger(
            ids, mask,
            words=[["a", "b", "c", "d"], ["e", "f", "g", "h"]],
            sentences=["sent one", "sent two"],
            doc_ids=["doc_0", "doc_1"],
        )
        assert out["words"]     == [["a", "b", "c", "d"], ["e", "f", "g", "h"]]
        assert out["sentences"] == ["sent one", "sent two"]
        assert out["doc_ids"]   == ["doc_0", "doc_1"]

    def test_metadata_absent_when_not_provided(self, tagger):
        ids, mask = _batch()
        out = tagger(ids, mask)
        assert "words"     not in out
        assert "sentences" not in out
        assert "doc_ids"   not in out


# ---------------------------------------------------------------------------
# Forward pass — training (with gold_tags)
# ---------------------------------------------------------------------------

class TestSparTaggerTraining:

    def test_loss_present_with_gold_tags(self, tagger):
        ids, mask = _batch()
        gold = torch.zeros(2, 6, dtype=torch.long)   # all PD-pad
        out = tagger(ids, mask, gold_tags=gold)
        assert "loss" in out

    def test_loss_is_scalar(self, tagger):
        ids, mask = _batch()
        gold = torch.zeros(2, 6, dtype=torch.long)
        out = tagger(ids, mask, gold_tags=gold)
        assert out["loss"].shape == torch.Size([])

    def test_loss_is_finite(self, tagger):
        ids, mask = _batch()
        gold = torch.zeros(2, 6, dtype=torch.long)
        out = tagger(ids, mask, gold_tags=gold)
        assert out["loss"].isfinite()

    def test_loss_is_differentiable(self, tagger):
        """Gradients should flow into the non-frozen parameters (LSTM, FFNN, CRF)."""
        ids, mask = _batch()
        gold = torch.zeros(2, 6, dtype=torch.long)
        out = tagger(ids, mask, gold_tags=gold)
        out["loss"].backward()
        # At least one non-BERT parameter should have a gradient
        trainable = [p for p in tagger.parameters() if p.requires_grad]
        assert any(p.grad is not None for p in trainable)

    def test_get_metrics_returns_f1_keys(self, tagger):
        ids, mask = _batch()
        gold = torch.zeros(2, 6, dtype=torch.long)
        tagger(ids, mask, gold_tags=gold)
        metrics = tagger.get_metrics()
        assert "f1-measure-overall" in metrics
        assert "precision-overall"  in metrics
        assert "recall-overall"     in metrics

    def test_get_metrics_reset_clears_state(self, tagger):
        ids, mask = _batch()
        gold = torch.zeros(2, 6, dtype=torch.long)
        tagger(ids, mask, gold_tags=gold)
        tagger.get_metrics(reset=True)
        # No data accumulated → overall F1 near 0
        metrics = tagger.get_metrics()
        assert metrics["f1-measure-overall"] == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# CRF constraints propagated through the model
# ---------------------------------------------------------------------------

class TestSparTaggerConstraints:

    def test_decoded_tags_satisfy_transition_constraints(self, tagger):
        """Every consecutive pair in each decoded path must be allowed."""
        from spar_lib.modules.adapted_crf import allowed_transitions
        valid_pairs = set(allowed_transitions("DiscontiguousTest", IDX_TO_TAG))

        ids, mask = _batch(batch_size=4, seq_len=8)
        out = tagger(ids, mask)

        for item_tags in out["tags"]:
            int_tags = [TAG_TO_IDX[t] for t in item_tags]
            for a, b in zip(int_tags[:-1], int_tags[1:]):
                assert (a, b) in valid_pairs, (
                    f"Forbidden transition {IDX_TO_TAG[a]} → {IDX_TO_TAG[b]}"
                )

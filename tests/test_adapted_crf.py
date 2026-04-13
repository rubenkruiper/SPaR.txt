"""
Unit tests for spar_lib/modules/adapted_crf.py.

Doctests cover the pure transition-rule functions (is_transition_allowed,
allowed_transitions).  These unit tests cover the numeric / stateful parts:
_viterbi_decode and ConditionalRandomField.
"""
import pytest
import torch

from spar_lib.modules.adapted_crf import (
    ConditionalRandomField,
    _viterbi_decode,
    allowed_transitions,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Minimal tag vocabulary matching the DiscontiguousTest scheme.
# Keys are integer ids; START/END sentinels are added by allowed_transitions.
TAGS = {0: "PD-pad", 1: "BH-obj", 2: "IH-obj", 3: "BH-act", 4: "IH-act"}
NUM_TAGS = len(TAGS)


def make_crf(constrained: bool = True) -> ConditionalRandomField:
    constraints = allowed_transitions("DiscontiguousTest", TAGS) if constrained else None
    return ConditionalRandomField(
        num_tags=NUM_TAGS,
        constraints=constraints,
        include_start_end_transitions=False,
    )


def _tag_seq(seq_len: int, num_tags: int) -> torch.Tensor:
    """
    Build a sentinel-padded emission tensor as viterbi_tags does internally.

    Shape: (seq_len + 2, num_tags + 2)
    Position 0  : START sentinel (only start_tag column has score 0).
    Position 1..seq_len : real emission (all -10 000, caller fills in).
    Position -1 : END sentinel (only end_tag column has score 0).
    """
    start_tag = num_tags
    end_tag = num_tags + 1
    total_tags = num_tags + 2
    ts = torch.full((seq_len + 2, total_tags), -10000.0)
    ts[0, start_tag] = 0.0
    ts[seq_len + 1, end_tag] = 0.0
    return ts


# ---------------------------------------------------------------------------
# _viterbi_decode
# ---------------------------------------------------------------------------

class TestViterbiDecode:

    def test_selects_highest_emission(self):
        """With uniform transitions, the tag with the highest emission wins."""
        # 2 real tags + START/END = 4 total; sequence length 1
        ts = _tag_seq(seq_len=1, num_tags=2)
        ts[1, 0] = 5.0   # tag 0: high emission
        ts[1, 1] = 1.0   # tag 1: low emission
        trans = torch.zeros(4, 4)

        paths, scores = _viterbi_decode(ts, trans, top_k=1)

        # Path includes START (=2) and END (=3) sentinels
        assert paths[0] == [2, 0, 3]

    def test_forbidden_transition_never_chosen(self):
        """A tag reachable only via forbidden transitions should never appear.

        Setup: START(2)→tag1 is blocked AND tag0→tag1 is blocked, so tag1
        can only be reached via tag1→tag1.  Since tag1 is unreachable at
        step 1, it must be absent from the entire path despite having the
        highest emission at step 2.
        """
        # tag indices: 0, 1; START=2, END=3
        ts = _tag_seq(seq_len=2, num_tags=2)
        ts[1, 0] = 5.0   # step 1: tag 0 has higher emission
        ts[1, 1] = 0.0
        ts[2, 0] = 0.0
        ts[2, 1] = 10.0  # step 2: tag 1 has highest emission …
        trans = torch.zeros(4, 4)
        trans[2, 1] = -10000.0  # … but START→tag1 is forbidden
        trans[0, 1] = -10000.0  # … and tag0→tag1 is forbidden

        paths, _ = _viterbi_decode(ts, trans, top_k=1)

        # tag1 must not appear at any position
        assert 1 not in paths[0]

    def test_longer_sequence_length(self):
        """Output path length equals sequence length + 2 (sentinels)."""
        seq_len = 6
        num_tags = 3
        ts = _tag_seq(seq_len=seq_len, num_tags=num_tags)
        for t in range(1, seq_len + 1):
            ts[t, torch.randint(num_tags, (1,)).item()] = 1.0
        trans = torch.zeros(num_tags + 2, num_tags + 2)

        paths, scores = _viterbi_decode(ts, trans, top_k=1)

        assert len(paths) == 1
        assert len(paths[0]) == seq_len + 2

    def test_top_k_greater_than_one_raises(self):
        ts = _tag_seq(seq_len=1, num_tags=2)
        trans = torch.zeros(4, 4)
        with pytest.raises(NotImplementedError):
            _viterbi_decode(ts, trans, top_k=2)


# ---------------------------------------------------------------------------
# ConditionalRandomField
# ---------------------------------------------------------------------------

class TestConditionalRandomField:

    def test_forward_returns_scalar(self):
        """forward() should return a scalar log-likelihood."""
        crf = make_crf()
        logits = torch.randn(2, 4, NUM_TAGS)   # batch=2, seq=4
        # All-PD-pad is a valid tag sequence under DiscontiguousTest
        tags = torch.zeros(2, 4, dtype=torch.long)
        mask = torch.ones(2, 4, dtype=torch.bool)

        ll = crf(logits, tags, mask)

        assert ll.shape == torch.Size([])

    def test_forward_is_differentiable(self):
        """Gradients should flow back through the CRF loss."""
        crf = make_crf()
        logits = torch.randn(1, 3, NUM_TAGS, requires_grad=True)
        tags = torch.zeros(1, 3, dtype=torch.long)
        mask = torch.ones(1, 3, dtype=torch.bool)

        loss = -crf(logits, tags, mask)
        loss.backward()

        assert logits.grad is not None
        assert not torch.isnan(logits.grad).any()

    def test_viterbi_tags_output_shape(self):
        """viterbi_tags should return one path per batch item."""
        crf = make_crf()
        batch_size, seq_len = 3, 5
        logits = torch.randn(batch_size, seq_len, NUM_TAGS)
        mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

        paths = crf.viterbi_tags(logits, mask, top_k=1)

        assert len(paths) == batch_size
        for item in paths:
            assert len(item) == 1            # top_k=1
            path, score = item[0]
            assert len(path) == seq_len

    def test_viterbi_tags_respects_constraints(self):
        """No transition in the decoded path should violate DiscontiguousTest."""
        crf = make_crf(constrained=True)
        logits = torch.randn(4, 6, NUM_TAGS)
        mask = torch.ones(4, 6, dtype=torch.bool)

        paths = crf.viterbi_tags(logits, mask, top_k=1)

        valid_pairs = set(allowed_transitions("DiscontiguousTest", TAGS))
        for item in paths:
            path, _ = item[0]
            for prev_tag, next_tag in zip(path[:-1], path[1:]):
                assert (prev_tag, next_tag) in valid_pairs, (
                    f"Forbidden transition {prev_tag} → {next_tag} in path {path}"
                )

    def test_viterbi_tags_flat_output_when_top_k_none(self):
        """Calling viterbi_tags(top_k=None) returns a flat list of (path, score)."""
        crf = make_crf()
        logits = torch.randn(2, 3, NUM_TAGS)
        mask = torch.ones(2, 3, dtype=torch.bool)

        result = crf.viterbi_tags(logits, mask, top_k=None)

        # Should be [(path, score), (path, score)] not [[(path,score)], ...]
        assert len(result) == 2
        path, score = result[0]
        assert len(path) == 3

    def test_unconstrained_crf_produces_valid_paths(self):
        """Even without constraints, paths should have the right length."""
        crf = make_crf(constrained=False)
        logits = torch.randn(2, 4, NUM_TAGS)
        mask = torch.ones(2, 4, dtype=torch.bool)

        paths = crf.viterbi_tags(logits, mask, top_k=1)

        for item in paths:
            path, _ = item[0]
            assert len(path) == 4

    def test_mask_shortens_effective_sequence(self):
        """Tokens where mask=False should not influence the decoded path length."""
        crf = make_crf()
        logits = torch.randn(1, 6, NUM_TAGS)
        # Only first 3 tokens are valid
        mask = torch.tensor([[True, True, True, False, False, False]])

        paths = crf.viterbi_tags(logits, mask, top_k=1)

        path, _ = paths[0][0]
        assert len(path) == 3

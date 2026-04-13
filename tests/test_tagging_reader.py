"""
Unit tests for spar_lib/readers/tagging_reader.py.

BertTokenizerFast is mocked throughout so tests run without downloading
model weights.  The mock splits on whitespace and produces predictable
char-offset and input_id values, sufficient to exercise the BRAT→tensor
pipeline end-to-end.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

from spar_lib.readers.tagging_reader import (
    IDX_TO_TAG,
    TAGS,
    TAG_TO_IDX,
    SparDataset,
    _Token,
    _tokenize,
    collate_fn,
)


# ---------------------------------------------------------------------------
# Mock tokenizer fixture
# ---------------------------------------------------------------------------

def _word_char_offsets(text: str):
    """Return (start, end) char offsets for each whitespace-delimited token."""
    offsets = []
    pos = 0
    for word in text.split():
        start = text.index(word, pos)
        end = start + len(word)
        offsets.append((start, end))
        pos = end
    return offsets


def _make_mock_tokenizer(text: str) -> MagicMock:
    """
    Return a MagicMock that behaves like BertTokenizerFast for *text*.

    Token ids: [CLS]=101, words get ids 1000, 1001, …, [SEP]=102.
    Char offsets match the actual whitespace positions in *text*.
    """
    words = text.split()
    offsets = _word_char_offsets(text)
    n = len(words)

    encoding = {
        "input_ids":           [101] + list(range(1000, 1000 + n)) + [102],
        "attention_mask":      [1] * (n + 2),
        "offset_mapping":      [(0, 0)] + offsets + [(0, 0)],
        "special_tokens_mask": [1] + [0] * n + [1],
    }

    tok = MagicMock()
    tok.return_value = encoding
    tok.convert_ids_to_tokens.side_effect = lambda ids: [
        "[CLS]" if i == 101 else ("[SEP]" if i == 102 else f"w{i - 1000}")
        for i in ids
    ]
    return tok


# ---------------------------------------------------------------------------
# _Token shim
# ---------------------------------------------------------------------------

class TestToken:

    def test_real_token_has_offsets(self):
        t = _Token(text="door", idx=4, idx_end=8)
        assert t.idx == 4
        assert t.idx_end == 8

    def test_special_token_has_none_offsets(self):
        cls_token = _Token(text="[CLS]", idx=None, idx_end=None)
        assert cls_token.idx is None
        assert cls_token.idx_end is None


# ---------------------------------------------------------------------------
# _tokenize helper
# ---------------------------------------------------------------------------

class TestTokenize:

    def test_returns_one_token_per_input_id(self):
        sentence = "The door shall resist fire."
        tok = _make_mock_tokenizer(sentence)
        encoding, token_list = _tokenize(sentence, tok, max_length=512)
        assert len(token_list) == len(encoding["input_ids"])

    def test_special_tokens_have_none_offsets(self):
        sentence = "hello world"
        tok = _make_mock_tokenizer(sentence)
        _, token_list = _tokenize(sentence, tok, max_length=512)
        # [CLS] at index 0, [SEP] at index -1
        assert token_list[0].idx is None
        assert token_list[-1].idx is None

    def test_real_tokens_have_char_offsets(self):
        sentence = "hello world"
        tok = _make_mock_tokenizer(sentence)
        _, token_list = _tokenize(sentence, tok, max_length=512)
        # "hello" starts at 0, "world" starts at 6
        assert token_list[1].idx == 0
        assert token_list[1].idx_end == 5
        assert token_list[2].idx == 6
        assert token_list[2].idx_end == 11


# ---------------------------------------------------------------------------
# SparDataset
# ---------------------------------------------------------------------------

SENTENCE = "The door shall resist fire."
# "door" spans chars 4–8 (Object_span); "shall" spans chars 9–14 (Action_span)
ANN_CONTENT = "T1\tObject_span 4 8\tdoor\nT2\tAction_span 9 14\tshall\n"


@pytest.fixture
def txt_only_dir(tmp_path: Path) -> Path:
    (tmp_path / "sent_0.txt").write_text(SENTENCE)
    (tmp_path / "sent_1.txt").write_text("Another sentence here.")
    return tmp_path


@pytest.fixture
def annotated_dir(tmp_path: Path) -> Path:
    (tmp_path / "sent_0.txt").write_text(SENTENCE)
    (tmp_path / "sent_0.ann").write_text(ANN_CONTENT)
    return tmp_path


class TestSparDataset:

    def test_len_equals_txt_file_count(self, txt_only_dir):
        tok = _make_mock_tokenizer(SENTENCE)
        ds = SparDataset(str(txt_only_dir), tok)
        assert len(ds) == 2

    def test_sample_has_required_keys(self, txt_only_dir):
        tok = _make_mock_tokenizer(SENTENCE)
        ds = SparDataset(str(txt_only_dir), tok)
        sample = ds[0]
        assert {"input_ids", "attention_mask", "tags", "words", "sentence", "doc_id"} <= sample.keys()

    def test_tags_none_without_ann_file(self, txt_only_dir):
        tok = _make_mock_tokenizer(SENTENCE)
        ds = SparDataset(str(txt_only_dir), tok)
        assert ds[0]["tags"] is None

    def test_input_ids_are_long_tensor(self, txt_only_dir):
        tok = _make_mock_tokenizer(SENTENCE)
        ds = SparDataset(str(txt_only_dir), tok)
        assert ds[0]["input_ids"].dtype == torch.long

    def test_attention_mask_is_bool_tensor(self, txt_only_dir):
        tok = _make_mock_tokenizer(SENTENCE)
        ds = SparDataset(str(txt_only_dir), tok)
        assert ds[0]["attention_mask"].dtype == torch.bool

    def test_attention_mask_all_true_for_short_sentence(self, txt_only_dir):
        tok = _make_mock_tokenizer(SENTENCE)
        ds = SparDataset(str(txt_only_dir), tok)
        assert ds[0]["attention_mask"].all()

    def test_doc_id_matches_filename_stem(self, txt_only_dir):
        tok = _make_mock_tokenizer(SENTENCE)
        ds = SparDataset(str(txt_only_dir), tok)
        doc_ids = {ds[i]["doc_id"] for i in range(len(ds))}
        assert "sent_0" in doc_ids
        assert "sent_1" in doc_ids

    def test_annotation_produces_tag_tensor(self, annotated_dir):
        tok = _make_mock_tokenizer(SENTENCE)
        ds = SparDataset(str(annotated_dir), tok)
        sample = ds[0]
        assert sample["tags"] is not None
        assert sample["tags"].dtype == torch.long

    def test_annotation_tag_tensor_length_matches_tokens(self, annotated_dir):
        tok = _make_mock_tokenizer(SENTENCE)
        ds = SparDataset(str(annotated_dir), tok)
        sample = ds[0]
        assert len(sample["tags"]) == len(sample["input_ids"])

    def test_object_span_annotated_correctly(self, annotated_dir):
        """'door' at word index 1 (token index 2 after [CLS]) should be BH-obj."""
        tok = _make_mock_tokenizer(SENTENCE)
        ds = SparDataset(str(annotated_dir), tok)
        tags = ds[0]["tags"]
        # token index 2 = "door" (0=CLS, 1="The", 2="door", ...)
        assert tags[2].item() == TAG_TO_IDX["BH-obj"]

    def test_action_span_annotated_correctly(self, annotated_dir):
        """'shall' at word index 2 (token index 3 after [CLS]) should be BH-act."""
        tok = _make_mock_tokenizer(SENTENCE)
        ds = SparDataset(str(annotated_dir), tok)
        tags = ds[0]["tags"]
        assert tags[3].item() == TAG_TO_IDX["BH-act"]

    def test_outside_tokens_are_pd_pad(self, annotated_dir):
        tok = _make_mock_tokenizer(SENTENCE)
        ds = SparDataset(str(annotated_dir), tok)
        tags = ds[0]["tags"]
        pd_pad_id = TAG_TO_IDX["PD-pad"]
        # [CLS] (0), "The" (1), and [SEP] (-1) should all be PD-pad
        assert tags[0].item() == pd_pad_id
        assert tags[1].item() == pd_pad_id
        assert tags[-1].item() == pd_pad_id

    def test_custom_tag_vocab_is_used(self, txt_only_dir):
        custom_vocab = {"PD-pad": 0, "BH-obj": 99}
        tok = _make_mock_tokenizer(SENTENCE)
        ds = SparDataset(str(txt_only_dir), tok, tag_to_idx=custom_vocab)
        assert ds.tag_vocab == custom_vocab

    def test_idx_to_tag_is_reverse_of_tag_vocab(self, txt_only_dir):
        tok = _make_mock_tokenizer(SENTENCE)
        ds = SparDataset(str(txt_only_dir), tok)
        for tag, idx in ds.tag_vocab.items():
            assert ds.idx_to_tag[idx] == tag


# ---------------------------------------------------------------------------
# collate_fn
# ---------------------------------------------------------------------------

def _make_sample(seq_len: int, has_tags: bool = True) -> dict:
    """Build a minimal dataset sample for collation tests."""
    return {
        "input_ids":      torch.ones(seq_len, dtype=torch.long),
        "attention_mask": torch.ones(seq_len, dtype=torch.bool),
        "tags":           torch.zeros(seq_len, dtype=torch.long) if has_tags else None,
        "words":          [f"w{i}" for i in range(seq_len)],
        "sentence":       "test sentence",
        "doc_id":         "doc_0",
    }


class TestCollateFn:

    def test_pads_input_ids_to_max_length(self):
        batch = [_make_sample(3), _make_sample(5)]
        out = collate_fn(batch)
        assert out["input_ids"].shape == (2, 5)

    def test_short_sample_padded_with_zeros(self):
        batch = [_make_sample(3), _make_sample(5)]
        out = collate_fn(batch)
        # Positions 3 and 4 of the first sample should be padded (0)
        assert out["input_ids"][0, 3].item() == 0
        assert out["input_ids"][0, 4].item() == 0

    def test_attention_mask_false_for_padding(self):
        batch = [_make_sample(3), _make_sample(5)]
        out = collate_fn(batch)
        assert not out["attention_mask"][0, 3]
        assert not out["attention_mask"][0, 4]

    def test_attention_mask_true_for_real_tokens(self):
        batch = [_make_sample(3), _make_sample(5)]
        out = collate_fn(batch)
        assert out["attention_mask"][0, :3].all()
        assert out["attention_mask"][1, :5].all()

    def test_tags_padded_with_zero(self):
        batch = [_make_sample(3), _make_sample(5)]
        out = collate_fn(batch)
        assert out["tags"][0, 3].item() == 0
        assert out["tags"][0, 4].item() == 0

    def test_tags_none_when_no_annotations(self):
        batch = [_make_sample(3, has_tags=False), _make_sample(4, has_tags=False)]
        out = collate_fn(batch)
        assert out["tags"] is None

    def test_metadata_lists_have_correct_length(self):
        batch = [_make_sample(3), _make_sample(5)]
        out = collate_fn(batch)
        assert len(out["words"]) == 2
        assert len(out["sentences"]) == 2
        assert len(out["doc_ids"]) == 2

    def test_uniform_length_batch_needs_no_padding(self):
        batch = [_make_sample(4), _make_sample(4)]
        out = collate_fn(batch)
        assert out["input_ids"].shape == (2, 4)
        assert out["attention_mask"].all()

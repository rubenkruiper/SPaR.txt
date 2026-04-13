"""
BRAT dataset reader — AllenNLP-free rewrite.

Replaces the AllenNLP DatasetReader / Instance / TextField stack with a plain
``torch.utils.data.Dataset``.  The BRAT parsing and tag-computation logic in
``spar_lib/readers/reader_utils/my_read_utils.py`` is unchanged; the only
adapter layer added here is the ``_Token`` shim, which exposes the ``.idx``
and ``.idx_end`` character-offset attributes that ``my_read_utils`` expects.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
from transformers import BertTokenizerFast

from spar_lib.readers.reader_utils.my_read_utils import get_annotations_from_ann_file


# ---------------------------------------------------------------------------
# Tag vocabulary  (order = integer label ids used by the CRF)
# ---------------------------------------------------------------------------

#: Ordered list of all tag strings produced by the DiscontiguousTest scheme.
#: Position in this list is the integer label id fed to the model and CRF.
TAGS: List[str] = [
    "PD-pad",                                    # 0  — "outside" / padding
    "BH-obj", "IH-obj", "BD-obj", "ID-obj",     # 1-4
    "BH-act", "IH-act", "BD-act", "ID-act",     # 5-8
    "BH-func", "IH-func",                        # 9-10
    "BH-dis",  "IH-dis",                         # 11-12
]

#: Map tag string → integer label id.
#:
#: Examples::
#:
#:     >>> TAG_TO_IDX["PD-pad"]
#:     0
#:     >>> TAG_TO_IDX["BH-obj"]
#:     1
#:     >>> TAG_TO_IDX["BD-act"]
#:     7
TAG_TO_IDX: Dict[str, int] = {tag: i for i, tag in enumerate(TAGS)}

#: Reverse map integer label id → tag string.
IDX_TO_TAG: Dict[int, str] = {i: tag for i, tag in enumerate(TAGS)}


# ---------------------------------------------------------------------------
# Token shim
# ---------------------------------------------------------------------------

@dataclass
class _Token:
    """
    Minimal token object exposing the ``.idx`` / ``.idx_end`` character-offset
    attributes required by ``my_read_utils.brat_to_PretainedTransformerTokenizer``.

    Special tokens (``[CLS]``, ``[SEP]``) receive ``idx=None`` so that
    ``brat_indices_to_token_indices`` skips them — mirroring the behaviour of
    AllenNLP's ``PretrainedTransformerTokenizer``.
    """
    text: str
    idx: Optional[int]      # char start offset; None for special tokens
    idx_end: Optional[int]  # char end offset (exclusive); None for special tokens


# ---------------------------------------------------------------------------
# Tokenization helper
# ---------------------------------------------------------------------------

def _tokenize(
    text: str,
    tokenizer: BertTokenizerFast,
    max_length: int,
) -> Tuple[dict, List[_Token]]:
    """
    Tokenise *text* and return the HuggingFace encoding dict together with a
    list of :class:`_Token` shims whose ``.idx`` / ``.idx_end`` fields match
    what ``my_read_utils`` expects.

    Parameters
    ----------
    text :
        Raw sentence string.
    tokenizer :
        A ``BertTokenizerFast`` instance (or compatible).
    max_length :
        Maximum number of tokens; sequences are truncated to this length.

    Returns
    -------
    encoding :
        HuggingFace tokenizer output dict (``input_ids``, ``attention_mask``,
        ``offset_mapping``, ``special_tokens_mask``).
    token_list :
        One :class:`_Token` per position in ``encoding["input_ids"]``.
        Special tokens have ``idx=None`` / ``idx_end=None``.
    """
    encoding = tokenizer(
        text,
        return_offsets_mapping=True,
        return_special_tokens_mask=True,
        max_length=max_length,
        truncation=True,
    )
    token_strings = tokenizer.convert_ids_to_tokens(encoding["input_ids"])
    token_list = [
        _Token(
            text=tok_str,
            idx=None if is_special else start,
            idx_end=None if is_special else end,
        )
        for tok_str, (start, end), is_special in zip(
            token_strings,
            encoding["offset_mapping"],
            encoding["special_tokens_mask"],
        )
    ]
    return encoding, token_list


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SparDataset(Dataset):
    """
    ``torch.utils.data.Dataset`` over a directory of BRAT ``.txt`` / ``.ann``
    file pairs.

    Each sample is a ``dict`` with the following keys:

    ``input_ids``
        ``LongTensor`` of shape ``(seq_len,)`` — BERT token ids.
    ``attention_mask``
        ``BoolTensor`` of shape ``(seq_len,)`` — ``True`` for real tokens.
    ``tags``
        ``LongTensor`` of shape ``(seq_len,)`` when a ``.ann`` file is present,
        otherwise ``None``.
    ``words``
        ``List[str]`` of length ``seq_len`` — subword token strings.
    ``sentence``
        Original sentence string.
    ``doc_id``
        Stem of the source ``.txt`` file.

    Parameters
    ----------
    data_dir :
        Path to the directory containing ``.txt`` (and optionally ``.ann``) files.
    tokenizer :
        A ``BertTokenizerFast`` instance.
    tag_to_idx :
        Optional custom tag→id mapping.  Defaults to :data:`TAG_TO_IDX`.
    max_length :
        Maximum token sequence length (truncation applied). Default: 512.
    """

    def __init__(
        self,
        data_dir: str,
        tokenizer: BertTokenizerFast,
        tag_to_idx: Optional[Dict[str, int]] = None,
        max_length: int = 512,
    ) -> None:
        self.tokenizer = tokenizer
        self.tag_to_idx = tag_to_idx if tag_to_idx is not None else TAG_TO_IDX
        self.max_length = max_length
        self._samples: List[dict] = self._load(Path(data_dir))

    # ------------------------------------------------------------------
    # Internal loading
    # ------------------------------------------------------------------

    def _load(self, data_dir: Path) -> List[dict]:
        txt_files = sorted(data_dir.glob("*.txt"))
        ann_by_stem = {f.stem: f for f in data_dir.glob("*.ann")}

        samples = []
        for txt_file in txt_files:
            doc_id = txt_file.stem
            sentence = txt_file.read_text().strip()
            encoding, token_list = _tokenize(sentence, self.tokenizer, self.max_length)

            tags: Optional[torch.Tensor] = None
            if doc_id in ann_by_stem:
                try:
                    tag_strings = get_annotations_from_ann_file(
                        ann_by_stem[doc_id], sentence, token_list
                    )
                    tags = torch.tensor(
                        [self.tag_to_idx.get(t, 0) for t in tag_strings],
                        dtype=torch.long,
                    )
                except ValueError:
                    # Annotation indices couldn't be aligned to tokens
                    # (e.g. due to truncation). Skip annotations for this sample.
                    pass

            samples.append({
                "input_ids": torch.tensor(encoding["input_ids"], dtype=torch.long),
                "attention_mask": torch.tensor(encoding["attention_mask"], dtype=torch.bool),
                "tags": tags,
                "words": self.tokenizer.convert_ids_to_tokens(encoding["input_ids"]),
                "sentence": sentence,
                "doc_id": doc_id,
            })
        return samples

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict:
        return self._samples[idx]

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def tag_vocab(self) -> Dict[str, int]:
        """Tag string → integer id mapping used by this dataset."""
        return self.tag_to_idx

    @property
    def idx_to_tag(self) -> Dict[int, str]:
        """Integer id → tag string reverse mapping."""
        return {v: k for k, v in self.tag_to_idx.items()}


# ---------------------------------------------------------------------------
# Collation
# ---------------------------------------------------------------------------

def collate_fn(batch: List[dict]) -> dict:
    """
    Pad a list of :class:`SparDataset` samples to the same length for batching.

    Padding fills ``input_ids`` with ``0`` (BERT's ``[PAD]`` token id),
    ``attention_mask`` with ``False``, and ``tags`` with ``0`` (``PD-pad``).

    Parameters
    ----------
    batch :
        List of sample dicts as returned by ``SparDataset.__getitem__``.

    Returns
    -------
    dict
        Batched tensors:
        ``input_ids``  ``(B, L)``, ``attention_mask`` ``(B, L)``,
        ``tags``       ``(B, L)`` or ``None``,
        ``words``      list of B word-lists,
        ``sentences``  list of B sentence strings,
        ``doc_ids``    list of B doc-id strings.
    """
    max_len = max(len(item["input_ids"]) for item in batch)
    has_tags = any(item["tags"] is not None for item in batch)

    input_ids = torch.zeros(len(batch), max_len, dtype=torch.long)
    attention_mask = torch.zeros(len(batch), max_len, dtype=torch.bool)
    tags = torch.zeros(len(batch), max_len, dtype=torch.long) if has_tags else None

    for i, item in enumerate(batch):
        n = len(item["input_ids"])
        input_ids[i, :n] = item["input_ids"]
        attention_mask[i, :n] = item["attention_mask"]
        if has_tags and item["tags"] is not None:
            t = item["tags"][:max_len]
            tags[i, :len(t)] = t

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "tags": tags,
        "words": [item["words"] for item in batch],
        "sentences": [item["sentence"] for item in batch],
        "doc_ids": [item["doc_id"] for item in batch],
    }

"""
SPaR.txt sequence tagger — AllenNLP-free rewrite.

Architecture (attention variant, from the original span_tagger_att.py):

    BERT (frozen)  →  dropout
    →  biLSTM(hidden=384, bidirectional → 768-dim)
    →  RelativeGlobalAttention(d_model=768, heads=12)   [Huang et al. 2018]
    →  concat([biLSTM_out, attn_out])                   → 1536-dim
    →  dropout
    →  FFNN(1536 → 60, ReLU, dropout)
    →  Linear(60 → num_tags)
    →  CRF  (DiscontiguousTest constraints, no start/end transitions)

Removed from original:
  - ``Model`` AllenNLP base class → plain ``nn.Module``
  - ``TextFieldEmbedder`` / ``TokenIndexer`` → ``BertModel`` directly
  - ``Seq2SeqEncoder`` AllenNLP wrapper → ``nn.LSTM``
  - ``TimeDistributed`` → ``nn.Linear`` (broadcasts over seq dim natively)
  - ``FeedForward`` AllenNLP module → ``nn.Sequential``
  - ``CategoricalAccuracy`` → dropped (F1 is the validation signal)
  - ``InitializerApplicator`` → PyTorch default init
  - ``Vocabulary`` → plain ``Dict[int, str]`` label map
  - ``@overrides`` decorator
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from transformers import BertModel

from spar_lib.metrics.crf_f1_measure import SpanBasedF1Measure
from spar_lib.modules.adapted_crf import ConditionalRandomField, allowed_transitions
from spar_lib.modules.multi_head_positional_attention import RelativeGlobalAttention
from spar_lib.readers.tagging_reader import IDX_TO_TAG, TAG_TO_IDX


class SparTagger(nn.Module):
    """
    Sequence tagger for SPaR.txt shallow parsing.

    Parameters
    ----------
    num_tags :
        Number of output tags.  Defaults to ``len(TAG_TO_IDX)`` (13).
    label_map :
        ``{int_id: tag_string}`` reverse vocabulary.  Defaults to
        ``IDX_TO_TAG`` from :mod:`spar_lib.readers.tagging_reader`.
    bert_model_name :
        HuggingFace model identifier for the BERT encoder.
    lstm_hidden_size :
        Per-direction hidden size of the biLSTM (output dim is doubled).
    ffnn_hidden_size :
        Hidden units in the feed-forward layer between attention and CRF.
    dropout :
        Dropout probability applied after BERT and after the attention concat.
    freeze_bert :
        Whether to freeze BERT parameters during training.
    attention_heads :
        Number of heads for :class:`RelativeGlobalAttention`.
    """

    def __init__(
        self,
        num_tags: int = len(TAG_TO_IDX),
        label_map: Optional[Dict[int, str]] = None,
        bert_model_name: str = "bert-base-cased",
        lstm_hidden_size: int = 384,
        ffnn_hidden_size: int = 60,
        dropout: float = 0.05,
        freeze_bert: bool = True,
        attention_heads: int = 12,
    ) -> None:
        super().__init__()

        self._label_map = label_map if label_map is not None else IDX_TO_TAG

        # ---- BERT encoder ----------------------------------------
        self.bert = BertModel.from_pretrained(bert_model_name)
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        bert_dim = self.bert.config.hidden_size  # 768

        # ---- Sequence encoder ------------------------------------
        self._dropout = nn.Dropout(dropout)
        lstm_out_dim = lstm_hidden_size * 2       # bidirectional → 768
        self.lstm = nn.LSTM(
            input_size=bert_dim,
            hidden_size=lstm_hidden_size,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )

        # ---- Relative global attention ---------------------------
        # Input/output: (B, L, lstm_out_dim)
        self.attention = RelativeGlobalAttention(lstm_out_dim, attention_heads)

        # ---- Feed-forward + projection ---------------------------
        concat_dim = lstm_out_dim * 2             # concat([lstm, attn]) → 1536
        self.ffnn = nn.Sequential(
            nn.Linear(concat_dim, ffnn_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.tag_projection = nn.Linear(ffnn_hidden_size, num_tags)

        # ---- CRF with DiscontiguousTest constraints ---------------
        constraints = allowed_transitions("DiscontiguousTest", self._label_map)
        self.crf = ConditionalRandomField(
            num_tags=num_tags,
            constraints=constraints,
            include_start_end_transitions=False,
        )

        # ---- F1 metric (accumulates across batches) --------------
        self._f1_metric = SpanBasedF1Measure(self._label_map)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.BoolTensor,
        gold_tags: Optional[torch.LongTensor] = None,
        words: Optional[List[List[str]]] = None,
        sentences: Optional[List[str]] = None,
        doc_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Parameters
        ----------
        input_ids :
            ``(B, L)`` BERT token ids.
        attention_mask :
            ``(B, L)`` boolean mask; ``True`` for real tokens.
        gold_tags :
            ``(B, L)`` integer tag ids.  When provided, computes CRF loss
            and updates the F1 metric.
        words, sentences, doc_ids :
            Optional metadata forwarded unchanged into the output dict for
            use by the serving layer (``spar_serving_utils.parse_spar_output``).

        Returns
        -------
        dict with keys:
            ``"logits"``   — ``(B, L, num_tags)`` raw scores.
            ``"mask"``     — ``(B, L)`` bool attention mask.
            ``"tags"``     — ``List[List[str]]`` decoded tag strings, one per item.
            ``"loss"``     — scalar CRF NLL (only when ``gold_tags`` provided).
            ``"words"``, ``"sentences"``, ``"doc_ids"`` — pass-through metadata.
        """
        bool_mask = attention_mask.bool()

        # ---- BERT ------------------------------------------------
        bert_out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask.long(),
        )
        embedded = self._dropout(bert_out.last_hidden_state)  # (B, L, 768)

        # ---- biLSTM (packed for correctness with padding) --------
        lengths = bool_mask.sum(dim=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths, batch_first=True, enforce_sorted=False
        )
        lstm_packed_out, _ = self.lstm(packed)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
            lstm_packed_out, batch_first=True, total_length=embedded.size(1)
        )  # (B, L, lstm_hidden*2)

        # ---- Relative global attention ---------------------------
        attn_out = self.attention(lstm_out)       # (B, L, lstm_hidden*2)

        # ---- Concat + dropout ------------------------------------
        encoded = self._dropout(
            torch.cat([lstm_out, attn_out], dim=-1)
        )                                         # (B, L, concat_dim)

        # ---- FFNN + projection -----------------------------------
        logits = self.tag_projection(self.ffnn(encoded))  # (B, L, num_tags)

        # ---- Viterbi decode --------------------------------------
        best_paths = self.crf.viterbi_tags(logits, bool_mask, top_k=None)
        # Decode integer ids → tag strings (top_k=None returns flat [(path, score), …])
        predicted_tags: List[List[str]] = [
            [self._label_map[tag_id] for tag_id in path]
            for (path, _score) in best_paths
        ]

        output: Dict[str, Any] = {
            "logits": logits,
            "mask": bool_mask,
            "tags": predicted_tags,
        }

        # ---- Loss + metric (training / evaluation) ---------------
        if gold_tags is not None:
            output["loss"] = -self.crf(logits, gold_tags, bool_mask)
            # Build a one-hot tensor from Viterbi paths so the F1 metric
            # accumulates the same predictions that are reported at test time,
            # not the raw-logit argmax (which ignores CRF constraints).
            viterbi_onehot = torch.zeros_like(logits)
            for i, path in enumerate(predicted_tags):
                for j, tag_str in enumerate(path):
                    viterbi_onehot[i, j, TAG_TO_IDX[tag_str]] = 1.0
            self._f1_metric(viterbi_onehot.detach(), gold_tags, bool_mask)

        # ---- Pass-through metadata (used by serving layer) -------
        if words     is not None: output["words"]     = words
        if sentences is not None: output["sentences"] = sentences
        if doc_ids   is not None: output["doc_ids"]   = doc_ids

        return output

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        """Return accumulated span F1 metrics and optionally reset counters."""
        return self._f1_metric.get_metric(reset=reset)

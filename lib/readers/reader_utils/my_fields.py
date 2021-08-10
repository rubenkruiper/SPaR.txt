from typing import Dict, Tuple

from overrides import overrides
import torch

from allennlp.common.checks import ConfigurationError
from allennlp.data.fields.field import Field
from allennlp.data.fields.sequence_field import SequenceField


class DiscontiguousMaskField(Field[torch.Tensor]):
    """
    Represents a possibly discontiguous mask over a textfield.
    """

    __slots__ = ["mask_field", "index_field", "sequence_field", "span_name"]

    def __init__(self,
                 span_tuple: Tuple[int, int, int, int],
                 sequence_field: SequenceField,
                 span_name: str ) -> None:

        ss, se, es, ee = span_tuple
        self.sequence_field = sequence_field
        self.span_name = span_name

        # Check if inputs are valid
        if not isinstance(ss, int) or not isinstance(se, int) or not isinstance(es, int) \
                or not isinstance(ee, int):
            raise TypeError(
                f"SpanFields must be passed integer indices. Found span indices: "
                f"({ss}, {es}, {se}, {ee}) with types "
                f"({type(ss)}, {type(es)}, {type(se)}, {type(ee)})"
            )
        if ss > ee or ss > se or ss > se or se > es or se > ee or es > ee:
            raise ValueError(
                f"Issue with the order of span indices; ({ss}, {se}, {es}, {ee})."
            )

        if ee > self.sequence_field.sequence_length() - 1:
            raise ValueError(
                f"span_end must be <= len(sequence_length) - 1, but found "
                f"{ee} and {self.sequence_field.sequence_length() - 1} respectively."
            )

        self.mask_field = self.span_indices_to_mask(span_tuple, sequence_field.sequence_length())
        self.index_field = self.span_indices_to_index_select(self.mask_field)

    @overrides
    def get_padding_lengths(self) -> Dict[str, int]:
        if self.sequence_field is None:
            raise ConfigurationError(
                "You must call pass a sequence_field in order to determine mask padding lengths."
            )
        return {self.span_name: len(self.sequence_field)}

    # @overrides
    def index_as_tensor(self, padding_lengths: Dict[str, int]) -> torch.Tensor:
        """
        Converts the list of indices to a tensor, using 0 (BERT's [CLS] token) as padding.
        """
        padding_length = padding_lengths[self.span_name]
        padded_index_field = self.index_field + [0] * (padding_length - len(self.index_field))
        return torch.Tensor(padded_index_field).long() # .bool()

    @overrides
    def as_tensor(self, padding_lengths: Dict[str, int]) -> torch.Tensor:
        """
        Pads the [0,1] tensor # and converts to boolean values (if it has to be a mask).
        FIXME NOTE: Not used currently in the seq2vec approach, because discontiguous masks
         cannot be passed to (simple?) recurrent models!
        :param padding_lengths:
        :return:
        """
        padding_length = padding_lengths[self.span_name]
        padded_mask = self.mask_field + [0] * (padding_length - len(self.mask_field))
        return torch.Tensor(padded_mask).long()

    def span_indices_to_index_select(self, mask_list):
        """        Convert span indices to all indices that should be selected      """
        index_list = [i for i, element in enumerate(mask_list) if element != 0]
        return index_list

    def span_indices_to_mask(self, indices, mask_length):
        """ Convert span indices to masks """
        mask = [0] * mask_length
        ss, se, es, ee = indices

        if ss == -1:
            # empty field
            return mask

        if se == ee:
            # contiguous span
            mask[ss:ee + 1] = [1] * (ee + 1 - ss)
        else:
            # discontiguous span
            mask[ss:se + 1] = [1] * (se + 1 - ss)
            mask[es:ee + 1] = [1] * (ee + 1 - es)
        return mask

    @overrides
    def empty_field(self):
        # used to be 'non_span_token'
        # now points to [SEP]
        return DiscontiguousMaskField((-1, -1, -1, -1), self.sequence_field.empty_field(), 'span_mask')
#                                      'span_for_token_' + str(self.sequence_field.sequence_length() - 1))

    def __str__(self) -> str:
        # return f"DiscontiguousSpanField with spans: ({self.span_start}:{self.gap_start} and {self.gap_end}:{self.span_end})."
        return "DiscontiguousMaskField."

    def __eq__(self, other) -> bool:
        # if isinstance(other, tuple) and len(other) == 4:
            # return other == (self.span_start, self.gap_start, self.gap_end, self.span_end)
        return super().__eq__(other)

    def __len__(self):
        # return 4
        return len(self.sequence_field)
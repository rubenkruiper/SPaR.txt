from typing import List, Callable, Tuple, TypeVar
from allennlp.data.tokenizers import Token


T = TypeVar("T", str, Token)


def my_enumerate_spans(
    sentence: List[T],
    offset: int = 0,
    max_gap_width: int = 1,
    max_span_width: int = None,
    filter_function: Callable[[List[T]], bool] = None,
) -> List[Tuple[int, int]]:
    """
    """
    max_span_width = max_span_width or len(sentence)
    filter_function = filter_function or (lambda x: True)
    spans: List[Tuple[int, int, int, int]] = []

    # contiguous spans
    for start_index in range(len(sentence)):
        # minimum span length is 1
        first_end_index = min(start_index, len(sentence))
        last_end_index = min(start_index + max_span_width, len(sentence))
        for end_index in range(first_end_index, last_end_index):
            start = offset + start_index
            end = offset + end_index
            # add 1 to end index because span indices are inclusive.
            if filter_function(sentence[slice(start_index, end_index + 1)]):
                # TODO - determine how to handle the single span masks
                # index 0 should point at [CLS] token?
                spans.append((start, end, end, end))

    # For discontiguous spans we only rely on variations of [start, end] that contain a
    # gap, where the gap has maximum length max_gap_width
    # TODO - extend to more than 2 discontiguous spans?
    for gap_size in range(1, max_gap_width + 1):
        for s, e, _, _ in spans:
            if e - s > gap_size:
                # every combo where len(part1) + len(part2) + gap = e - s
                gaps_exclusive = [(s_end, s_end + gap_size + 1) for s_end in range(s, e - 1) if
                                  (s_end + gap_size + 1 <= e)]
                [spans.append((s, s_end, e_start, e)) for s_end, e_start in gaps_exclusive]

    return spans

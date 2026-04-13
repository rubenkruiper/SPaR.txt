"""
Conditional random field — AllenNLP-free rewrite.

All logic is identical to the original; the only changes are:
  - ConfigurationError  →  ValueError
  - allennlp.nn.util.logsumexp  →  torch.logsumexp
  - allennlp.nn.util.viterbi_decode  →  _viterbi_decode (implemented below)
"""
from typing import List, Tuple, Dict, Union

import torch

VITERBI_DECODING = Tuple[List[int], float]  # a list of tags, and a viterbi score


# ---------------------------------------------------------------------------
# Viterbi decoder (replaces allennlp.nn.util.viterbi_decode)
# ---------------------------------------------------------------------------

def _viterbi_decode(
    tag_sequence: torch.Tensor,
    transition_matrix: torch.Tensor,
    top_k: int = 1,
) -> Tuple[List[List[int]], List[float]]:
    """
    Viterbi decoding over a sequence of emission scores and a transition matrix.

    Parameters
    ----------
    tag_sequence : (seq_len, num_tags)
        Emission scores per timestep.  The caller pads the first position with
        a START sentinel and the last with an END sentinel (only those tag ids
        have finite scores; all others are –10 000).
    transition_matrix : (num_tags, num_tags)
        transition_matrix[i, j] = score of transitioning FROM tag i TO tag j.
        Forbidden transitions are set to –10 000 by the caller.
    top_k : int
        Only 1 is supported (the model never uses k > 1 in practice).

    Returns
    -------
    paths  : list of length top_k, each element is a List[int] tag sequence
    scores : list of length top_k, each element is a float Viterbi score
    """
    if top_k != 1:
        raise NotImplementedError("top_k > 1 is not supported")

    seq_len, num_tags = tag_sequence.shape

    # viterbi[tag] = best total score to reach that tag at the current step
    viterbi = tag_sequence[0].clone()
    backpointers: List[torch.Tensor] = []

    for t in range(1, seq_len):
        # (num_tags, 1) broadcasts with (num_tags, num_tags):
        # trans_scores[i, j] = viterbi[i] + transition[i, j]
        trans_scores = viterbi.unsqueeze(1) + transition_matrix
        best_scores, best_from = trans_scores.max(0)  # max over from-tag dim
        backpointers.append(best_from)
        viterbi = best_scores + tag_sequence[t]

    # Find the best final tag and backtrack
    best_last_score, best_last_tag = viterbi.max(0)
    path = [best_last_tag.item()]
    for bp in reversed(backpointers):
        path.append(bp[path[-1]].item())
    path.reverse()

    return [path], [best_last_score.item()]


# ---------------------------------------------------------------------------
# Transition constraints
# ---------------------------------------------------------------------------

def allowed_transitions(constraint_type: str, labels: Dict[int, str]) -> List[Tuple[int, int]]:
    """
    Given labels and a constraint type, returns the allowed transitions. It will
    additionally include transitions for the start and end states, which are used
    by the conditional random field.

    Parameters
    ----------
    constraint_type : str
        Indicates which constraint to apply ~ assuming 'DiscontiguousTest'
    labels : Dict[int, str]
        A mapping {label_id -> label}. Most commonly this would be the value from
        Vocabulary.get_index_to_token_vocabulary()

    Returns
    -------
    List[Tuple[int, int]]
        The allowed transitions (from_label_id, to_label_id).
    """
    num_labels = len(labels)
    start_tag = num_labels
    end_tag = num_labels + 1
    labels_with_boundaries = list(labels.items()) + [(start_tag, "START"), (end_tag, "END")]

    allowed = []
    for from_label_index, from_label in labels_with_boundaries:
        if from_label in ("START", "END"):
            from_tag = from_label
            from_entity = ""
        else:
            from_tag, from_entity = from_label.split('-')
        for to_label_index, to_label in labels_with_boundaries:
            if to_label in ("START", "END"):
                to_tag = to_label
                to_entity = ""
            else:
                to_tag, to_entity = to_label.split('-')
            if is_transition_allowed(constraint_type, from_tag, from_entity, to_tag, to_entity):
                allowed.append((from_label_index, to_label_index))
    return allowed


def is_transition_allowed(
    constraint_type: str, from_tag: str, from_type: str, to_tag: str, to_type: str
) -> bool:
    """
    Return whether a CRF transition is allowed under the given constraint type.

    Parameters
    ----------
    constraint_type : str
        Currently only ``'DiscontiguousTest'`` is supported.
    from_tag, from_type : str
        The tag prefix (e.g. ``'BH'``) and entity type (e.g. ``'obj'``) of the
        source state.  Use empty string for pseudo-tags (``'START'``, ``'END'``,
        ``'PD'``).
    to_tag, to_type : str
        Same fields for the destination state.

    Examples
    --------
    Only PD is reachable from START:

    >>> is_transition_allowed('DiscontiguousTest', 'START', '', 'PD', '')
    True
    >>> is_transition_allowed('DiscontiguousTest', 'START', '', 'BH', 'obj')
    False

    BH/IH/BD/ID can all continue inside the *same* type:

    >>> is_transition_allowed('DiscontiguousTest', 'BH', 'obj', 'IH', 'obj')
    True
    >>> is_transition_allowed('DiscontiguousTest', 'BH', 'obj', 'IH', 'act')
    False

    Any span tag can transition to PD or start a new BH/BD:

    >>> is_transition_allowed('DiscontiguousTest', 'IH', 'obj', 'BH', 'act')
    True
    >>> is_transition_allowed('DiscontiguousTest', 'IH', 'obj', 'PD', '')
    True

    Only PD can transition to END:

    >>> is_transition_allowed('DiscontiguousTest', 'PD', '', 'END', '')
    True
    >>> is_transition_allowed('DiscontiguousTest', 'IH', 'obj', 'END', '')
    False
    """

    if to_tag == "START" or from_tag == "END":
        # Cannot transition into START or from END
        return False

    if constraint_type == "DiscontiguousTest":
        if from_tag == "START":
            return to_tag in ("PD",)
        if to_tag == "END":
            return from_tag in ("PD",)
        if from_tag == "PD":
            return to_tag in ("BH", "END")
        return any(
            [
                # for any types;
                # BH can transition to BH-*, BD-*, P
                # BD can transition to BH-*, BD-*, P
                # IH can transition to BH-*, BD-*, P
                # ID can transition to BH-*, BD-*, P
                from_tag in ("BH", "BD", "IH", "ID") and to_tag in ("BH", "BD", "PD"),
                # for same type;
                # BH-x can only transition to IH-x
                # BD-x can only transition to ID-x
                # IH-x can only transition to IH-x
                # ID-x can only transition to ID-x
                from_tag in ("BH", "BD", "IH", "ID") and to_tag in ("IH", "ID") and from_type == to_type
            ]
        )
    else:
        raise ValueError(f"Unknown constraint type: {constraint_type}")


# ---------------------------------------------------------------------------
# Conditional Random Field
# ---------------------------------------------------------------------------

class ConditionalRandomField(torch.nn.Module):
    """
    This module uses the "forward-backward" algorithm to compute
    the log-likelihood of its inputs assuming a conditional random field model.

    See, e.g. http://www.cs.columbia.edu/~mcollins/fb.pdf

    Parameters
    ----------
    num_tags : int
        The number of tags.
    constraints : List[Tuple[int, int]], optional
        An optional list of allowed transitions (from_tag_id, to_tag_id).
        These are applied to `viterbi_tags()` but do not affect `forward()`.
        These should be derived from `allowed_transitions` so that the
        start and end transitions are handled correctly for your tag type.
    include_start_end_transitions : bool, optional (default True)
        Whether to include the start and end transition parameters.
    """

    def __init__(
        self,
        num_tags: int,
        constraints: List[Tuple[int, int]] = None,
        include_start_end_transitions: bool = True,
    ) -> None:
        super().__init__()
        self.num_tags = num_tags

        # transitions[i, j] is the logit for transitioning from state i to state j.
        self.transitions = torch.nn.Parameter(torch.Tensor(num_tags, num_tags))

        # _constraint_mask indicates valid transitions (based on supplied constraints).
        # Include special start of sequence (num_tags + 1) and end of sequence tags (num_tags + 2)
        if constraints is None:
            # All transitions are valid.
            constraint_mask = torch.Tensor(num_tags + 2, num_tags + 2).fill_(1.0)
        else:
            constraint_mask = torch.Tensor(num_tags + 2, num_tags + 2).fill_(0.0)
            for i, j in constraints:
                constraint_mask[i, j] = 1.0

        self._constraint_mask = torch.nn.Parameter(constraint_mask, requires_grad=False)

        # Also need logits for transitioning from "start" state and to "end" state.
        self.include_start_end_transitions = include_start_end_transitions
        if include_start_end_transitions:
            self.start_transitions = torch.nn.Parameter(torch.Tensor(num_tags))
            self.end_transitions = torch.nn.Parameter(torch.Tensor(num_tags))

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_normal_(self.transitions)
        if self.include_start_end_transitions:
            torch.nn.init.normal_(self.start_transitions)
            torch.nn.init.normal_(self.end_transitions)

    def _input_likelihood(self, logits: torch.Tensor, mask: torch.BoolTensor) -> torch.Tensor:
        """
        Computes the (batch_size,) denominator term for the log-likelihood, which is the
        sum of the likelihoods across all possible state sequences.
        """
        batch_size, sequence_length, num_tags = logits.size()

        # Transpose batch size and sequence dimensions
        mask = mask.transpose(0, 1).contiguous()
        logits = logits.transpose(0, 1).contiguous()

        # Initial alpha is the (batch_size, num_tags) tensor of likelihoods combining the
        # transitions to the initial states and the logits for the first timestep.
        if self.include_start_end_transitions:
            alpha = self.start_transitions.view(1, num_tags) + logits[0]
        else:
            alpha = logits[0]

        for i in range(1, sequence_length):
            emit_scores = logits[i].view(batch_size, 1, num_tags)
            transition_scores = self.transitions.view(1, num_tags, num_tags)
            broadcast_alpha = alpha.view(batch_size, num_tags, 1)

            inner = broadcast_alpha + emit_scores + transition_scores

            alpha = torch.logsumexp(inner, dim=1) * mask[i].view(batch_size, 1) + alpha * (
                ~mask[i]
            ).view(batch_size, 1)

        if self.include_start_end_transitions:
            stops = alpha + self.end_transitions.view(1, num_tags)
        else:
            stops = alpha

        return torch.logsumexp(stops, dim=-1)

    def _joint_likelihood(
        self, logits: torch.Tensor, tags: torch.Tensor, mask: torch.BoolTensor
    ) -> torch.Tensor:
        """
        Computes the numerator term for the log-likelihood, which is just score(inputs, tags)
        """
        batch_size, sequence_length, _ = logits.data.shape

        # Transpose batch size and sequence dimensions:
        logits = logits.transpose(0, 1).contiguous()
        mask = mask.transpose(0, 1).contiguous()
        tags = tags.transpose(0, 1).contiguous()

        if self.include_start_end_transitions:
            score = self.start_transitions.index_select(0, tags[0])
        else:
            score = 0.0

        for i in range(sequence_length - 1):
            current_tag, next_tag = tags[i], tags[i + 1]
            transition_score = self.transitions[current_tag.view(-1), next_tag.view(-1)]
            emit_score = logits[i].gather(1, current_tag.view(batch_size, 1)).squeeze(1)
            score = score + transition_score * mask[i + 1] + emit_score * mask[i]

        last_tag_index = mask.sum(0).long() - 1
        last_tags = tags.gather(0, last_tag_index.view(1, batch_size)).squeeze(0)

        if self.include_start_end_transitions:
            last_transition_score = self.end_transitions.index_select(0, last_tags)
        else:
            last_transition_score = 0.0

        last_inputs = logits[-1]
        last_input_score = last_inputs.gather(1, last_tags.view(-1, 1))
        last_input_score = last_input_score.squeeze()

        score = score + last_transition_score + last_input_score * mask[-1]

        return score

    def forward(
        self, inputs: torch.Tensor, tags: torch.Tensor, mask: torch.BoolTensor = None
    ) -> torch.Tensor:
        """
        Computes the log likelihood.
        """
        if mask is None:
            mask = torch.ones(*tags.size(), dtype=torch.bool)
        else:
            mask = mask.to(torch.bool)

        log_denominator = self._input_likelihood(inputs, mask)
        log_numerator = self._joint_likelihood(inputs, tags, mask)

        return torch.sum(log_numerator - log_denominator)

    def viterbi_tags(
        self, logits: torch.Tensor, mask: torch.BoolTensor = None, top_k: int = None
    ) -> Union[List[VITERBI_DECODING], List[List[VITERBI_DECODING]]]:
        """
        Uses viterbi algorithm to find most likely tags for the given inputs.
        If constraints are applied, disallows all other transitions.

        Returns a list of results, of the same size as the batch (one result per batch member)
        Each result is a List of length top_k, containing the top K viterbi decodings
        Each decoding is a tuple  (tag_sequence, viterbi_score)

        For backwards compatibility, if top_k is None, then instead returns a flat list of
        tag sequences (the top tag sequence for each batch item).
        """
        if mask is None:
            mask = torch.ones(*logits.shape[:2], dtype=torch.bool, device=logits.device)

        if top_k is None:
            top_k = 1
            flatten_output = True
        else:
            flatten_output = False

        _, max_seq_length, num_tags = logits.size()

        logits, mask = logits.data, mask.data

        # Augment transitions matrix with start and end transitions
        start_tag = num_tags
        end_tag = num_tags + 1
        transitions = torch.Tensor(num_tags + 2, num_tags + 2).fill_(-10000.0)

        # Apply transition constraints
        constrained_transitions = self.transitions * self._constraint_mask[
            :num_tags, :num_tags
        ] + -10000.0 * (1 - self._constraint_mask[:num_tags, :num_tags])
        transitions[:num_tags, :num_tags] = constrained_transitions.data

        if self.include_start_end_transitions:
            transitions[
                start_tag, :num_tags
            ] = self.start_transitions.detach() * self._constraint_mask[
                start_tag, :num_tags
            ].data + -10000.0 * (
                1 - self._constraint_mask[start_tag, :num_tags].detach()
            )
            transitions[:num_tags, end_tag] = self.end_transitions.detach() * self._constraint_mask[
                :num_tags, end_tag
            ].data + -10000.0 * (1 - self._constraint_mask[:num_tags, end_tag].detach())
        else:
            transitions[start_tag, :num_tags] = -10000.0 * (
                1 - self._constraint_mask[start_tag, :num_tags].detach()
            )
            transitions[:num_tags, end_tag] = -10000.0 * (
                1 - self._constraint_mask[:num_tags, end_tag].detach()
            )

        best_paths = []
        # Pad the max sequence length by 2 to account for start_tag + end_tag.
        tag_sequence = torch.Tensor(max_seq_length + 2, num_tags + 2)

        for prediction, prediction_mask in zip(logits, mask):
            mask_indices = prediction_mask.nonzero(as_tuple=False).squeeze()
            masked_prediction = torch.index_select(prediction, 0, mask_indices)
            sequence_length = masked_prediction.shape[0]

            tag_sequence.fill_(-10000.0)
            tag_sequence[0, start_tag] = 0.0
            tag_sequence[1 : (sequence_length + 1), :num_tags] = masked_prediction
            tag_sequence[sequence_length + 1, end_tag] = 0.0

            viterbi_paths, viterbi_scores = _viterbi_decode(
                tag_sequence=tag_sequence[: (sequence_length + 2)],
                transition_matrix=transitions,
                top_k=top_k,
            )
            top_k_paths = []
            for viterbi_path, viterbi_score in zip(viterbi_paths, viterbi_scores):
                # Get rid of START and END sentinels and append.
                viterbi_path = viterbi_path[1:-1]
                top_k_paths.append((viterbi_path, viterbi_score))
            best_paths.append(top_k_paths)

        if flatten_output:
            return [top_k_paths[0] for top_k_paths in best_paths]

        return best_paths

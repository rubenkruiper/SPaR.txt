import torch
from allennlp.common.checks import ConfigurationError
from typing import Optional


def flatten_and_batch_shift_index_mask(index_mask: torch.Tensor, sequence_length: int) -> torch.Tensor:
    """

    """
    # Shape: (batch_size, span_size)
    indices_into_sequences = torch.nonzero(torch.flatten(index_mask))

    if torch.max(index_mask) >= (sequence_length * index_mask.size(0)) or torch.min(index_mask) < 0:
        raise ConfigurationError(
            f"All elements in indices should be in range (0, {sequence_length - 1})"
        )

    # (batch_size, sequence_length)
    padded_indices = []
    for i in range(index_mask.size(0)):
        padding_vector = get_padding_vector(sequence_length, get_device_of(index_mask)) * (i+1)
        padded_index_mask = padding_vector.where(~index_mask[i].bool(), indices_into_sequences[i])
        padded_indices.append(padded_index_mask)

    # (batch_size * sequence_length)
    padded_indices = torch.stack(padded_indices).view(-1)
    return padded_indices


def batched_index_select(
    target: torch.Tensor,
    indices_mask: torch.LongTensor,
) -> torch.Tensor:
    """

    """

    # convert padded indices back to indices (index_into_batch, indices_into_sequence)
    # flattened_indices = torch.flatten(torch.nonzero(torch.flatten(indices_mask)))

    flattened_indices = flatten_and_batch_shift_index_mask(indices_mask, target.size(1))

    # Shape: (batch_size * sequence_length, embedding_size)
    flattened_target = target.view(-1, target.size(-1))

    # Shape: (batch_size * d_1 * ... * d_n, embedding_size)
    flattened_selected = flattened_target.index_select(0, flattened_indices)
    selected_shape = list(indices_mask.size()) + [target.size(-1)] # ToDo change
    # Shape: (batch_size, d_1, ..., d_n, embedding_size)
    selected_targets = flattened_selected.view(*selected_shape)
    return selected_targets


def get_padding_vector(size: int, device: int) -> torch.Tensor:
    """
    Returns a vector of sequence length with indices for [SEP]. The CUDA implementation
    is meant to avoid copy data from CPU to GPU.
    """
    if device > -1:
        # ToDo - try using [CLS] as padding (index 0),
        #  also try padding every sequence with at least one [PAD] token?
        return torch.LongTensor(size).fill_(size-1)
    else:
        return torch.zeros(size, dtype=torch.long).add(size-1)


def get_device_of(tensor: torch.Tensor) -> int:
    """
    Returns the device of the tensor.
    """
    if not tensor.is_cuda:
        return -1
    else:
        return tensor.get_device()

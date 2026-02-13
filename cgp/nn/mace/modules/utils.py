###########################################################################################
# Utilities
# Authors: Ilyes Batatia, Gregor Simm and David Kovacs
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

from typing import Tuple

import torch
import torch.nn
import torch.utils.data




def adjacency_matrix(positions: torch.Tensor, r_max: torch.Tensor) -> torch.Tensor:
    # Calculate pairwise distances
    dist_matrix = torch.cdist(positions, positions)
    # Threshold distances based on cutoff distance
    adjacency = (dist_matrix <= r_max).to(torch.int64)
    # Ensure the diagonal elements are zero
    adjacency.fill_diagonal_(0)
    # Convert sparse adjacency matrix to dense
    edge_index = torch.nonzero(adjacency).t()
    return edge_index


def get_edge_vectors_and_lengths(
    positions: torch.Tensor,  # [n_nodes, 3]
    edge_index: torch.Tensor,  # [2, n_edges]
    # shifts: torch.Tensor,  # [n_edges, 3]
    normalize: bool = False,
    eps: float = 1e-9,
) -> Tuple[torch.Tensor, torch.Tensor]:
    sender = edge_index[0]
    receiver = edge_index[1]
    vectors = positions[receiver] - positions[sender]  # + shifts  # [n_edges, 3]
    lengths = torch.linalg.norm(vectors, dim=-1, keepdim=True)  # [n_edges, 1]
    if normalize:
        vectors_normed = vectors / (lengths + eps)
        return vectors_normed, lengths

    return vectors, lengths

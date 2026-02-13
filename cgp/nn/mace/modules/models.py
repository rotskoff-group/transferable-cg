###########################################################################################
# Implementation of MACE models and other models based E(3)-Equivariant MPNNs
# Authors: Ilyes Batatia, Gregor Simm
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

from typing import Any, Callable, Dict, List, Optional, Type, Union, Tuple
import numpy as np
import torch
import cuequivariance as cue
import cuequivariance_torch as cuet
from e3nn import o3
from e3nn.util.jit import compile_mode
from .utils import adjacency_matrix as adj_matrix_wrapper
from .cue_utils import CuEquivarianceConfig
from cgp.cg import all_pairwise_types

from cgp.nn.mace import modules

from .blocks import (
    EquivariantProductBasisBlock,
    LinearNodeEmbeddingBlock,
    LinearReadoutBlock,
    NonLinearReadoutBlock,
    RadialEmbeddingBlock
)
from .utils import (
    get_edge_vectors_and_lengths
)
from .scatter import scatter_sum

# pylint: disable=C0302

@compile_mode("script")
class MACE(torch.nn.Module):
    """
    Arguments:
        r_max: Distance cutoff (in Ang)
            default: 5.0
        num_bessel: Number of radial basis functions
            default: 8
        num_polynomial_cutoff: Number of basis functions for smooth cutoff
            default: 5
        max_ell: Highest ell of spherical harmonics
            default: 3
        interaction_cls: Interaction block class
            default: RealAgnosticResidualInteractionBlock
            options = RealAgnosticResidualInteractionBlock
                      RealAgnosticAttResidualInteractionBlock
                      RealAgnosticInteractionBlock
        interaction_cls_first: Name of first interaction block
            default: RealAgnosticResidualInteractionBlock
            options = RealAgnosticResidualInteractionBlock
                      RealAgnosticInteractionBlock
        num_interactions: Number of interaction layers
            default: 2
        hidden_irreps: Irreps for hidden node states
            default: 128x0e + 128x1o
        MLP_irreps: Hidden irreps of the MLP in last readout
            default: 16x0e
        avg_num_neighbors: Normalization factor for the message
        correlation: Correlation order at each layer
        gate: Non-linearity for last readout
            options: silu, tanh abs, None
        radial_MLP: List of hidden units for radial MLP
        radial_type: Type of radial basis functions
            choices: bessel, gaussian

    """
    def __init__(
        self,
        nn_cutoff: float = 20.0,
        num_bessel: int = 8, 
        num_polynomial_cutoff: int = 5, 
        max_ell: int = 3, 
        interaction_cls: str = "RealAgnosticResidualInteractionBlock", 
        interaction_cls_first: str = "RealAgnosticResidualInteractionBlock",
        num_interactions: int = 2, 
        hidden_irreps: str = "128x0e + 128x1o",
        MLP_irreps: str = "16x0e", 
        avg_num_neighbors: float = 1.0,
        correlation: Union[int, List[int]] = 3, 
        gate: str = "silu", 
        radial_MLP: Optional[List[int]] = [],
        radial_type: Optional[str] = "bessel",
        max_z: int = None,
        use_cueq: bool = True,
    ):
        super().__init__()
        if use_cueq:
            self.cueq_config = CuEquivarianceConfig(
                    enabled=True,
                    layout="ir_mul",
                    group="O3_e3nn",
                    optimize_all=True,
                    conv_fusion=True, #TODO: True if CUDA is available
                )
        else:
            self.cueq_config = None
        if max_z is None:
            self.max_z = len(all_pairwise_types)
        else:
            self.max_z = max_z

        self.register_buffer(
            "r_max", torch.tensor(nn_cutoff, dtype=torch.get_default_dtype())
        )
        self.register_buffer(
            "num_interactions", torch.tensor(num_interactions, dtype=torch.int64)
        )
        if isinstance(correlation, int):
            correlation = [correlation] * num_interactions
        gate = modules.gate_dict[gate]
  
        # Embedding
        hidden_irreps = o3.Irreps(hidden_irreps)
        MLP_irreps = o3.Irreps(MLP_irreps)
        node_attr_irreps = o3.Irreps([(self.max_z , (0, 1))]) # max_z scaler values 
        node_feats_irreps = o3.Irreps([(hidden_irreps.count(o3.Irrep(0, 1)), (0, 1))])
        self.node_embedding = LinearNodeEmbeddingBlock(
            irreps_in=node_attr_irreps, 
            irreps_out=node_feats_irreps, 
            cueq_config=self.cueq_config
        )
        self.radial_embedding = RadialEmbeddingBlock(
            r_max=nn_cutoff,
            num_bessel=num_bessel,
            num_polynomial_cutoff=num_polynomial_cutoff,
            radial_type=radial_type,
        )
        edge_feats_irreps = o3.Irreps(f"{self.radial_embedding.out_dim}x0e")

        sh_irreps = o3.Irreps.spherical_harmonics(max_ell)
        num_features = hidden_irreps.count(o3.Irrep(0, 1))
        interaction_irreps = (sh_irreps * num_features).sort()[0].simplify()
        
        if use_cueq:
            self.spherical_harmonics = cuet.SphericalHarmonics(
                sh_irreps.ls, normalize=True
            )
        else:
            self.spherical_harmonics = o3.SphericalHarmonics( 
                sh_irreps, normalize=True, normalization="component"
            )
        
        if radial_MLP is None:
            radial_MLP = [64, 64, 64]
        # Interactions and readout
        interaction_cls_first = modules.interaction_classes[interaction_cls_first]
        interaction_cls = modules.interaction_classes[interaction_cls]

        inter = interaction_cls_first(
            node_attrs_irreps=node_attr_irreps,
            node_feats_irreps=node_feats_irreps,
            edge_attrs_irreps=sh_irreps,
            edge_feats_irreps=edge_feats_irreps,
            target_irreps=interaction_irreps,
            hidden_irreps=hidden_irreps,
            avg_num_neighbors=avg_num_neighbors,
            radial_MLP=radial_MLP,
            cueq_config=self.cueq_config,
        )
        self.interactions = torch.nn.ModuleList([inter])

        # Use the appropriate self connection at the first layer for proper E0
        use_sc_first = False
        if "Residual" in str(interaction_cls_first):
            use_sc_first = True

        node_feats_irreps_out = inter.target_irreps
        prod = EquivariantProductBasisBlock(
            node_feats_irreps=node_feats_irreps_out,
            target_irreps=hidden_irreps,
            correlation=correlation[0],
            num_elements=self.max_z,
            use_sc=use_sc_first,
            cueq_config=self.cueq_config,
        )
        self.products = torch.nn.ModuleList([prod])

        self.readouts = torch.nn.ModuleList()
        self.readouts.append(LinearReadoutBlock(hidden_irreps, cueq_config=self.cueq_config))

        for i in range(num_interactions - 1):
            if i == num_interactions - 2:
                hidden_irreps_out = str(
                    hidden_irreps[0]
                )  # Select only scalars for last layer
            else:
                hidden_irreps_out = hidden_irreps
            inter = interaction_cls(
                node_attrs_irreps=node_attr_irreps,
                node_feats_irreps=hidden_irreps,
                edge_attrs_irreps=sh_irreps,
                edge_feats_irreps=edge_feats_irreps,
                target_irreps=interaction_irreps,
                hidden_irreps=hidden_irreps_out,
                avg_num_neighbors=avg_num_neighbors,
                radial_MLP=radial_MLP,
                cueq_config=self.cueq_config,
            )
            self.interactions.append(inter)
            prod = EquivariantProductBasisBlock(
                node_feats_irreps=interaction_irreps,
                target_irreps=hidden_irreps_out,
                correlation=correlation[i + 1],
                num_elements=self.max_z, 
                use_sc=True,
                cueq_config=self.cueq_config,
            )
            self.products.append(prod)
            if i == num_interactions - 2:
                self.readouts.append(
                    NonLinearReadoutBlock(hidden_irreps_out, MLP_irreps, 
                                          cueq_config=self.cueq_config, gate = gate)
                )
            else:
                self.readouts.append(LinearReadoutBlock(hidden_irreps, cueq_config=self.cueq_config))
    
    def adjacency_matrix(self, positions : torch.Tensor) -> torch.Tensor:
        return adj_matrix_wrapper(positions, self.r_max)

    def create_one_hot_encoding(self, labels : torch.Tensor):
        """
        labels: list of tuples (atom type, residue type)
        max_zs: Maximum number of residue types and atom types (max residue, max atom)
        """
        # index = [int((labels[0] * max_zs[1]) + labels[1]) for labels in labels]
        # return torch.nn.functional.one_hot(torch.tensor(index), max_zs[0] *
        #             max_zs[1]).type(torch.float32).to(labels.device)
        return torch.nn.functional.one_hot(labels, self.max_z).type(torch.float32).to(labels.device)

    def forward(
        self,
        positions: torch.Tensor,
        features : torch.Tensor,
        edge_indices : torch.Tensor = torch.zeros(0, dtype=torch.int64),
        atom_sctr_indices : torch.Tensor = torch.zeros(0, dtype=torch.int64),
        batch_size : Optional[int] = None,
        update_edge_indices : bool = False, 
    ) -> torch.Tensor:
        node_attrs = self.create_one_hot_encoding(features)

        # Embeddings
        node_feats = self.node_embedding(node_attrs)
        if update_edge_indices:
            edge_indices = edge_indices[(torch.norm(positions[edge_indices[:, 0]] - positions[edge_indices[:, 1]], dim=-1)) < self.r_max]

        edge_index = edge_indices.T
        vectors, lengths = get_edge_vectors_and_lengths(
            positions=positions,
            edge_index=edge_index,
            #shifts=data["shifts"],
        )
        edge_attrs = self.spherical_harmonics(vectors)
        edge_feats = self.radial_embedding(lengths)

        # Interactions
        energies = []
        node_energies_list = []
        node_feats_list = []
        for interaction, product, readout in zip(
            self.interactions, self.products, self.readouts
        ):
            node_feats, sc = interaction(
                node_attrs=node_attrs,
                node_feats=node_feats,
                edge_attrs=edge_attrs,
                edge_feats=edge_feats,
                edge_index=edge_index,
            )
            node_feats = product(
                node_feats=node_feats,
                sc=sc,
                node_attrs=node_attrs,
            )
            node_feats_list.append(node_feats)
            node_energies = readout(node_feats).squeeze(-1)  # [n_nodes, ]
            energy = torch.zeros(batch_size, dtype = positions.dtype).to(positions.device)
            # energy = scatter_sum(node_energies, atom_sctr_indices)  # [n_graphs,]
            energy.scatter_add_(0, atom_sctr_indices, node_energies)
            energies.append(energy)
            node_energies_list.append(node_energies)

        # Sum over energy contributions
        contributions = torch.stack(energies, dim=-1)
        total_energy = torch.sum(contributions, dim=-1)  # [n_graphs, ]

        return total_energy

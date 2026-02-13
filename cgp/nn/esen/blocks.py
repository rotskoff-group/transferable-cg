"""
Copyright (c) Meta, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import os
import copy
import torch
import torch.nn as nn

from .utils.rotation import (
    init_edge_rot_mat,
    rotation_to_wigner,
)
from .utils.so3 import (
    CoefficientMapping,
    SO3_Grid,
)
from .utils.embedding import EdgeDegreeEmbedding
from .utils.layer_norm import (
    get_normalization_layer,
)
from .utils.radial import EnvelopedBesselBasis, GaussianSmearing, PolynomialEnvelope
from .utils.so3_layers import SO3_Linear

from .utils.activation import GateActivation, SeparableS2Activation
from .utils.so2_layers import SO2_Convolution


class eSEN_Backbone(nn.Module):
    def __init__(
        self,
        max_num_elements: int = 100,
        sphere_channels: int = 128,
        lmax: int = 2,
        mmax: int = 2,
        grid_resolution: int | None = None,
        cutoff: float = 5.0,
        edge_channels: int = 128,
        distance_function: str = "gaussian",
        num_distance_basis: int = 512,
        direct_forces: bool = True,
        # escnmd specific
        num_layers: int = 2,
        hidden_channels: int = 128,
        norm_type: str = "rms_norm_sh",
        act_type: str = "s2",
        mlp_type: str = "spectral",
        use_envelope: bool = True,
        activation_checkpointing: bool = False,
    ):
        super().__init__()

        self.max_num_elements = max_num_elements
        self.lmax = lmax
        self.mmax = mmax
        self.sphere_channels = sphere_channels
        self.grid_resolution = grid_resolution

        self.direct_forces = direct_forces

        self.activation_checkpointing = activation_checkpointing

        self.mlp_type = mlp_type
        self.use_envelope = use_envelope

        # rotation utils
        Jd_list = torch.load(os.path.join(os.path.dirname(__file__), "Jd.pt"))
        for ell in range(self.lmax + 1):
            self.register_buffer(f"Jd_{ell}", Jd_list[ell])
        self.sph_feature_size = int((self.lmax + 1) ** 2)
        self.mappingReduced = CoefficientMapping(self.lmax, self.mmax)

        # lmax_lmax for node, lmax_mmax for edge
        self.SO3_grid = nn.ModuleDict()
        self.SO3_grid["lmax_lmax"] = SO3_Grid(
            self.lmax, self.lmax, resolution=grid_resolution, rescale=True
        )
        self.SO3_grid["lmax_mmax"] = SO3_Grid(
            self.lmax, self.mmax, resolution=grid_resolution, rescale=True
        )

        # atom embedding
        self.sphere_embedding = nn.Embedding(
            self.max_num_elements, self.sphere_channels
        )

        # edge distance embedding
        self.cutoff = cutoff
        self.edge_channels = edge_channels
        self.distance_function = distance_function
        self.num_distance_basis = num_distance_basis

        if self.distance_function == "gaussian":
            self.distance_expansion = GaussianSmearing(
                0.0,
                self.cutoff,
                self.num_distance_basis,
                2.0,
            )
        elif self.distance_function == "bessel":
            self.distance_expansion = EnvelopedBesselBasis(
                num_radial=self.num_distance_basis,
                cutoff=cutoff,
            )
            self.distance_expansion.offset = [self.cutoff]
            self.distance_expansion.num_output = self.num_distance_basis
        else:
            raise ValueError("Unknown distance function")

        # equivariant initial embedding
        self.source_embedding = nn.Embedding(self.max_num_elements, self.edge_channels)
        self.target_embedding = nn.Embedding(self.max_num_elements, self.edge_channels)
        nn.init.uniform_(self.source_embedding.weight.data, -0.001, 0.001)
        nn.init.uniform_(self.target_embedding.weight.data, -0.001, 0.001)

        self.edge_channels_list = [
            self.num_distance_basis + 2 * self.edge_channels,
            self.edge_channels,
            self.edge_channels,
        ]

        self.edge_degree_embedding = EdgeDegreeEmbedding(
            sphere_channels=self.sphere_channels,
            lmax=self.lmax,
            mmax=self.mmax,
            max_num_elements=self.max_num_elements,
            edge_channels_list=self.edge_channels_list,
            rescale_factor=5.0,
            cutoff=self.cutoff,
            mappingReduced=self.mappingReduced,
            out_mask=self.SO3_grid["lmax_lmax"].mapping.coefficient_idx(
                self.lmax, self.mmax
            ),
            use_envelope=use_envelope,
        )

        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        self.norm_type = norm_type
        self.act_type = act_type

        # Initialize the blocks for each layer
        self.blocks = nn.ModuleList()
        for _ in range(self.num_layers):
            block = eSEN_Block(
                self.sphere_channels,
                self.hidden_channels,
                self.lmax,
                self.mmax,
                self.mappingReduced,
                self.SO3_grid,
                self.edge_channels_list,
                self.cutoff,
                self.norm_type,
                self.act_type,
                self.mlp_type,
                self.use_envelope,
            )
            self.blocks.append(block)

        self.norm = get_normalization_layer(
            self.norm_type,
            lmax=self.lmax,
            num_channels=self.sphere_channels,
        )

    def get_rotmat_and_wigner(self, edge_distance_vecs):
        edge_rot_mat = init_edge_rot_mat(
            edge_distance_vecs, rot_clip=(not self.direct_forces)
        )

        Jd_buffers = [
            getattr(self, f"Jd_{ell}").type(edge_rot_mat.dtype)
            for ell in range(self.lmax + 1)
        ]

        wigner = rotation_to_wigner(
            edge_rot_mat,
            0,
            self.lmax,
            Jd_buffers,
            rot_clip=(not self.direct_forces),
        )
        wigner_inv = torch.transpose(wigner, 1, 2).contiguous()

        return edge_rot_mat, wigner, wigner_inv

    def forward(self, pos, features, edge_indices, update_edge_indices=False):
        ###############################################################
        # Initialize edges
        ###############################################################
        atomic_numbers = features.long()
        edge_distance_vec = pos[edge_indices[:, 0], :] - pos[edge_indices[:, 1], :]
        edge_distance = torch.norm(edge_distance_vec, dim=-1)

        if update_edge_indices:
            indices = edge_distance < self.cutoff
            edge_indices = edge_indices[indices]
            edge_distance_vec = edge_distance_vec[indices]
            edge_distance = edge_distance[indices]

        edge_indices = edge_indices.T

        _, wigner, wigner_inv = self.get_rotmat_and_wigner(edge_distance_vec)

        ###############################################################
        # Initialize node embeddings
        ###############################################################

        x_message = torch.zeros(
            pos.shape[0],
            self.sph_feature_size,
            self.sphere_channels,
            device=pos.device,
            dtype=pos.dtype,
        )
        x_message[:, 0, :] = self.sphere_embedding(atomic_numbers)

        # edge degree embedding
        edge_distance_embedding = self.distance_expansion(edge_distance)
        source_embedding = self.source_embedding(atomic_numbers[edge_indices[0]])
        target_embedding = self.target_embedding(atomic_numbers[edge_indices[1]])
        x_edge = torch.cat(
            (edge_distance_embedding, source_embedding, target_embedding), dim=1
        )
        x_message = self.edge_degree_embedding(
            x_message,
            x_edge,
            edge_distance,
            edge_indices,
            wigner_inv,
        )

        ###############################################################
        # Update spherical node embeddings
        ###############################################################
        if edge_indices.shape[1] != 0:
            for i in range(self.num_layers):
                if self.activation_checkpointing:
                    x_message = torch.utils.checkpoint.checkpoint(
                        self.blocks[i],
                        x_message,
                        x_edge,
                        edge_distance,
                        edge_indices,
                        wigner,
                        wigner_inv,
                        use_reentrant=False,
                    )
                else:
                    x_message = self.blocks[i](
                        x_message,
                        x_edge,
                        edge_distance,
                        edge_indices,
                        wigner,
                        wigner_inv,
                    )

        # Final layer norm
        x_message = self.norm(x_message)
        return x_message


class MLP_Energy_Head(nn.Module):
    def __init__(self, backbone):
        super().__init__()

        self.sphere_channels = backbone.sphere_channels
        self.hidden_channels = backbone.hidden_channels
        self.energy_block = nn.Sequential(
            nn.Linear(self.sphere_channels, self.hidden_channels, bias=True),
            nn.SiLU(),
            nn.Linear(self.hidden_channels, self.hidden_channels, bias=True),
            nn.SiLU(),
            nn.Linear(self.hidden_channels, 1, bias=True),
        )

    def forward(self, node_embedding, atom_sctr_indices, batch_size):
        node_energy = self.energy_block(node_embedding.narrow(1, 0, 1).squeeze()).view(
            -1, 1, 1
        )

        energy = torch.zeros(
            batch_size,
            device=node_energy.device,
            dtype=node_energy.dtype,
        )

        energy.scatter_add_(0, atom_sctr_indices, node_energy.view(-1))
        return energy


class Linear_Force_Head(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.linear = SO3_Linear(backbone.sphere_channels, 1, lmax=1)

    def forward(self, node_embedding):
        forces = self.linear(node_embedding.narrow(1, 0, 4))
        forces = forces.narrow(1, 1, 3)
        forces = forces.view(-1, 3).contiguous()
        return forces


class Edgewise(torch.nn.Module):
    def __init__(
        self,
        sphere_channels: int,
        hidden_channels: int,
        lmax: int,
        mmax: int,
        edge_channels_list,
        mappingReduced,
        SO3_grid,
        cutoff,
        act_type="gate",
        use_envelope: bool = True,
    ):
        super().__init__()

        self.sphere_channels = sphere_channels
        self.hidden_channels = hidden_channels
        self.lmax = lmax
        self.mmax = mmax

        self.mappingReduced = mappingReduced
        self.SO3_grid = SO3_grid
        self.edge_channels_list = copy.deepcopy(edge_channels_list)
        self.act_type = act_type

        if self.act_type == "gate":
            self.act = GateActivation(
                lmax=self.lmax, mmax=self.mmax, num_channels=self.hidden_channels
            )
            extra_m0_output_channels = self.lmax * self.hidden_channels
        elif self.act_type == "s2":
            self.act = SeparableS2Activation(
                lmax=self.lmax, mmax=self.mmax, SO3_grid=self.SO3_grid
            )
            extra_m0_output_channels = self.hidden_channels
        else:
            raise ValueError(f"Unknown activation type {self.act_type}")

        self.so2_conv_1 = SO2_Convolution(
            2 * self.sphere_channels,
            self.hidden_channels,
            self.lmax,
            self.mmax,
            self.mappingReduced,
            internal_weights=False,
            edge_channels_list=self.edge_channels_list,
            extra_m0_output_channels=extra_m0_output_channels,
        )

        self.so2_conv_2 = SO2_Convolution(
            self.hidden_channels,
            self.sphere_channels,
            self.lmax,
            self.mmax,
            self.mappingReduced,
            internal_weights=True,
            edge_channels_list=None,
            extra_m0_output_channels=None,
        )

        self.use_envelope = use_envelope
        if self.use_envelope:
            self.cutoff = cutoff
            self.envelope = PolynomialEnvelope(exponent=5)

        self.out_mask = self.SO3_grid["lmax_lmax"].mapping.coefficient_idx(
            self.lmax, self.mmax
        )

    def forward(
        self,
        x,
        x_edge,
        edge_distance,
        edge_index,
        wigner,
        wigner_inv,
        node_offset: int = 0,
    ):
        x_source = x[edge_index[0]]
        x_target = x[edge_index[1]]
        x_message = torch.cat((x_source, x_target), dim=2)

        # Rotate the irreps to align with the edge
        x_message = torch.bmm(wigner[:, self.out_mask, :], x_message)

        # SO2 convolution
        x_message, x_0_gating = self.so2_conv_1(x_message, x_edge)
        x_message = self.act(x_0_gating, x_message)
        x_message = self.so2_conv_2(x_message, x_edge)

        # envelope
        if self.use_envelope:
            dist_scaled = edge_distance / self.cutoff
            env = self.envelope(dist_scaled)
            x_message = x_message * env.view(-1, 1, 1)

        # Rotate back the irreps
        x_message = torch.bmm(wigner_inv[:, :, self.out_mask], x_message)

        # Compute the sum of the incoming neighboring messages for each target node
        new_embedding = torch.zeros(
            (x.shape[0],) + x_message.shape[1:],
            dtype=x_message.dtype,
            device=x_message.device,
        )

        new_embedding.index_add_(0, edge_index[1] - node_offset, x_message)

        return new_embedding


class SpectralAtomwise(torch.nn.Module):
    def __init__(
        self,
        sphere_channels: int,
        hidden_channels: int,
        lmax: int,
        mmax: int,
        SO3_grid,
    ):
        super().__init__()
        self.sphere_channels = sphere_channels
        self.hidden_channels = hidden_channels
        self.lmax = lmax
        self.mmax = mmax
        self.SO3_grid = SO3_grid

        self.scalar_mlp = nn.Sequential(
            nn.Linear(
                self.sphere_channels,
                self.lmax * self.hidden_channels,
                bias=True,
            ),
            nn.SiLU(),
        )

        self.so3_linear_1 = SO3_Linear(
            self.sphere_channels, self.hidden_channels, lmax=self.lmax
        )
        self.act = GateActivation(
            lmax=self.lmax, mmax=self.lmax, num_channels=self.hidden_channels
        )
        self.so3_linear_2 = SO3_Linear(
            self.hidden_channels, self.sphere_channels, lmax=self.lmax
        )

    def forward(self, x):
        gating_scalars = self.scalar_mlp(x.narrow(1, 0, 1))
        x = self.so3_linear_1(x)
        x = self.act(gating_scalars, x)
        return self.so3_linear_2(x)


class GridAtomwise(torch.nn.Module):
    def __init__(
        self,
        sphere_channels: int,
        hidden_channels: int,
        lmax: int,
        mmax: int,
        SO3_grid,
    ):
        super().__init__()
        self.sphere_channels = sphere_channels
        self.hidden_channels = hidden_channels
        self.lmax = lmax
        self.mmax = mmax
        self.SO3_grid = SO3_grid

        self.grid_mlp = nn.Sequential(
            nn.Linear(self.sphere_channels, self.hidden_channels, bias=False),
            nn.SiLU(),
            nn.Linear(self.hidden_channels, self.hidden_channels, bias=False),
            nn.SiLU(),
            nn.Linear(self.hidden_channels, self.sphere_channels, bias=False),
        )

    def forward(self, x):
        # Project to grid
        x_grid = self.SO3_grid["lmax_lmax"].to_grid(x, self.lmax, self.lmax)
        # Perform point-wise operations
        x_grid = self.grid_mlp(x_grid)
        # Project back to spherical harmonic coefficients
        return self.SO3_grid["lmax_lmax"].from_grid(x_grid, self.lmax, self.lmax)


class eSEN_Block(torch.nn.Module):
    def __init__(
        self,
        sphere_channels: int,
        hidden_channels: int,
        lmax: int,
        mmax: int,
        mappingReduced,
        SO3_grid,
        edge_channels_list: list[int],
        cutoff: float,
        norm_type: str,
        act_type: str,
        mlp_type: str,
        use_envelope: bool,
    ) -> None:
        super().__init__()
        self.sphere_channels = sphere_channels
        self.hidden_channels = hidden_channels
        self.lmax = lmax
        self.mmax = mmax

        self.norm_1 = get_normalization_layer(
            norm_type, lmax=self.lmax, num_channels=sphere_channels
        )

        self.use_envelope = use_envelope

        self.edge_wise = Edgewise(
            sphere_channels=sphere_channels,
            hidden_channels=hidden_channels,
            lmax=lmax,
            mmax=mmax,
            edge_channels_list=edge_channels_list,
            mappingReduced=mappingReduced,
            SO3_grid=SO3_grid,
            cutoff=cutoff,
            act_type=act_type,
            use_envelope=use_envelope,
        )

        self.norm_2 = get_normalization_layer(
            norm_type, lmax=self.lmax, num_channels=sphere_channels
        )

        if mlp_type == "spectral":
            self.atom_wise = SpectralAtomwise(
                sphere_channels=sphere_channels,
                hidden_channels=hidden_channels,
                lmax=lmax,
                mmax=mmax,
                SO3_grid=SO3_grid,
            )
        elif mlp_type == "grid":
            self.atom_wise = GridAtomwise(
                sphere_channels=sphere_channels,
                hidden_channels=hidden_channels,
                lmax=lmax,
                mmax=mmax,
                SO3_grid=SO3_grid,
            )
        else:
            raise ValueError(f"Unknown MLP type {mlp_type}")

    def forward(
        self,
        x,
        x_edge,
        edge_distance,
        edge_index,
        wigner,
        wigner_inv,
        node_offset: int = 0,
    ):
        x_res = x
        x = self.norm_1(x)

        x = self.edge_wise(
            x,
            x_edge,
            edge_distance,
            edge_index,
            wigner,
            wigner_inv,
            node_offset,
        )
        x = x + x_res

        x_res = x
        x = self.norm_2(x)

        x = self.atom_wise(x)
        return x + x_res

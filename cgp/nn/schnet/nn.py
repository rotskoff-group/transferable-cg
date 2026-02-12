import torch
import torch.nn as nn
import os
import numpy as np
import math

class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, max_len, dropout_p=0.0):
        """Args:
            dim_model: Dimension of the model
            max_len: Maximum length of the sequence
            dropout_p: Dropout probability - use 0
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout_p)

        pos_encoding = torch.zeros(max_len, dim_model)
        positions_list = torch.arange(0, max_len).unsqueeze(-1).float()

        k = torch.arange(0, dim_model//2, 1).float()
        log_w_k = (-(2 * k)/dim_model) * math.log(10000.0)
        w_k = torch.exp(log_w_k)

        pos_encoding[:, 0::2] = torch.sin(positions_list * w_k)
        pos_encoding[:, 1::2] = torch.cos(positions_list * w_k)
        self.register_buffer("pos_encoding", pos_encoding)

    def forward(self, atom_sctr_indices):
        """Args:
            token_embedding: (batch_size, seq_len, dim_model)
        """
        _, counts = torch.unique_consecutive(atom_sctr_indices, return_counts=True)
        total = torch.sum(counts)
        offset = torch.cat([torch.tensor([0]).to(atom_sctr_indices.device), 
                            torch.cumsum(counts, 0)[:-1]])
        indices = torch.arange(total).to(atom_sctr_indices.device) - offset.repeat_interleave(counts)
        return self.dropout(self.pos_encoding[indices, :])

class GaussianBasis(nn.Module):
    def __init__(self, n_gaussians, cutoff):
        super().__init__()
        self.register_buffer('centers',
                             torch.linspace(0.0, cutoff, n_gaussians))
        self.variance = (self.centers[1] - self.centers[0]) ** 2

    def forward(self, distances):
        distances = (distances.unsqueeze(-1) - self.centers) ** 2
        return torch.exp(-distances / (2 * self.variance))
    
class ExpNormalBasis(nn.Module):
    def __init__(self, n_basis, cutoff):
        super().__init__()
        self.cutoff = torch.tensor(cutoff)
        self.register_buffer('centers',
                             torch.linspace(torch.exp(- self.cutoff), 1.0, n_basis))
        self.beta = (2  * (1 - torch.exp(-self.cutoff)) / n_basis) ** (-2)
        self.alpha = 5.0 / self.cutoff
    
    def smooth_cutoff(self, distances):
        distances = distances / self.cutoff
        x = 1 - (6 * (distances ** 5)) + (15 * (distances ** 4)) - (10 * (distances ** 3))
        return x
    
    def forward(self, distances):
        distances = distances.unsqueeze(-1)
        smooth_cutoff = self.smooth_cutoff(distances)
        expanded = torch.exp(-self.beta * (torch.exp(-self.alpha * distances) - self.centers) ** 2)
        return expanded  * smooth_cutoff
    

class CosExpNormalBasis(nn.Module):
    def __init__(self, n_basis, cutoff):
        super().__init__()
        self.cutoff = torch.tensor(cutoff)
        self.register_buffer('centers',
                             torch.linspace(torch.exp(- self.cutoff), 1.0, n_basis))
        self.beta = (2  * (1 - torch.exp(-self.cutoff)) / n_basis) ** (-2)
        self.alpha = 5.0 / self.cutoff
    
    def smooth_cutoff(self, distances):
        distances = torch.pi * ((2 * distances / self.cutoff) + 1)
        x = (0.5 * torch.cos(distances)) + 0.5
        return x
    
    def forward(self, distances):
        distances = distances.unsqueeze(-1)
        smooth_cutoff = self.smooth_cutoff(distances)
        expanded = torch.exp(-self.beta * (torch.exp(-self.alpha * distances) - self.centers) ** 2)
        return expanded  * smooth_cutoff


class SchNetInteraction(nn.Module):
    def __init__(self, n_atom_basis, n_radial_basis, n_filters, include_bond_embedding=False):
        super().__init__()
        self.filter_network_1 = nn.Linear(n_radial_basis, n_filters)
        self.filter_network_2 = nn.Linear(n_filters, n_filters)

        self.in2f = nn.Linear(n_atom_basis, n_filters)
        self.f2out = nn.Linear(n_filters, n_atom_basis)

        self.activation = nn.Tanh()
        self.dense = nn.Linear(n_atom_basis, n_atom_basis)

        self.include_bond_embedding = include_bond_embedding
        if self.include_bond_embedding:
            self.embedding = nn.Embedding(4, n_filters)
            nn.init.normal_(self.embedding.weight, mean=0.0, std=1e-3)

    def embed_bond(self, nonbonded_edge_indices):
        edges = torch.abs(nonbonded_edge_indices[:, 0] - nonbonded_edge_indices[:, 1]) - 1
        edges[edges > 2] = 3
        return self.embedding(edges)


    def forward(self, x, f_ij, nonbonded_edge_indices):
        """
        Arguments:
            x: A torch tensor of (n_configs, n_atoms, n_atom_basis) representing
               the embedding of atoms
            f_ij: A torch tensor of (n_configs, n_bonds, n_gaussian_basis)
                representing the gaussian expansion. Diagonals are excluded resulting
                in (n_atoms, n_atoms - 1)
        """
        w = self.activation(self.filter_network_1(f_ij))
        if self.include_bond_embedding:
            bond_embedding = self.embed_bond(nonbonded_edge_indices)   
            w = w + bond_embedding
        w = self.filter_network_2(w) 
        
        nonbonded_edge_indices = nonbonded_edge_indices.to(torch.int64)
        y = self.in2f(x)

        output = torch.zeros(x.shape[0], w.shape[1]).to(x.device)
        w = y[nonbonded_edge_indices[:, 1], :] * w # combining atom embedding and bond information
        output.scatter_add_(0, nonbonded_edge_indices[:, 0].unsqueeze(-1).expand(-1, w.shape[1]), w)
        output = self.activation(self.f2out(output))
        output = self.dense(output)
        return output
    
class SchNet(nn.Module):
    def __init__(self, 
                 nn_cutoff = 20.0,
                 n_atom_basis=128, 
                 n_filters=128, 
                 n_interaction_blocks=2,
                 n_radial_basis=25, 
                 radial_basis_type='GaussianBasis', # GaussianBasis or ExpNormalBasis
                 max_z=67,
                 include_positional_encoding=False,
                 include_bond_embedding=False):
        """
        Arguments:
            n_atom_basis: Basis size of atom embedding
            n_filters: Number of filters to use for position convolution
            n_interaction_blocks: Number of SchNet interaction blocks to use
            n_gaussians: Number of gaussian basis functions to use
            cutoff: Cutoff in Å
            max_z: Max number of atomic_numbers that will be used
        """
        super().__init__()
        self.cutoff = nn_cutoff
        self.n_atom_basis = n_atom_basis

        self.include_positional_encoding = include_positional_encoding

        self.embedding = nn.Embedding(max_z, n_atom_basis)

        if include_positional_encoding:
            self.positional_encoding = PositionalEncoding(n_atom_basis, max_len=5000, dropout_p=0.1)
        else:
            self.positional_encoding = None

        self.radial_basis = globals()[radial_basis_type](n_radial_basis, nn_cutoff)

        self.interactions = nn.ModuleList([SchNetInteraction(n_atom_basis=n_atom_basis,
                                                             n_radial_basis=n_radial_basis,
                                                             n_filters=n_filters,
                                                             include_bond_embedding=include_bond_embedding) 
                                                                for _ in range(n_interaction_blocks)])
        self.energy_1 = nn.Linear(n_atom_basis, n_atom_basis)
        self.energy_2 = nn.Linear(n_atom_basis, 1)
        self.activation = nn.Tanh()

        
    def embed(self, atom_features, atom_sctr_indices):
        x = self.embedding(atom_features.long())
            
        if self.include_positional_encoding:
            x = x + self.positional_encoding(atom_sctr_indices)

        return x

    def forward(self, atomic_positions, features,
                edge_indices, atom_sctr_indices,
                batch_size, update_edge_indices=False):
        """
        Arguments:
            atomic_positions: A tensor of size (n_configs, n_atoms, 3)
            atomic_numbers: A tensor of size (n_configs, n_atoms)
            Nonbonded edge indices must include forward and reverse pairs
        """
        x = self.embed(features, atom_sctr_indices)

        pair_dist = torch.linalg.norm(atomic_positions[edge_indices[:, 0], :] -
                                        atomic_positions[edge_indices[:, 1], :],
                                        dim=-1)
        if update_edge_indices:
            indices = pair_dist < self.cutoff
            edge_indices = edge_indices[indices]
            pair_dist = pair_dist[indices]
        # dist_centered_squared = (pair_dist.unsqueeze(-1) - self.centers) ** 2
        # f_ij = torch.exp(-(0.5 / self.variance) * dist_centered_squared)

        f_ij = self.radial_basis(pair_dist)

        # Compute Interaction Block
        for interactions in self.interactions:
            v = interactions(x, f_ij, edge_indices)
            x = x + v

        # Compute Energy
        x = self.activation(self.energy_1(x))
        x = self.energy_2(x).reshape(-1, 1)
        x = x.sum(-1)
        energy = torch.zeros(batch_size, dtype=x.dtype).to(x.device)
        energy.scatter_add_(0, atom_sctr_indices, x.squeeze(-1))
        return energy
    

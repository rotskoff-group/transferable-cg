import torch
import torch.nn as nn
from cgp.cg import all_bond_types, all_angle_types, all_dihedral_angle_types, all_pairwise_types
import os
import numpy as np
import math

class ForceField(nn.Module):
    def __init__(self,
                 nn_cutoff = None,
                 return_forces = False,
                 has_bonded=True,
                 has_angles=True,
                 has_vdw=True,
                 has_wca=True,
                 has_coulomb=True,
                 has_dihedrals=True,
                 has_calvados_vdw=False,
                 init_bonded_terms=None,
                 init_angle_terms=None,
                 init_dihedral_terms=None,
                 init_nonbonded_terms=None,
                 init_calvados_vdw_terms=None,
                 num_dihedral_basis=10,
                 dielectric_constant=1.0,
                 vdw_cutoff_distance=20.0
                 ):
        super().__init__()
        folder_path = os.path.dirname(__file__)
        if init_bonded_terms:
            init_bonded_terms = f"{folder_path}/init_terms/init_bonded_terms.npy"
            init_bonded_terms = np.load(
                init_bonded_terms, allow_pickle=True).item()
            self.bonds_mean = nn.Parameter(torch.tensor(init_bonded_terms["bonds_mean"]).float())
            self.bonds_k = nn.Parameter(torch.tensor(init_bonded_terms["bonds_k"]).float()) 
        else:
            num_bonded_terms = len(all_bond_types)
            self.bonds_mean = nn.Parameter(torch.randn(num_bonded_terms)/1000)
            self.bonds_k = nn.Parameter(torch.randn(num_bonded_terms)/1000)

        if init_angle_terms:
            init_angle_terms = f"{folder_path}/init_terms/init_angle_terms.npy"
            init_angle_terms = np.load(
                init_angle_terms, allow_pickle=True).item()
            self.angles_mean = nn.Parameter(torch.tensor(init_angle_terms["angles_mean"]).float())
            self.angles_k = nn.Parameter(torch.tensor(init_angle_terms["angles_k"]).float())
        else:
            num_angle_terms = len(all_angle_types)
            self.angles_mean = nn.Parameter(torch.randn(num_angle_terms)/1000)
            self.angles_k = nn.Parameter(torch.randn(num_angle_terms)/1000)

        self.num_dihedral_basis = num_dihedral_basis
        dihedral_period = torch.arange(1, self.num_dihedral_basis + 1).float()
        if init_dihedral_terms:
            init_dihedral_terms = f"{folder_path}/init_terms/init_dihedral_terms.npy"
            init_dihedral_terms = np.load(
                init_dihedral_terms, allow_pickle=True).item()
            self.dihedral_a1 = nn.Parameter(torch.tensor(init_dihedral_terms["dihedral_a1"]).float())
            self.dihedral_a2 = nn.Parameter(torch.tensor(init_dihedral_terms["dihedral_a2"]).float()) 
        else:
            num_dihedral_terms = len(all_dihedral_angle_types)
            self.dihedral_a1 = nn.Parameter(torch.randn(num_dihedral_terms,
                                                        num_dihedral_basis)/1000)
            self.dihedral_a2 = nn.Parameter(torch.randn(num_dihedral_terms,
                                                        num_dihedral_basis)/1000)

        if init_nonbonded_terms:
            init_nonbonded_terms = f"{folder_path}/init_terms/init_nonbonded_terms.npy"
            init_nonbonded_terms = np.load(
                init_nonbonded_terms, allow_pickle=True).item()
            self.lj_r = nn.Parameter(torch.tensor(init_nonbonded_terms["lj_r"]).float())
            self.lj_epsilon = nn.Parameter(torch.tensor(init_nonbonded_terms["lj_epsilon"]).float())
            self.charge = nn.Parameter(torch.tensor(init_nonbonded_terms["charge"]).float())
            self.lj_r_1_4 = nn.Parameter(torch.tensor(init_nonbonded_terms["lj_r_1_4"]).float())
            self.lj_epsilon_1_4 = nn.Parameter(torch.tensor(init_nonbonded_terms["lj_epsilon_1_4"]).float())
            self.charge_1_4 = nn.Parameter(torch.tensor(init_nonbonded_terms["charge_1_4"]).float())
        else:
            num_nonbonded_terms = len(all_pairwise_types)
            self.lj_r = nn.Parameter(torch.abs(torch.rand(num_nonbonded_terms)))
            self.lj_epsilon = nn.Parameter(torch.abs(torch.rand(num_nonbonded_terms)))
            self.charge = nn.Parameter(torch.rand(num_nonbonded_terms) / 1000)
            self.lj_r_1_4 = nn.Parameter(torch.abs(torch.rand(num_nonbonded_terms)))
            self.lj_epsilon_1_4 = nn.Parameter(torch.abs(torch.rand(num_nonbonded_terms)))
            self.charge_1_4 = nn.Parameter(torch.rand(num_nonbonded_terms) / 1000)
        
        if init_calvados_vdw_terms:
            init_calvados_vdw_terms = f"{folder_path}/init_terms/init_calvados_vdw_terms.npy"
            init_calvados_vdw_terms = np.load(
                init_calvados_vdw_terms, allow_pickle=True).item()
            print("Initializing calvados vdw terms from file", flush=True)
            self.calvados_lambdas = nn.Parameter(torch.tensor(init_calvados_vdw_terms["calvados_lambdas"]).float())
            self.calvados_sigmas = nn.Parameter(torch.tensor(init_calvados_vdw_terms["calvados_sigmas"]).float())
            self.calvados_epsilon = nn.Parameter(torch.tensor(init_calvados_vdw_terms["calvados_epsilon"]).float())
        else:
            num_ca = 21
            self.calvados_lambdas = nn.Parameter(torch.abs(torch.rand(num_ca)))
            self.calvados_sigmas = nn.Parameter(torch.abs(torch.rand(num_ca)))
            self.calvados_epsilon = nn.Parameter(torch.abs(torch.rand(1))*10)
            
        self.has_bonded = has_bonded
        self.has_angles = has_angles
        self.has_vdw = has_vdw
        self.has_wca = has_wca
        self.has_coulomb = has_coulomb
        self.has_dihedrals = has_dihedrals
        self.has_calvados_vdw = has_calvados_vdw

        length_units = "angstroms"
        energy_units = "kilocalories_per_mole"

        self.register_buffer("dihedral_period", dihedral_period)

        self.cutoff_distance = vdw_cutoff_distance
        krf = ((self.cutoff_distance ** (-3))
               * (dielectric_constant - 1) / ((2 * dielectric_constant) + 1))
        self.register_buffer("krf", torch.tensor(krf))
        crf = ((3 * dielectric_constant)
               / (self.cutoff_distance * ((2 * dielectric_constant) + 1)))
        self.register_buffer("crf", torch.tensor(crf))

        self.return_forces = return_forces

    def compute_angles(self, all_x, angle_indices):
        p1 = all_x[angle_indices[:, 0], :]
        p2 = all_x[angle_indices[:, 1], :]
        p3 = all_x[angle_indices[:, 2], :]
        v1 = p1 - p2
        v2 = p3 - p2
        dot_product = (v1 * v2).sum(-1)
        norm_v1 = torch.linalg.norm(v1, axis=-1)
        norm_v2 = torch.linalg.norm(v2, axis=-1)
        cos_angle = dot_product / (norm_v1 * norm_v2)
        angle = torch.arccos(cos_angle)
        return angle

    def compute_dihedral(self, x, dihedral_indices):
        p1 = x[dihedral_indices[:, 0], :]
        p2 = x[dihedral_indices[:, 1], :]
        p3 = x[dihedral_indices[:, 2], :]
        p4 = x[dihedral_indices[:, 3], :]
        b1 = p1 - p2
        b2 = p3 - p2
        b3 = p3 - p4

        c1 = torch.cross(b2, b3, dim=-1)
        c2 = torch.cross(b1, b2, dim=-1)

        p1 = (b1 * c1).sum(-1)
        p1 *= (b2 * b2).sum(-1) ** 0.5
        p2 = (c1 * c2).sum(-1)
        angles = torch.atan2(p1, p2)
        return angles

    def forward(self, x,
                bond_distance_indices, bond_distance_features, bond_distances_sctr_indices,
                angle_indices, angle_features, angle_sctr_indices,
                dihedral_indices, dihedral_features, dihedral_sctr_indices,
                nonbonded_edge_indices, nonbonded_edge_features, nonbonded_edge_sctr_indices, batch_size,
                update_nonbonded_edges=False,
                return_forces=None):
        """
        x : (batch_size * num_nodes) x 3
        features : batch_size x num_nodes x 1
        """
        if return_forces is not None:
            self.return_forces = return_forces
        energy = torch.zeros(batch_size, dtype=x.dtype).to(x.device)
        if self.has_bonded:
            distances = torch.linalg.norm(x[bond_distance_indices[:, 0], :] -
                                          x[bond_distance_indices[:, 1], :], dim=-1)
            k_per_bond_distance = self.bonds_k[bond_distance_features]
            mean_per_bond_distance = self.bonds_mean[bond_distance_features]
            bond_energy = (0.5 * k_per_bond_distance
                           * ((distances - mean_per_bond_distance) ** 2))
            energy.scatter_add_(0, bond_distances_sctr_indices, bond_energy)
        if self.has_angles:
            angles = self.compute_angles(x, angle_indices)
            k_per_bond_angle = self.angles_k[angle_features]
            mean_per_bond_angle = self.angles_mean[angle_features]
            angle_energy = (0.5 * k_per_bond_angle
                            * ((angles - mean_per_bond_angle) ** 2))
            energy.scatter_add_(0, angle_sctr_indices, angle_energy)

        if self.has_dihedrals:
            dihedrals = self.compute_dihedral(x, dihedral_indices).unsqueeze(1)

            dihedrals_repeat = dihedrals.repeat_interleave(self.num_dihedral_basis,
                                                    dim=1)
            dihedral_period = self.dihedral_period.repeat(dihedral_indices.shape[0],
                                                          1)
            dihedral_a1 = self.dihedral_a1[dihedral_features]
            dihedral_a2 = self.dihedral_a2[dihedral_features]
            dihedral_energy = torch.sum(dihedral_a1 * (torch.cos(dihedral_period * dihedrals_repeat)) +
                                        dihedral_a2 * (torch.sin(dihedral_period * dihedrals_repeat)), dim=-1)
            energy.scatter_add_(0, dihedral_sctr_indices, dihedral_energy)

            
        if self.has_vdw or self.has_coulomb or self.has_wca:
            if update_nonbonded_edges: 
                all_distances = torch.linalg.norm(x[nonbonded_edge_indices[:, 0], :] -
                                                    x[nonbonded_edge_indices[:, 1], :],
                                                    dim=-1)
                mask = all_distances < self.cutoff_distance
                nonbonded_edge_indices = nonbonded_edge_indices[mask, :]
                nonbonded_edge_features = nonbonded_edge_features[mask, :]
                nonbonded_edge_sctr_indices = nonbonded_edge_sctr_indices[mask]
                pairwise_distance = all_distances[mask] 
            else:
                pairwise_distance = torch.linalg.norm(x[nonbonded_edge_indices[:, 0], :] -
                                                    x[nonbonded_edge_indices[:, 1], :],
                                                    dim=-1)
            mask_1_4 = ((nonbonded_edge_indices[:, 1] - nonbonded_edge_indices[:, 0]) == 4)
            # print(pairwise_distance[~mask_1_4].min(), pairwise_distance[mask_1_4].min())
            atom_feature_1 = nonbonded_edge_features[:, 0]
            atom_feature_2 = nonbonded_edge_features[:, 1] 
            if self.has_coulomb:
                charge = (self.charge[atom_feature_1]
                          * self.charge[atom_feature_2])
                charge_1_4 = (self.charge_1_4[atom_feature_1]
                                * self.charge_1_4[atom_feature_2])
                charge[mask_1_4] = charge_1_4[mask_1_4]
                
                coulomb_energy = (charge
                                  * ((1 / pairwise_distance)
                                     + (self.krf * (pairwise_distance ** 2))
                                     - self.crf))
                energy.scatter_add_(0,
                                    nonbonded_edge_sctr_indices,
                                    coulomb_energy)
                
            if self.has_vdw or self.has_wca:
                sigma = (torch.abs(self.lj_r[atom_feature_1])
                         + torch.abs(self.lj_r[atom_feature_2]))/2
                sigma_1_4 = (torch.abs(self.lj_r_1_4[atom_feature_1])
                             + torch.abs(self.lj_r_1_4[atom_feature_2]))/2
                sigma[mask_1_4] = sigma_1_4[mask_1_4]

                epsilon = torch.abs(self.lj_epsilon[atom_feature_1]
                                    * self.lj_epsilon[atom_feature_2])
                epsilon_1_4 = torch.abs(self.lj_epsilon_1_4[atom_feature_1]
                                        * self.lj_epsilon_1_4[atom_feature_2])
                epsilon[mask_1_4] = epsilon_1_4[mask_1_4]

                vdw_energy = epsilon * ((sigma / pairwise_distance) ** 12
                                        - (2 * ((sigma / pairwise_distance) ** 6)))

                if self.has_wca: # TODO: replaced torch.where with sigmoid for gradient, not quite wca
                    cutoff = (2**(1/6)) * sigma
                    mask = pairwise_distance < cutoff
                    vdw_energy = vdw_energy[mask]
                    # nonbonded_edge_sctr_indices = nonbonded_edge_sctr_indices[mask]
                    energy.scatter_add_(0, nonbonded_edge_sctr_indices[mask], vdw_energy)
                else:
                    energy.scatter_add_(0, nonbonded_edge_sctr_indices, vdw_energy)
                
        if self.has_calvados_vdw:
            atom_feature_1 = nonbonded_edge_features[:, 0]
            atom_feature_2 = nonbonded_edge_features[:, 1] 
            
            # get only alpha carbon terms
            mask_ca = (((atom_feature_1 - 1) % 3 == 0) & (atom_feature_1 <= 61)) & (((atom_feature_2 - 1) % 3 == 0) & (atom_feature_2 <= 61))
            atom_feature_1 = ((atom_feature_1[mask_ca] - 1) / 3).int()
            atom_feature_2 = ((atom_feature_2[mask_ca] - 1) / 3).int()
            nonbonded_edge_indices = nonbonded_edge_indices[mask_ca, :]
            nonbonded_edge_sctr_indices = nonbonded_edge_sctr_indices[mask_ca]
            
            if update_nonbonded_edges: 
                all_distances = torch.linalg.norm(x[nonbonded_edge_indices[:, 0], :] -
                                                    x[nonbonded_edge_indices[:, 1], :],
                                                    dim=-1)
                mask = all_distances < self.cutoff_distance
                nonbonded_edge_indices = nonbonded_edge_indices[mask, :]
                atom_feature_1 = (atom_feature_1[mask]).int()
                atom_feature_2 = (atom_feature_2[mask]).int()
                nonbonded_edge_sctr_indices = nonbonded_edge_sctr_indices[mask]
                pairwise_distance = all_distances[mask] 
            else:
                pairwise_distance = torch.linalg.norm(x[nonbonded_edge_indices[:, 0], :] -
                                                    x[nonbonded_edge_indices[:, 1], :],
                                                    dim=-1)
                    
            # calvados vdw
            sigma = (self.calvados_sigmas[atom_feature_1]
                        + self.calvados_sigmas[atom_feature_2])/2
            lambdas = (self.calvados_lambdas[atom_feature_1]
                        + self.calvados_lambdas[atom_feature_2])/2
            lj_energy = 4 * self.calvados_epsilon * ((sigma / pairwise_distance)**12 - (sigma / pairwise_distance)**6)
            lj_cutoff = 4 * self.calvados_epsilon * ((sigma/self.cutoff_distance)**12 - (sigma/self.cutoff_distance)**6)
            vdw_energy = torch.zeros_like(lj_energy)
            mask = pairwise_distance <= (sigma * (2**(1/6)))
            vdw_energy[mask] = lj_energy[mask] - (lambdas[mask] * lj_cutoff[mask]) + (self.calvados_epsilon * (1 - lambdas[mask]))
            vdw_energy[~mask] = lambdas[~mask] * (lj_energy[~mask] - lj_cutoff[~mask])
            energy.scatter_add_(0, nonbonded_edge_sctr_indices, vdw_energy)
                
        if self.return_forces:
            forces = -torch.autograd.grad(energy, x, create_graph=True, 
                                          grad_outputs = torch.ones_like(energy))[0]
            return forces
        return energy
    
    def l1_loss(self):
        l1_loss = (torch.abs(self.dihedral_a1) + torch.abs(self.dihedral_a2)).mean(0).sum()
        l1_loss += torch.relu(3 - self.lj_r).mean()
        l1_loss += torch.relu(2 - self.lj_r_1_4).mean()
        return l1_loss
    

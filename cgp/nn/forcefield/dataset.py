import torch
import numpy as np

def forcefield_collate_features(bond_distance_indices,
                    bond_distance_features,
                    angle_indices,
                    angle_features,
                    dihedral_indices,
                    dihedral_features,
                    nonbonded_edge_indices,
                    nonbonded_edge_features, **kwargs):

        batch_size = len(bond_distance_features)
        num_atoms = [bond_distance_features[i].shape[0] + 1 for i in range(batch_size)]
        num_atoms_cum = np.insert(np.cumsum(num_atoms[:-1]), 0, 0)

        bond_distances_sctr_indices = torch.cat([torch.zeros(len(bond_distance_features[i]), dtype=torch.int64) + i
                                                for i in range(batch_size)])

        bond_distance_indices = [bond_distance_indices[i] + num_atoms_cum[i]
                                for i in range(batch_size)]
        bond_distance_indices = torch.cat(bond_distance_indices, dim=0).type(torch.int64)
        bond_distance_features = torch.cat(bond_distance_features, dim=0)

        angle_sctr_indices = torch.cat([torch.zeros(len(angle_features[i]), dtype=torch.int64) + i
                                        for i in range(batch_size)])
        angle_indices = [angle_indices[i] + num_atoms_cum[i]
                        for i in range(batch_size)]
        angle_indices = torch.cat(angle_indices, dim=0).type(torch.int64)
        angle_features = torch.cat(angle_features, dim=0)

        dihedral_sctr_indices = torch.cat([torch.zeros(len(dihedral_features[i]), dtype=torch.int64) + i
                                        for i in range(batch_size)])
        dihedral_indices = [dihedral_indices[i] + num_atoms_cum[i]
                            for i in range(batch_size)]
        dihedral_indices = torch.cat(dihedral_indices, dim=0).type(torch.int64)
        dihedral_features = torch.cat(dihedral_features, dim=0)

        nonbonded_edge_sctr_indices = torch.cat([torch.zeros(len(nonbonded_edge_features[i]), dtype=torch.int64) + i
                                                for i in range(batch_size)])
        
        nonbonded_edge_indices = [nonbonded_edge_indices[i] + num_atoms_cum[i]
                                    for i in range(batch_size)]
        
        nonbonded_edge_indices = torch.cat(nonbonded_edge_indices, dim=0).type(torch.int64)
        nonbonded_edge_features = torch.cat(nonbonded_edge_features, dim=0)

        return {"bond_distance_indices": bond_distance_indices,
                "bond_distance_features": bond_distance_features,
                "bond_distances_sctr_indices": bond_distances_sctr_indices.to(bond_distance_indices.device),
                "angle_indices": angle_indices,
                "angle_features": angle_features,
                "angle_sctr_indices": angle_sctr_indices.to(angle_indices.device),
                "dihedral_indices": dihedral_indices,
                "dihedral_features": dihedral_features,
                "dihedral_sctr_indices": dihedral_sctr_indices.to(dihedral_indices.device),
                "nonbonded_edge_indices": nonbonded_edge_indices,
                "nonbonded_edge_features": nonbonded_edge_features,
                "nonbonded_edge_sctr_indices": nonbonded_edge_sctr_indices.to(nonbonded_edge_indices.device),
                "batch_size": batch_size}
    
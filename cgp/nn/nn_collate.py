import torch
import numpy as np
from cgp.nn import NNDataset

def nn_collate_features(atom_features, edge_indices, update_edge_indices, **kwargs):
        batch_size = len(atom_features)
        num_atoms = [atom_features[i].shape[0] for i in range(batch_size)]
        num_atoms_cum = np.insert(np.cumsum(num_atoms[:-1]), 0, 0)

        edge_indices = [edge_indices[i] + num_atoms_cum[i]
                                    for i in range(batch_size)]
        edge_indices = torch.cat(edge_indices, dim=0).to(torch.int64)
        
        atom_features = torch.cat(atom_features, dim=0).type(torch.int64)
        atom_sctr_indices = torch.cat([torch.zeros(num_atoms[i], dtype=torch.int64) + i
                                    for i in range(batch_size)])
        
        return {'features': atom_features,
            'edge_indices': edge_indices.to(atom_features.device),
            'atom_sctr_indices': atom_sctr_indices.to(atom_features.device),
            'batch_size': batch_size,
            'update_edge_indices': update_edge_indices}
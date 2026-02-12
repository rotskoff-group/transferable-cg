import torch
import torch.nn as nn
from .blocks import eSEN_Backbone, MLP_Energy_Head, Linear_Force_Head

class eSEN(nn.Module):
    def __init__(
        self,
        nn_cutoff: float = 20.0,
        max_num_elements: int = 67,
        sphere_channels: int = 128,
        lmax: int = 2,
        mmax: int = 2,
        grid_resolution: int | None = None,
        edge_channels: int = 128,
        distance_function: str = "gaussian", # gaussian or bessel
        num_distance_basis: int = 10,
        direct_forces: bool = False,
        # escnmd specific
        num_layers: int = 2,
        hidden_channels: int = 128,
        norm_type: str = "rms_norm_sh",
        act_type: str = "gate", # gate or s2
        mlp_type: str = "spectral",
        use_envelope: bool = True,
        activation_checkpointing: bool = False,
    ):
        super().__init__()
        self.cutoff = nn_cutoff

        self.backbone = eSEN_Backbone(
            max_num_elements=max_num_elements,
            sphere_channels=sphere_channels,
            lmax=lmax,
            mmax=mmax,
            grid_resolution=grid_resolution,
            cutoff=nn_cutoff,
            edge_channels=edge_channels,
            distance_function=distance_function,
            num_distance_basis=num_distance_basis,
            direct_forces=direct_forces,
            num_layers=num_layers,
            hidden_channels=hidden_channels,
            norm_type=norm_type,
            act_type=act_type,
            mlp_type=mlp_type,
            use_envelope=use_envelope,
            activation_checkpointing=activation_checkpointing,
        )
        
        self.energy_head = MLP_Energy_Head(self.backbone)
        self.force_head = Linear_Force_Head(self.backbone)
        self.direct_forces = direct_forces
        
    def forward(self, pos, features, edge_indices, atom_sctr_indices, 
                batch_size, update_edge_indices=False):
        node_embeddings = self.backbone(pos, features, edge_indices, update_edge_indices)
        
        if self.direct_forces:
            output = self.force_head(node_embeddings)
        else:
            output = self.energy_head(node_embeddings, atom_sctr_indices, batch_size)
            
        return output
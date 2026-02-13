from cgp.nn import UModel, get_collate_fn, generate_edges
import torch
import mdtraj as md
from cgp.cg import (
    get_bond_distance_features,
    get_bond_angle_features,
    get_dihedral_angle_features,
    get_atom_features,
    get_nonbonded_features,
)


class CGSimModel(UModel):
    def __init__(
        self,
        nn_model_name,
        nn_model_args,
        prior_model_name,
        prior_model_args,
        bias_force=None,
        **kwargs,
    ):
        super().__init__(
            nn_model_name, nn_model_args, prior_model_name, prior_model_args, **kwargs
        )
        if bias_force is not None:
            bias_force = torch.jit.load(bias_force).to(self.device)
        self.bias_force = bias_force

    def store_features(self, batch_size, pdb_file, nn_config, prior_config, edge_args):
        _, collate_fn = get_collate_fn(nn_config, prior_config)

        pdb = md.load(pdb_file)
        backbone = pdb.top.select("name CA or name N or name C")
        cg = pdb.atom_slice(backbone)
        n_atoms = cg.n_atoms
        cg_topology = cg.topology
        num_atoms_cum = [i * n_atoms for i in range(batch_size)]

        def repeat_tensor(tensor, n):
            return [tensor for _ in range(n)]

        length_indices, length_features = get_bond_distance_features(
            cg_topology, batch_size=0, as_tensor=True, device=self.device
        )
        angle_indices, angle_features = get_bond_angle_features(
            cg_topology, batch_size=0, as_tensor=True, device=self.device
        )
        dihedral_indices, dihedral_features = get_dihedral_angle_features(
            cg_topology, batch_size=0, as_tensor=True, device=self.device
        )
        atom_features = get_atom_features(
            cg_topology, batch_size=0, as_tensor=True, device=self.device
        )
        all_nonbonded_indices, all_nonbonded_features = get_nonbonded_features(
            cg_topology, batch_size=0, as_tensor=True, device=self.device
        )

        ### Create all possible edge indices (regardless of cutoff) ###
        edges_batch = generate_edges(
            num_atoms=n_atoms,
            nonbonded_edge_indices=all_nonbonded_indices,
            bonded_indices=length_indices,
            angle_indices=angle_indices,
            dihedral_indices=dihedral_indices,
            nn_only_cutoff_edges=edge_args["only_cutoff_edges"],
            nn_only_nonbonded=edge_args["only_nonbonded"],
            include_reverse_edges=True,
        )

        (
            length_indices,
            length_features,
            angle_indices,
            angle_features,
            dihedral_indices,
            dihedral_features,
            atom_features,
            all_nonbonded_indices,
            all_nonbonded_features,
            edges,
        ) = map(
            lambda x: repeat_tensor(x, batch_size),
            (
                length_indices,
                length_features,
                angle_indices,
                angle_features,
                dihedral_indices,
                dihedral_features,
                atom_features,
                all_nonbonded_indices,
                all_nonbonded_features,
                edges_batch,
            ),
        )

        self.features = collate_fn(
            {
                "batch_size": batch_size,
                "num_atoms_cum": num_atoms_cum,
                "bond_distance_indices": length_indices,
                "bond_distance_features": length_features,
                "angle_indices": angle_indices,
                "angle_features": angle_features,
                "dihedral_indices": dihedral_indices,
                "dihedral_features": dihedral_features,
                "nonbonded_edge_indices": all_nonbonded_indices,
                "nonbonded_edge_features": all_nonbonded_features,
                "edge_indices": edges,
                "atom_features": atom_features,
                "update_edge_indices": edge_args["only_cutoff_edges"],
            }
        )

        self.features["prior_features"]["update_nonbonded_edges"] = True

    def get_force(self, pos):
        pos.requires_grad = True
        u_x = self.nn(pos, **self.features)
        if self.nn.return_forces:
            force = u_x
            if self.bias_force is not None:
                bias_force = self.bias_force(pos)
                bias_force = -torch.autograd.grad(
                    bias_force,
                    pos,
                    create_graph=True,
                    grad_outputs=torch.ones_like(bias_force),
                )[0]
                force += bias_force
        else:
            if self.bias_force is not None:
                u_x += self.bias_force(pos)
            force = -torch.autograd.grad(
                u_x, pos, create_graph=True, grad_outputs=torch.ones_like(u_x)
            )[0]
        return force

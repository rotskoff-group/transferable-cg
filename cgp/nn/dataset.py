import torch
from torch.utils.data import Dataset
from typing import Optional
from torch.types import Device


class NNDataset(Dataset):
    def __init__(
        self,
        get_hdf5_data,
        only_cutoff_edges,
        only_nonbonded,
        update_edge_indices,
        data_in_memory=True,
        requires_forces=True,
        requires_energies=False,
        random_rotate=False,
    ):
        self.get_hdf5_data = get_hdf5_data
        self.data_in_memory = data_in_memory
        self.requires_forces = requires_forces
        self.requires_energies = requires_energies
        self.random_rotate = random_rotate
        self.open_hdf5()
        self.length = [len(x["positions"]) for x in self.cg_hdf5]
        self.index = [(i, j) for i, x in enumerate(self.length) for j in range(x)]
        self.length = len(self.index)
        self.nn_only_cutoff_edges = only_cutoff_edges  # only use edges less than cutoff
        self.nn_only_nonbonded = only_nonbonded  # only use edges i,<j+4
        self.update_edge_indices = (
            update_edge_indices  # all dataset built to have a cutoff of 20.0
        )

    def __len__(self):
        return self.length

    def open_hdf5(self):
        self.cg_hdf5 = self.get_hdf5_data()
        if type(self.cg_hdf5) is not list:
            self.cg_hdf5 = [self.cg_hdf5]

        self.positions = [x["positions"] for x in self.cg_hdf5]
        if self.requires_forces:
            self.forces = [x["forces"] for x in self.cg_hdf5]
        if self.requires_energies:
            self.energies = [x["energies"] for x in self.cg_hdf5]

        self.angle_indices = [x["angle_indices"] for x in self.cg_hdf5]
        self.angle_features = [x["angle_features"] for x in self.cg_hdf5]

        self.dihedral_indices = [x["dihedral_indices"] for x in self.cg_hdf5]
        self.dihedral_features = [x["dihedral_features"] for x in self.cg_hdf5]

        self.bond_distance_indices = [x["bond_distance_indices"] for x in self.cg_hdf5]
        self.bond_distance_features = [
            x["bond_distance_features"] for x in self.cg_hdf5
        ]

        self.nonbonded_edge_indices = [
            x["nonbonded_edge_indices"] for x in self.cg_hdf5
        ]
        self.nonbonded_edge_features = [
            x["nonbonded_edge_features"] for x in self.cg_hdf5
        ]

        self.atom_features = [x["atom_features"] for x in self.cg_hdf5]

        if self.data_in_memory:
            self.positions = [x[:] for x in self.positions]
            if self.requires_forces:
                self.forces = [x[:] for x in self.forces]
            if self.requires_energies:
                self.energies = [x[:] for x in self.energies]

            self.angle_indices = [x[:] for x in self.angle_indices]
            self.angle_features = [x[:] for x in self.angle_features]

            self.dihedral_indices = [x[:] for x in self.dihedral_indices]
            self.dihedral_features = [x[:] for x in self.dihedral_features]

            self.bond_distance_indices = [x[:] for x in self.bond_distance_indices]
            self.bond_distance_features = [x[:] for x in self.bond_distance_features]

            self.nonbonded_edge_indices = [x[:] for x in self.nonbonded_edge_indices]
            self.nonbonded_edge_features = [x[:] for x in self.nonbonded_edge_features]

            self.atom_features = [x[:] for x in self.atom_features]

    def __getitem__(self, idx):
        if not hasattr(self, "cg_hdf5"):
            print("Opening HDF5")
            self.open_hdf5()
        i, j = self.index[idx]

        atom_features = self.atom_features[i][j]
        atom_features = torch.tensor(atom_features[atom_features != -1])
        num_atoms = atom_features.shape[0]

        positions = torch.tensor(self.positions[i][j][:num_atoms])
        if self.requires_forces:
            forces = torch.tensor(self.forces[i][j][:num_atoms])
        else:
            forces = None
        if self.requires_energies:
            energies = torch.tensor(self.energies[i][j])
        else:
            energies = None

        bond_distance_features = self.bond_distance_features[i][j]
        bond_distance_features = torch.tensor(
            bond_distance_features[bond_distance_features != -1]
        )

        angle_indices = self.angle_indices[i][j]
        angle_features = self.angle_features[i][j]
        angle_indices = torch.tensor(angle_indices[angle_features != -1])
        angle_features = torch.tensor(angle_features[angle_features != -1])

        dihedral_indices = self.dihedral_indices[i][j]
        dihedral_features = self.dihedral_features[i][j]
        dihedral_indices = torch.tensor(dihedral_indices[dihedral_features != -1])
        dihedral_features = torch.tensor(dihedral_features[dihedral_features != -1])

        bond_distance_indices = self.bond_distance_indices[i][j]
        bond_distance_features = self.bond_distance_features[i][j]
        bond_distance_indices = torch.tensor(
            bond_distance_indices[bond_distance_features != -1]
        )
        bond_distance_features = torch.tensor(
            bond_distance_features[bond_distance_features != -1]
        )

        nonbonded_edge_indices = self.nonbonded_edge_indices[i][j]
        nonbonded_edge_features = self.nonbonded_edge_features[i][j]
        nonbonded_edge_indices = torch.tensor(
            nonbonded_edge_indices[nonbonded_edge_features[:, 0] != -1, :]
        )
        nonbonded_edge_features = torch.tensor(
            nonbonded_edge_features[nonbonded_edge_features[:, 0] != -1, :]
        )

        edges = generate_edges(
            num_atoms=num_atoms,
            nonbonded_edge_indices=nonbonded_edge_indices,
            bonded_indices=bond_distance_indices,
            angle_indices=angle_indices,
            dihedral_indices=dihedral_indices,
            nn_only_cutoff_edges=not self.update_edge_indices,
            nn_only_nonbonded=self.nn_only_nonbonded,  # TODO hardcoded for now nonbonded
            include_reverse_edges=True,
        )

        if self.random_rotate:
            positions, forces = randomly_rotate(positions, forces)
        return (
            positions,
            forces,
            energies,
            angle_indices.to(torch.int64),
            angle_features,
            dihedral_indices.to(torch.int64),
            dihedral_features,
            bond_distance_indices.to(torch.int64),
            bond_distance_features,
            nonbonded_edge_indices.to(torch.int64),
            nonbonded_edge_features,
            edges,
            atom_features,
            self.update_edge_indices,
        )


def generate_edges(
    num_atoms,
    nonbonded_edge_indices,
    bonded_indices,
    angle_indices,
    dihedral_indices,
    nn_only_cutoff_edges,
    nn_only_nonbonded,
    include_reverse_edges=False,
):
    if nn_only_nonbonded:
        if nn_only_cutoff_edges:
            edges = torch.tensor(nonbonded_edge_indices).to(torch.int64)
        else:
            edges = torch.combinations(torch.arange(num_atoms), r=2).to(torch.int64)
            edges = edges[edges[:, 1] - edges[:, 0] > 3]
    else:
        if nn_only_cutoff_edges:  # bonded and nonbonded less than cutoff
            # dihedral_indices = torch.tensor(dihedral_indices.clone())
            # nonbonded_edge_indices = torch.tensor(nonbonded_edge_indices.clone())
            i2 = torch.stack([bonded_indices[:, 0], bonded_indices[:, -1]], dim=1)
            i3 = torch.stack([angle_indices[:, 0], angle_indices[:, -1]], dim=1)
            i4 = torch.stack([dihedral_indices[:, 0], dihedral_indices[:, -1]], dim=1)
            edges = torch.cat([i2, i3, i4, nonbonded_edge_indices], dim=0)

        else:
            edges = torch.combinations(torch.arange(num_atoms), r=2).to(torch.int64)
    if include_reverse_edges:
        edges = torch.cat([edges, edges[:, [1, 0]]], dim=0).to(torch.int64)
    return edges


def randomly_rotate(p, f):
    """Apply random rotations to positions and forces.

    Args:
        p: (N, 3) tensor of positions.
        f: (N, 3) tensor of forces.
    """
    R = random_rotations(1, dtype=p.dtype, device=p.device).squeeze(0)  # (3, 3)
    if p is not None:
        p = p @ R.T  # (N, 3)
    if f is not None:
        f = f @ R.T  # (N, 3)
    return p, f


# taken from https://github.com/rotskoff-group/MoLE/blob/main/src/mole/utils/structure.py
def _copysign(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Return a tensor where each element has the absolute value taken from the,
    corresponding element of a, with sign taken from the corresponding
    element of b. This is like the standard copysign floating-point operation,
    but is not careful about negative 0 and NaN.

    Args:
        a: source tensor.
        b: tensor whose signs will be used, of the same shape as a.

    Returns:
        Tensor of the same shape as a with the signs of b.
    """
    signs_differ = (a < 0) != (b < 0)
    return torch.where(signs_differ, -a, a)


def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def random_quaternions(
    n: int, dtype: Optional[torch.dtype] = None, device: Optional[Device] = None
) -> torch.Tensor:
    """
    Generate random quaternions representing rotations,
    i.e. versors with nonnegative real part.

    Args:
        n: Number of quaternions in a batch to return.
        dtype: Type to return.
        device: Desired device of returned tensor. Default:
            uses the current device for the default tensor type.

    Returns:
        Quaternions as tensor of shape (N, 4).
    """
    if isinstance(device, str):
        device = torch.device(device)
    o = torch.randn((n, 4), dtype=dtype, device=device)
    s = (o * o).sum(1)
    o = o / _copysign(torch.sqrt(s), o[:, 0])[:, None]
    return o


def random_rotations(
    n: int, dtype: Optional[torch.dtype] = None, device: Optional[Device] = None
) -> torch.Tensor:
    """
    Generate random rotations as 3x3 rotation matrices.

    Args:
        n: Number of rotation matrices in a batch to return.
        dtype: Type to return.
        device: Device of returned tensor. Default: if None,
            uses the current device for the default tensor type.

    Returns:
        Rotation matrices as tensor of shape (n, 3, 3).
    """
    quaternions = random_quaternions(n, dtype=dtype, device=device)
    return quaternion_to_matrix(quaternions)

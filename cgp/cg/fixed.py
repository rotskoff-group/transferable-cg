import numpy as np
import mdtraj as md
from .res_info import (
    all_pairwise_types,
    amber_res_code_to_standard,
)
import openmm.unit
import h5py
import dask.array as da
from .utils import (
    get_bond_distance_features,
    get_bond_angle_features,
    get_dihedral_angle_features,
    get_atom_features,
)


class CGFixed:
    def __init__(
        self,
        save_folder_name,
        root_data_folder_name,
        dataset_subset_indices_filename,
        cg_type,
        protein_name,
        store_in_dataset_args,
        cg_length_units,
        cg_energy_units,
        feature_args,
        nonbonded_feature_args,
    ):
        # Defining CG Map
        if cg_type == "backbone":
            atom_names = ["CA", "C", "N"]
        elif cg_type == "c_alpha":
            atom_names = ["CA"]
        else:
            raise ValueError(f"cg_type must be either 'backbone' or 'alpha'")

        data_folder_name = f"{root_data_folder_name}/{protein_name}"
        protein_filename = f"{data_folder_name}/{protein_name}"

        cg_length_units = getattr(openmm.unit, cg_length_units)
        cg_energy_units = getattr(openmm.unit, cg_energy_units)

        if dataset_subset_indices_filename is not None:
            dataset_subset_indices_filename = (
                f"{data_folder_name}/{dataset_subset_indices_filename}.npy"
            )

        fg_positions, fg_forces, fg_energies = self._load_fg_data(
            protein_filename,
            cg_length_units,
            cg_energy_units,
            store_in_dataset_args,
            dataset_subset_indices_filename,
        )

        fg_topology = md.load(
            f"{data_folder_name}/{protein_name}.pdb", frame=0
        ).topology

        cg_indices = [
            atom.index for atom in fg_topology.atoms if atom.name in atom_names
        ]
        cg_topology = fg_topology.subset(cg_indices)

        self.cg_dataset_filename = f"{save_folder_name}/dataset.hdf5"

        file = self._create_cg_dataset(cg_length_units, cg_energy_units)
        if store_in_dataset_args["store_positions"]:
            self._store_cg_positions(fg_positions, file, cg_indices)
        if store_in_dataset_args["store_forces"]:
            self._store_cg_forces(fg_forces, file, cg_indices)
        if store_in_dataset_args["store_energies"]:
            self._store_cg_energies(fg_energies, file)
        if store_in_dataset_args["store_features"]:
            single_config_feature = self._get_feature(cg_topology, **feature_args)
            single_config_feature = da.from_array(single_config_feature, chunks="auto")
            self._store_cg_features(single_config_feature, file["features"])
        if store_in_dataset_args["store_bond_distance_features"]:
            self._store_bond_distance_features(fg_positions, file, cg_topology)
        if store_in_dataset_args["store_bond_angle_features"]:
            self._store_bond_angle_features(fg_positions, file, cg_topology)
        if store_in_dataset_args["store_dihedral_angle_features"]:
            self._store_dihedral_angle_features(fg_positions, file, cg_topology)
        if store_in_dataset_args["store_nonbonded_features"]:
            self._store_nonbonded_features(
                fg_positions,
                file,
                cg_indices,
                cg_length_units,
                cg_topology,
                **nonbonded_feature_args,
            )
        if store_in_dataset_args["store_atom_features"]:
            self._store_atom_features(fg_positions, file, cg_topology)

        file.flush()

        self.cg_topology = cg_topology  # Could make this a class variable earlier
        self.cg_lenth_units = cg_length_units

    def _load_fg_data(
        self,
        protein_filename,
        cg_length_units,
        cg_energy_units,
        store_in_dataset_args,
        dataset_subset_indices_filename,
    ):
        """Loads the full atomistic data from a HDF5 file
        Args:
            protein_filename (str): The name of the protein file
        Returns:
            fg_positions (dask.array): The full atomistic positions
            fg_forces (dask.array): The full atomistic forces
            fg_energies (dask.array): The full atomistic energies
        """
        fg_traj_tw = h5py.File(f"{protein_filename}_traj_all.hdf5", "r")
        if store_in_dataset_args["store_forces"]:
            fg_forces = da.from_array(fg_traj_tw["forces"], chunks="auto")
            fg_energy_units = fg_traj_tw.attrs["energy_units"]
            fg_length_units = fg_traj_tw.attrs["length_units"]
            fg_energy_units = getattr(openmm.unit, fg_energy_units)
            fg_length_units = getattr(openmm.unit, fg_length_units)
            fg_forces = fg_forces * (
                fg_energy_units / fg_length_units
            ).conversion_factor_to(((cg_energy_units / cg_length_units)))
        else:
            fg_forces = None

        if store_in_dataset_args["store_positions"]:
            fg_positions = da.from_array(fg_traj_tw["positions"], chunks="auto")
            fg_length_units = fg_traj_tw.attrs["length_units"]
            fg_length_units = getattr(openmm.unit, fg_length_units)
            fg_positions = fg_positions * fg_length_units.conversion_factor_to(
                cg_length_units
            )
        else:
            fg_positions = None

        if store_in_dataset_args["store_energies"]:
            fg_energies = da.from_array(fg_traj_tw["pe"], chunks="auto")
            fg_energy_units = fg_traj_tw.attrs["energy_units"]
            fg_energy_units = getattr(openmm.unit, fg_energy_units)
            fg_energies = fg_energies * fg_energy_units.conversion_factor_to(
                cg_energy_units
            )
        else:
            fg_energies = None

        if dataset_subset_indices_filename is not None:
            dataset_subset_indices = np.load(dataset_subset_indices_filename)
            if fg_positions is not None:
                fg_positions = fg_positions[dataset_subset_indices]
            if fg_forces is not None:
                fg_forces = fg_forces[dataset_subset_indices]
            if fg_energies is not None:
                fg_energies = fg_energies[dataset_subset_indices]

        if fg_positions is not None and fg_forces is not None:
            assert fg_positions.shape[1] == fg_forces.shape[1]
            assert fg_positions.shape[0] == fg_forces.shape[0]
        if fg_energies is not None:
            if fg_positions is not None:
                assert fg_energies.shape[0] == fg_positions.shape[0]
            if fg_forces is not None:
                assert fg_energies.shape[0] == fg_forces.shape[0]

        return fg_positions, fg_forces, fg_energies

    def _create_cg_dataset(
        self,
        #    num_data_points, num_cg_beads, num_features,
        cg_length_units,
        cg_energy_units,
    ):
        """Creates a HDF5 file to store the coarse-grained data
        Args:
            num_data_points (int): The number of data points
            num_cg_beads (int): The number of coarse-grained beads
            num_features (int): The number of features
            cg_length_units (openmm.unit): The units of length
            cg_energy_units (openmm.unit): The units of energy
        Returns:
            file (h5py.File): The HDF5 file
        """
        file = h5py.File(self.cg_dataset_filename, "w")
        file.attrs["length_units"] = cg_length_units.get_name()
        file.attrs["energy_units"] = cg_energy_units.get_name()
        return file

    def _store_cg_positions(self, fg_positions, file, indices_to_keep):
        """Stores the coarse-grained positions in a HDF5 file
        Args:
            fg_positions (dask.array): The full atomistic positions (num_data_points, num_atoms, 3)
            file (h5py.File): The HDF5 file
            indices_to_keep (list): The indices to keep (num_cg_beads)
        """
        cg_positions_dataset = file.create_dataset(
            "positions", (fg_positions.shape[0], len(indices_to_keep), 3), dtype="f4"
        )
        fg_positions[:, indices_to_keep].store(cg_positions_dataset)

    def _store_cg_forces(self, fg_forces, file, indices_to_keep):
        """Stores the coarse-grained forces in a HDF5 file
        Args:
            fg_forces (dask.array): The full atomistic forces (num_data_points, num_atoms, 3)
            file (h5py.File): The HDF5 file
            indices_to_keep (list): The indices to keep (num_cg_beads)
        """
        cg_forces_dataset = file.create_dataset(
            "forces", (fg_forces.shape[0], len(indices_to_keep), 3), dtype="f4"
        )
        fg_forces[:, indices_to_keep].store(cg_forces_dataset)

    def _store_cg_energies(self, fg_energies, file):
        """Stores the coarse-grained energies in a HDF5 file
        Args:
            fg_energies (dask.array): The full atomistic energies (num_data_points, 1)
            file (h5py.File): The HDF5 file
        """
        energies_dataset = file.create_dataset(
            "energies", (fg_energies.shape[0], 1), dtype="f4"
        )
        pass

    def _store_cg_features(self, single_config_feature, file, feature_args):
        """Stores the coarse-grained features in a HDF5 file
        Args:
            single_config_feature (dask.array): The features for a single configuration (1, num_cg_beads, num_features)
            file (h5py.File): The HDF5 file
        """
        # add_features here
        # cg_features_dataset = file.create_dataset(
        #     "features", (num_data_points, num_cg_beads, num_features), dtype="i4"
        # )
        # single_config_feature[[0] * len(cg_features_dataset)].store(cg_features_dataset)
        raise NotImplementedError("This function should be implemented in the subclass")
    
    def _get_feature(
        self,
        cg_topology,
    ):
        """Gets the features for a single configuration
        Args:
            cg_topology (mdtraj.Topology): The coarse-grained topology
            use_atomic_number_features (bool): Whether to use atomic number features
            use_residue_features (bool): Whether to use residue features
        Returns:
            all_features (np.array): The features for a single configuration (1, num_cg_beads, num_features)
        """
        raise NotImplementedError("This function should be implemented in the subclass")

    def _store_bond_distance_features(self, fg_positions, file, cg_topology):
        """Stores the coarse-grained distances in a HDF5 file
        Args:
        """
        all_bond_indices, all_bond_features = get_bond_distance_features(
            cg_topology=cg_topology, batch_size=fg_positions.shape[0]
        )

        file.create_dataset("bond_distance_indices", data=all_bond_indices)
        file.create_dataset("bond_distance_features", data=all_bond_features)

    def _store_bond_angle_features(
        self,
        fg_positions,
        file,
        cg_topology,
    ):
        """Stores the coarse-grained distances in a HDF5 file
        Args:
        """
        all_angle_indices, all_angle_features = get_bond_angle_features(
            cg_topology=cg_topology, batch_size=fg_positions.shape[0]
        )

        file.create_dataset("angle_indices", data=all_angle_indices)
        file.create_dataset("angle_features", data=all_angle_features)

    def _store_dihedral_angle_features(self, fg_positions, file, cg_topology):
        """Stores the coarse-grained distances in a HDF5 file
        Args:
        """
        all_dihdral_indices, all_dihedral_features = get_dihedral_angle_features(
            cg_topology=cg_topology, batch_size=fg_positions.shape[0]
        )

        file.create_dataset("dihedral_indices", data=all_dihdral_indices)
        file.create_dataset("dihedral_features", data=all_dihedral_features)

    def _store_atom_features(self, fg_positions, file, cg_topology):
        all_atom_features = get_atom_features(
            cg_topology=cg_topology, batch_size=fg_positions.shape[0]
        )
        file.create_dataset("atom_features", data=all_atom_features)

    def _store_nonbonded_features(
        self,
        fg_positions,
        file,
        indices_to_keep,
        cg_length_units,
        cg_topology,
        cutoff,
        cutoff_units,
    ):
        """Stores the coarse-grained distances in a HDF5 file
        Args:
            fg_positions (dask.array): The full atomistic positions (num_data_points, num_atoms, 3)
            cg_distances_dataset (h5py.Dataset): The dataset to store the coarse-grained positions
            indices_to_keep (list): The indices to keep (num_cg_beads)
        """
        cg_positions = fg_positions[:, indices_to_keep]
        all_pairwise_indices = [
            [i, j]
            for i in range(cg_topology.n_atoms)
            for j in range(i + 4, cg_topology.n_atoms)
        ]
        all_pairwise_indices = np.array(all_pairwise_indices)
        all_pairwise_features = [
            [
                all_pairwise_types[
                    amber_res_code_to_standard[
                        cg_topology.atom(atom_index_1).residue.name
                    ],
                    cg_topology.atom(atom_index_1).name,
                ],
                all_pairwise_types[
                    amber_res_code_to_standard[
                        cg_topology.atom(atom_index_2).residue.name
                    ],
                    cg_topology.atom(atom_index_2).name,
                ],
            ]
            for (atom_index_1, atom_index_2) in all_pairwise_indices
        ]
        all_pairwise_features = np.array(all_pairwise_features)
        all_distances = da.linalg.norm(
            cg_positions[:, all_pairwise_indices[:, 0]]
            - cg_positions[:, all_pairwise_indices[:, 1]],
            axis=-1,
        )
        cutoff = cutoff * getattr(openmm.unit, cutoff_units).conversion_factor_to(
            cg_length_units
        )

        # Compute max num edges
        chunk_size = 150
        num_chunks = (all_distances.shape[0] // chunk_size) + 1
        all_num_edges = []
        for i in range(num_chunks):
            distances_sub = np.array(
                all_distances[i * chunk_size : (i + 1) * chunk_size]
            )
            num_edges = [
                np.stack(np.where(distances_sub_i < cutoff)).shape[1]
                for distances_sub_i in distances_sub
            ]
            all_num_edges.extend(num_edges)
        max_num_edges = max(all_num_edges)

        # Compute edges (under max_r) and save corresponding edge_distances
        file.create_dataset(
            "nonbonded_edge_indices",
            (all_distances.shape[0], max_num_edges, 2),
            dtype="i4",
            fillvalue=-1,
        )
        file.create_dataset(
            "nonbonded_edge_features",
            (all_distances.shape[0], max_num_edges, 2),
            dtype="i4",
            fillvalue=-1,
        )

        # Compute edges and distances
        chunk_size = 150
        num_chunks = (all_distances.shape[0] // chunk_size) + 1
        for i in range(num_chunks):
            distances_sub = np.array(
                all_distances[i * chunk_size : (i + 1) * chunk_size]
            )
            edge_indices_sub = [
                all_pairwise_indices[np.stack(np.where(distances_sub_i < cutoff))]
                for distances_sub_i in distances_sub
            ]
            edge_features_sub = [
                all_pairwise_features[np.stack(np.where(distances_sub_i < cutoff))]
                for distances_sub_i in distances_sub
            ]
            for j, (edges_sub_i, edge_features_sub_i) in enumerate(
                zip(edge_indices_sub, edge_features_sub)
            ):

                file["nonbonded_edge_indices"][
                    i * chunk_size + j, : edges_sub_i.shape[1], :
                ] = edges_sub_i
                file["nonbonded_edge_features"][
                    i * chunk_size + j, : edges_sub_i.shape[1], :
                ] = edge_features_sub_i


    def get_cg_dataset_filename(self):
        """Gets the formatted cg dataset
        Returns:
            file (h5py.File): The dataset
        """
        return self.cg_dataset_filename

    def get_traj(self, config_index):
        """Gets the cg topology
        Args:
            config_index (slice): The indices of the configurations to save
        Returns:
            cg_trajectory (md.Trajectory): The coarse-grained trajectory
        """
        reader = h5py.File(self.cg_dataset_filename, "r")
        cg_positions = reader["positions"][config_index]
        conversion_factor = self.cg_lenth_units.conversion_factor_to(
            openmm.unit.nanometer
        )
        cg_positions = cg_positions * conversion_factor
        cg_trajectory = md.Trajectory(cg_positions, self.cg_topology)

        return cg_trajectory

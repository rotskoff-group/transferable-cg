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
        force_map_args,
        mean_force_estimation,
    ):
        # Defining CG Map
        if cg_type == "backbone":
            atom_names = ["CA", "C", "N"]
        elif cg_type == "c_alpha":
            atom_names = ["CA"]
        else:
            raise ValueError("cg_type must be either 'backbone' or 'alpha'")

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

        force_map_strategy = force_map_args["strategy"]
        if force_map_strategy not in ("slice", "optimized"):
            raise ValueError("force_map_args.strategy must be 'slice' or 'optimized'")
        if force_map_strategy == "optimized":
            if force_map_args.get("cutoff") is None:
                raise ValueError(
                    "force_map_args.cutoff must be specified when strategy is 'optimized'"
                )
            if fg_positions is None:
                raise ValueError(
                    "store_positions must be True when force_map_args.strategy is 'optimized'"
                )
        if mean_force_estimation:
            if store_in_dataset_args["store_energies"]:
                raise ValueError(
                    "store_energies must be False when mean_force_estimation is True"
                )
            if fg_positions is None:
                raise ValueError(
                    "store_positions must be True when mean_force_estimation is True"
                )
            self._assert_cg_positions_fixed(fg_positions, cg_indices)

        # For features that depend only on topology (bond, angle, dihedral, atom),
        # batch_size is read from fg_positions.shape[0]. With mean_force_estimation
        # we want batch_size=1, so pass a single-frame slice.
        positions_for_features = fg_positions[0:1] if mean_force_estimation else fg_positions

        self.cg_dataset_filename = f"{save_folder_name}/dataset.hdf5"

        file = self._create_cg_dataset(cg_length_units, cg_energy_units)
        if store_in_dataset_args["store_positions"]:
            self._store_cg_positions(
                fg_positions, file, cg_indices, mean_force_estimation=mean_force_estimation
            )
        if store_in_dataset_args["store_forces"]:
            if force_map_strategy == "slice":
                self._store_cg_forces(
                    fg_forces, file, cg_indices, mean_force_estimation=mean_force_estimation
                )
            else:
                self._store_optimized_cg_forces(
                    fg_positions,
                    fg_forces,
                    file,
                    cg_indices,
                    force_map_args["cutoff"],
                    force_map_args.get("random_subset_size"),
                    random_seed=force_map_args.get("random_seed"),
                    mean_force_estimation=mean_force_estimation,
                )
        if store_in_dataset_args["store_energies"]:
            self._store_cg_energies(fg_energies, file)
        if store_in_dataset_args["store_features"]:
            single_config_feature = self._get_feature(cg_topology, **feature_args)
            single_config_feature = da.from_array(single_config_feature, chunks="auto")
            self._store_cg_features(single_config_feature, file["features"])
        if store_in_dataset_args["store_bond_distance_features"]:
            self._store_bond_distance_features(positions_for_features, file, cg_topology)
        if store_in_dataset_args["store_bond_angle_features"]:
            self._store_bond_angle_features(positions_for_features, file, cg_topology)
        if store_in_dataset_args["store_dihedral_angle_features"]:
            self._store_dihedral_angle_features(positions_for_features, file, cg_topology)
        if store_in_dataset_args["store_nonbonded_features"]:
            self._store_nonbonded_features(
                positions_for_features,
                file,
                cg_indices,
                cg_length_units,
                cg_topology,
                **nonbonded_feature_args,
            )
        if store_in_dataset_args["store_atom_features"]:
            self._store_atom_features(positions_for_features, file, cg_topology)

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
            ).conversion_factor_to((cg_energy_units / cg_length_units))
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

    def _store_cg_positions(self, fg_positions, file, indices_to_keep, mean_force_estimation=False):
        """Stores the coarse-grained positions in a HDF5 file
        Args:
            fg_positions (dask.array): The full atomistic positions (num_data_points, num_atoms, 3)
            file (h5py.File): The HDF5 file
            indices_to_keep (list): The indices to keep (num_cg_beads)
            mean_force_estimation (bool): If True, store only the first frame as a (1, n_beads, 3) dataset
        """
        if mean_force_estimation:
            first_frame = np.array(fg_positions[0:1, indices_to_keep])  # (1, n_beads, 3)
            dataset = file.create_dataset(
                "positions", (1, len(indices_to_keep), 3), dtype="f4"
            )
            dataset[:] = first_frame
        else:
            cg_positions_dataset = file.create_dataset(
                "positions", (fg_positions.shape[0], len(indices_to_keep), 3), dtype="f4"
            )
            fg_positions[:, indices_to_keep].store(cg_positions_dataset)

    def _store_cg_forces(self, fg_forces, file, indices_to_keep, mean_force_estimation=False):
        """Stores the coarse-grained forces in a HDF5 file
        Args:
            fg_forces (dask.array): The full atomistic forces (num_data_points, num_atoms, 3)
            file (h5py.File): The HDF5 file
            indices_to_keep (list): The indices to keep (num_cg_beads)
            mean_force_estimation (bool): If True, store the mean force as a (1, n_beads, 3) dataset
        """
        if mean_force_estimation:
            mean_forces = np.array(fg_forces[:, indices_to_keep].mean(axis=0))  # (n_beads, 3)
            dataset = file.create_dataset(
                "forces", (1, len(indices_to_keep), 3), dtype="f4"
            )
            dataset[0] = mean_forces
        else:
            cg_forces_dataset = file.create_dataset(
                "forces", (fg_forces.shape[0], len(indices_to_keep), 3), dtype="f4"
            )
            fg_forces[:, indices_to_keep].store(cg_forces_dataset)

    def _assert_cg_positions_fixed(self, fg_positions, cg_indices):
        """Assert that CG bead positions are identical across all frames."""
        cg_positions = np.array(fg_positions[:, cg_indices])  # materialise (n_frames, n_beads, 3)
        reference = cg_positions[0]                            # (n_beads, 3)
        diff = cg_positions - reference[None, :, :]
        max_displacement = np.abs((diff[:, 0:1, :] - diff[:, 1:, :]).max())

        assert max_displacement < 1e-3, (
            f"mean_force_estimation=True requires all CG positions to be identical, "
            f"but max displacement is {max_displacement}"
        )

    def _store_optimized_cg_forces(
        self, fg_positions, fg_forces, file, cg_indices, cutoff, random_subset_size,
        random_seed=None, mean_force_estimation=False,
    ):
        """Stores optimized CG forces via local least-squares force matching.

        If random_subset_size is given, a random subset of frames is used to fit the
        per-bead weights; the fitted weights are then applied to ALL frames so the
        stored forces have the same length as the full dataset. cutoff must be in the
        same units as fg_positions.

        If mean_force_estimation is True, the per-frame optimal forces are averaged
        and stored as a single-frame (1, n_beads, 3) dataset.

        Args:
            fg_positions (dask.array): Full atomistic positions (n_frames, n_atoms, 3)
            fg_forces (dask.array): Full atomistic forces (n_frames, n_atoms, 3)
            file (h5py.File): The HDF5 file
            cg_indices (list): Indices of CG representative atoms
            cutoff (float): Local neighbourhood cutoff in same units as positions
            random_subset_size (int or None): Number of frames used for fitting only
            mean_force_estimation (bool): If True, store the mean force over all frames
        """
        import torch
        from .force_map import get_all_optimal_cg_forces

        all_positions = torch.from_numpy(np.array(fg_positions))
        all_forces = torch.from_numpy(np.array(fg_forces))
        cg_indices_tensor = torch.tensor(cg_indices)

        fit_positions = fit_forces = None
        if random_subset_size is not None:
            generator = torch.Generator()
            if random_seed is not None:
                generator.manual_seed(random_seed)
            subset_idx = torch.randperm(all_positions.shape[0], generator=generator)[:random_subset_size]
            fit_positions = all_positions[subset_idx]
            fit_forces = all_forces[subset_idx]

        # get_all_optimal_cg_forces fits on fit_positions/fit_forces (or all frames if
        # None) and applies the learned weights to all_positions/all_forces
        cg_forces = get_all_optimal_cg_forces(
            all_positions, all_forces, cg_indices_tensor, cutoff,
            fit_positions=fit_positions, fit_forces=fit_forces,
        )
        if mean_force_estimation:
            # (n_beads, n_frames, 3) -> (n_beads, 1, 3) -> (1, n_beads, 3)
            cg_forces = cg_forces.mean(dim=1, keepdim=True).permute(1, 0, 2).numpy()
        else:
            # (n_beads, n_frames, 3) -> (n_frames, n_beads, 3)
            cg_forces = cg_forces.permute(1, 0, 2).numpy()

        cg_forces_dataset = file.create_dataset("forces", cg_forces.shape, dtype="f4")
        cg_forces_dataset[:] = cg_forces

    def _store_cg_energies(self, fg_energies, file):
        """Stores the coarse-grained energies in a HDF5 file
        Args:
            fg_energies (dask.array): The full atomistic energies (num_data_points, 1)
            file (h5py.File): The HDF5 file
        """
        file.create_dataset("energies", (fg_energies.shape[0], 1), dtype="f4")
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

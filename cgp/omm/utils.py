import os
import re
import numpy as np
import h5py
import openmm.unit

omm_state_input_keys = {
    "save_forces": "getForces",
    "save_positions": "getPositions",
    "save_velocities": "getVelocities",
    "save_energy": "getEnergy",
    "enforce_periodic_box": "enforcePeriodicBox",
}

kT = (
    300
    * openmm.unit.kelvin
    * openmm.unit.BOLTZMANN_CONSTANT_kB
    * openmm.unit.AVOGADRO_CONSTANT_NA
)


def parse_checkpoints(chk_dir, name):
    """Scan checkpoint files to determine simulation resume state.

    Returns:
        i_cur: index of the file to write next (= i_last_final + 1)
        j_cur: index of the last in-progress checkpoint within i_cur (-1 if none)
        i_last_final: index of the last fully completed file (-1 if none)
    """
    final_pat = re.compile(rf"^{re.escape(name)}_(\d+)_final\.chk$")
    inprog_pat = re.compile(rf"^{re.escape(name)}_(\d+)_(\d+)\.chk$")

    chk_files = os.listdir(chk_dir)
    final_idxs = [int(m.group(1)) for f in chk_files if (m := final_pat.match(f))]
    i_last_final = max(final_idxs) if final_idxs else -1
    i_cur = i_last_final + 1

    inprog = [
        (int(m.group(1)), int(m.group(2)))
        for f in chk_files
        if (m := inprog_pat.match(f))
    ]
    inprog_for_i = [(fi, j) for (fi, j) in inprog if fi == i_cur]
    j_cur = max(j for (_, j) in inprog_for_i) if inprog_for_i else -1

    return i_cur, j_cur, i_last_final


def concatenate_trajectories(traj_dir, name, n_files, out_path, extra_datasets=None):
    """Concatenate per-file HDF5 trajectories into out_path one file at a time.

    Streams data without holding all files in memory simultaneously.
    Copies file-level attrs from the first source file to the output.
    extra_datasets: optional {key: array} written to the output file after concatenation.
    """
    first_path = os.path.join(traj_dir, f"{name}_0.hdf5")
    with h5py.File(first_path, "r") as f0:
        keys = list(f0.keys())
        tail_shapes = {k: f0[k].shape[1:] for k in keys}
        dtypes = {k: f0[k].dtype for k in keys}
        attrs = dict(f0.attrs)

    with h5py.File(out_path, "w") as out:
        out.attrs.update(attrs)

        for k in keys:
            out.create_dataset(
                k,
                shape=(0,) + tail_shapes[k],
                maxshape=(None,) + tail_shapes[k],
                dtype=dtypes[k],
            )

        for i in range(n_files):
            traj_path = os.path.join(traj_dir, f"{name}_{i}.hdf5")
            with h5py.File(traj_path, "r") as src:
                for k in keys:
                    chunk = src[k][:]
                    n_cur = out[k].shape[0]
                    out[k].resize(n_cur + len(chunk), axis=0)
                    out[k][n_cur:] = chunk

        if extra_datasets:
            for k, v in extra_datasets.items():
                out.create_dataset(k, data=v)


def update_standard_error(prev_mean, prev_std, new_data, n):
    new_mean = prev_mean + ((new_data - prev_mean) / n)
    prev_var = prev_std**2
    new_std = np.sqrt(
        prev_var + ((((new_data - prev_mean) * (new_data - new_mean)) - prev_var) / n)
    )
    se = new_std / np.sqrt(n)
    return new_mean, new_std, se


def estimate_mean_force(
    protein,
    position=None,
    minimizationIterations=None,
    standard_error_cutoff=None,
    max_steps=10000,
    save_freq=10,
    length_units="angstroms",
    time_units="picoseconds",
    energy_units="kilocalories_per_mole",
    enforce_periodic_box=True,
):
    """
    Estimate the mean force on fixed atoms of the system
    Arguments:
        position: The position to start from (if None, the current position is used)
        minimizationIterations: The number of minimization iterations to run (if None, no minimization is run)
        standard_error_cutoff: Mean force estimation is run until the standard error is below this value (if None, the full number of steps is run) in kT/length_units
        max_steps: The maximum number of steps to run (if None, mean force estimation is run until the standard error is below the cutoff)
        save_freq: The frequency to save forces to compute mean force and standard error
    Returns:
        mean_force: The mean force on protein atoms (only fixed atoms are meaningful)
        standard_error: The standard error of the mean force on fixed atoms
    """
    if standard_error_cutoff is None:
        assert (
            max_steps is not None
        )  # If standard_error_cutoff is not None, max_steps must be specified
        max_i = max_steps // save_freq
    else:
        standard_error_cutoff = (standard_error_cutoff / protein.beta) / getattr(
            openmm.unit, length_units
        )
        standard_error_cutoff = standard_error_cutoff.in_units_of(
            getattr(openmm.unit, energy_units) / getattr(openmm.unit, length_units)
        )._value
        max_i = None
        if max_steps is not None:
            max_i = max_steps // save_freq

    if position is not None:
        protein.update_positions_and_velocities(
            position, velocities=None, length_units=length_units, time_units=time_units
        )
    if minimizationIterations is not None:
        protein.simulation.minimizeEnergy(maxIterations=minimizationIterations)

    _, _, _, pe, _ = protein.get_information(
        length_units=length_units,
        time_units=time_units,
        energy_units=energy_units,
        as_numpy=True,
        enforce_periodic_box=enforce_periodic_box,
    )
    if pe > 1e4:
        return None, None, None, None

    # Relax system before estimating mean force
    protein.simulation.step(1000)

    inst_forces = []
    positions = []
    energy = []
    i = 0
    while True:
        if max_i is not None and i >= max_i:
            mean_force = np.mean(
                np.array(inst_forces)[:, protein.fixed_atom_indices, :], axis=0
            )
            break
        protein.simulation.step(save_freq)
        p, _, forces, pe, _ = protein.get_information(
            length_units=length_units,
            time_units=time_units,
            energy_units=energy_units,
            as_numpy=True,
            enforce_periodic_box=enforce_periodic_box,
        )
        inst_forces.append(forces[protein.target_atom_indices, :])
        positions.append(p[protein.target_atom_indices, :])
        energy.append(pe)
        if i == 999:
            mean_force = np.mean(
                np.array(inst_forces)[:, protein.fixed_atom_indices, :], axis=0
            )
            std_force = np.std(
                np.array(inst_forces)[:, protein.fixed_atom_indices, :], axis=0
            )
            standard_error = std_force / np.sqrt(i)

        elif i >= 1000:
            mean_force, std_force, standard_error = update_standard_error(
                mean_force, std_force, forces[protein.fixed_atom_indices], i + 1
            )
            if np.max(standard_error) < standard_error_cutoff:
                break
        i += 1
    output_mean_force = np.zeros((len(protein.target_atom_indices), 3))
    output_mean_force[protein.fixed_atom_indices, :] = mean_force
    positions = np.array(positions)
    inst_forces = np.array(inst_forces)
    energy = np.array(energy)
    assert positions.shape[0] == inst_forces.shape[0] == energy.shape[0]
    return output_mean_force, inst_forces, positions, energy


class TrajWriter:
    def __init__(self, filename, target_atom_indices, num_data_points, tw_args):
        self.filename = filename
        self.target_atom_indices = target_atom_indices
        num_atoms = len(self.target_atom_indices)
        self.omm_state_inputs = self._get_omm_info(tw_args)

        if os.path.exists(self.filename):
            self.file = h5py.File(self.filename, "a")
            self._validate_existing_file(num_data_points, num_atoms)
        else:
            self.file = h5py.File(self.filename, "w")
            self.file.create_dataset(
                "forces", (num_data_points, num_atoms, 3), dtype="f4"
            )
            self.file.create_dataset(
                "positions", (num_data_points, num_atoms, 3), dtype="f4"
            )
            self.file.create_dataset(
                "velocities", (num_data_points, num_atoms, 3), dtype="f4"
            )
            self.file.create_dataset("pe", (num_data_points,), dtype="f4")
            self.file.create_dataset("ke", (num_data_points,), dtype="f4")

            for key, value in tw_args.items():
                self.file.attrs[key] = value

    def _validate_existing_file(self, num_data_points, num_atoms):
        """Assert that an existing HDF5 file has the expected datasets and shapes.

        Raises ValueError if required datasets are missing or their shapes are
        incompatible with the current num_data_points / target_atom_indices so
        that configuration mismatches are caught immediately rather than
        silently corrupting data.
        """
        required_3d = ("forces", "positions", "velocities")
        required_1d = ("pe", "ke")
        required = list(required_3d) + list(required_1d)
        for k in required:
            if k not in self.file.keys():
                self.file.close()
                raise ValueError(
                    f"Existing trajectory file '{self.filename}' is missing required "
                    f"dataset '{k}'. Cannot resume."
                )

        for k in required_3d:
            expected = (num_data_points, num_atoms, 3)
            actual = self.file[k].shape
            if actual != expected:
                self.file.close()
                raise ValueError(
                    f"Dataset '{k}' in '{self.filename}' has shape {actual} but "
                    f"expected {expected}. frames_per_file or atom selection may "
                    f"have changed between runs. Cannot resume."
                )

        for k in required_1d:
            expected = (num_data_points,)
            actual = self.file[k].shape
            if actual != expected:
                self.file.close()
                raise ValueError(
                    f"Dataset '{k}' in '{self.filename}' has shape {actual} but "
                    f"expected {expected}. frames_per_file may have changed "
                    f"between runs. Cannot resume."
                )

    def _get_omm_info(self, tw_args):
        assert "length_units" in tw_args
        assert "energy_units" in tw_args
        assert "time_units" in tw_args
        assert "save_forces" in tw_args
        assert "save_positions" in tw_args
        assert "save_velocities" in tw_args
        assert "save_energy" in tw_args
        assert "enforce_periodic_box" in tw_args

        omm_state_inputs = {}
        for tw_key, omm_key in omm_state_input_keys.items():
            omm_state_inputs[omm_key] = tw_args[tw_key]

        length_units = tw_args["length_units"]
        energy_units = tw_args["energy_units"]
        time_units = tw_args["time_units"]
        self.length_units = getattr(openmm.unit, length_units)
        self.energy_units = getattr(openmm.unit, energy_units)
        self.time_units = getattr(openmm.unit, time_units)
        return omm_state_inputs

    def write(self, simulation, frame):

        state = simulation.context.getState(**self.omm_state_inputs)

        if self.omm_state_inputs["getForces"]:
            forces = (
                state.getForces(asNumpy=True)
                .in_units_of(self.energy_units / self.length_units)
                ._value
            )
            forces = forces[self.target_atom_indices, :]
            self.file["forces"][frame] = forces
        if self.omm_state_inputs["getPositions"]:
            positions = (
                state.getPositions(asNumpy=True).in_units_of(self.length_units)._value
            )
            positions = positions[self.target_atom_indices, :]
            self.file["positions"][frame] = positions
        if self.omm_state_inputs["getVelocities"]:
            velocities = (
                state.getVelocities(asNumpy=True)
                .in_units_of(self.length_units / self.time_units)
                ._value
            )
            velocities = velocities[self.target_atom_indices, :]
            self.file["velocities"][frame] = velocities
        if self.omm_state_inputs["getEnergy"]:
            pe = state.getPotentialEnergy().in_units_of(self.energy_units)._value
            self.file["pe"][frame] = pe
            ke = state.getKineticEnergy().in_units_of(self.energy_units)._value
            self.file["ke"][frame] = ke

        self.file.flush()

    def close(self):
        self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

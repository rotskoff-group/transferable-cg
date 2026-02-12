import numpy as np
import h5py
import mdtraj as md
import openmm.unit
from openmm.app import AmberPrmtopFile
from omegaconf import OmegaConf

omm_state_input_keys = {"save_forces": "getForces",
                        "save_positions": "getPositions",
                        "save_velocities": "getVelocities",
                        "save_energy": "getEnergy",
                        "enforce_periodic_box": "enforcePeriodicBox"}

kT = (300 * openmm.unit.kelvin
      * openmm.unit.BOLTZMANN_CONSTANT_kB
      * openmm.unit.AVOGADRO_CONSTANT_NA)

def update_standard_error(prev_mean, prev_std, new_data, n):
    new_mean = prev_mean + ((new_data - prev_mean) / n)
    prev_var = prev_std**2
    new_std = np.sqrt(prev_var + ((((new_data - prev_mean) * (new_data - new_mean)) - prev_var) / n))
    se = new_std / np.sqrt(n)
    return new_mean, new_std, se

def estimate_mean_force(protein, 
                        position=None, 
                        minimizationIterations=None, 
                        standard_error_cutoff=None, 
                        max_steps=10000, 
                        save_freq=10,
                        length_units="angstroms",
                        time_units="picoseconds",
                        energy_units="kilocalories_per_mole",
                        enforce_periodic_box=True):
        '''
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
        '''
        if standard_error_cutoff is None:
            assert max_steps is not None # If standard_error_cutoff is not None, max_steps must be specified
            max_i = max_steps // save_freq
        else:
            standard_error_cutoff = (standard_error_cutoff / protein.beta) / getattr(openmm.unit, length_units) 
            standard_error_cutoff = standard_error_cutoff.in_units_of(getattr(openmm.unit, energy_units) / getattr(openmm.unit, length_units))._value
            max_i = None
            if max_steps is not None:
                max_i = max_steps // save_freq

        if position is not None:
            protein.update_positions_and_velocities(position,
                                                velocities=None,
                                                length_units=length_units,
                                                time_units=time_units)
        if minimizationIterations is not None:
            protein.simulation.minimizeEnergy(maxIterations=minimizationIterations)

        _, _, _, pe, _ = protein.get_information(length_units=length_units,
                                            time_units=time_units,
                                            energy_units=energy_units,
                                            as_numpy=True,
                                            enforce_periodic_box=enforce_periodic_box)
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
                mean_force = np.mean(np.array(inst_forces)[:, protein.fixed_atom_indices, :], axis=0)
                break
            protein.simulation.step(save_freq)
            p, _, forces, pe, _ = protein.get_information(length_units=length_units,
                                                    time_units=time_units,
                                                    energy_units=energy_units,
                                                    as_numpy=True,
                                                    enforce_periodic_box=enforce_periodic_box)
            inst_forces.append(forces[protein.target_atom_indices, :])
            positions.append(p[protein.target_atom_indices, :])
            energy.append(pe)
            if i == 999:
                mean_force = np.mean(np.array(inst_forces)[:, protein.fixed_atom_indices, :], axis=0)
                std_force = np.std(np.array(inst_forces)[:, protein.fixed_atom_indices, :], axis=0)
                standard_error = std_force / np.sqrt(i)

            elif i >= 1000:
                mean_force, std_force, standard_error = update_standard_error(mean_force, std_force, forces[protein.fixed_atom_indices], i+1)
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
    def __init__(self, filename, target_atom_indices,
                 num_data_points, tw_args):
        self.filename = filename
        self.target_atom_indices = target_atom_indices
        num_atoms = len(self.target_atom_indices)
        self.omm_state_inputs = self._get_omm_info(tw_args)

        self.file = h5py.File(self.filename, "w")
        self.file.create_dataset(f"forces", (num_data_points, num_atoms, 3),
                                 dtype='f4')
        self.file.create_dataset(f"positions", (num_data_points, num_atoms, 3),
                                 dtype='f4')
        self.file.create_dataset(f"velocities", (num_data_points, num_atoms, 3),
                                 dtype='f4')
        self.file.create_dataset(f"pe", (num_data_points, ),
                                 dtype='f4')
        self.file.create_dataset(f"ke", (num_data_points, ),
                                 dtype='f4')

        for key, value in tw_args.items():
            self.file.attrs[key] = value

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
        for (tw_key, omm_key) in omm_state_input_keys.items():
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
            forces = state.getForces(asNumpy=True).in_units_of(
                self.energy_units/self.length_units)._value
            forces = forces[self.target_atom_indices, :]
            self.file["forces"][frame] = forces
        if self.omm_state_inputs["getPositions"]:
            positions = state.getPositions(
                asNumpy=True).in_units_of(self.length_units)._value
            positions = positions[self.target_atom_indices, :]
            self.file["positions"][frame] = positions
        if self.omm_state_inputs["getVelocities"]:
            velocities = state.getVelocities(
                asNumpy=True).in_units_of(self.length_units/self.time_units)._value
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
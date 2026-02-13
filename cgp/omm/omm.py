# ruff: noqa: F403, F405
from openmm import *
from openmm.app import *
import openmm.unit
import openmm.app
import warnings
import cgp.omm.custom_integrator as custom_integrator
from .utils import TrajWriter
import numpy as np


class Protein:
    def __init__(
        self,
        topology,
        system,
        positions,
        simulation_args,
        chkpt_filename=None,
        save_filename="",
    ):
        temperature = simulation_args["temperature"] * getattr(
            openmm.unit, simulation_args["temperature_units"]
        )
        friction = simulation_args["friction"] / getattr(
            openmm.unit, simulation_args["time_units"]
        )
        dt = simulation_args["dt"] * getattr(openmm.unit, simulation_args["time_units"])
        self.beta = 1 / (
            temperature
            * openmm.unit.BOLTZMANN_CONSTANT_kB
            * openmm.unit.AVOGADRO_CONSTANT_NA
        )

        self.fixed_atom_indices = []
        if simulation_args["fix"] is not None:
            if simulation_args["fix"] == "backbone":
                for res in topology.residues():
                    if res.name not in ["HOH", "Na+", "Cl-"]:
                        for atom in res.atoms():
                            if atom.name in [
                                "N",
                                "CA",
                                "C",
                                "CH3",
                                "O",
                                "CB",
                                "HA",
                                "HA2",
                                "HA3",
                            ]:
                                system.setParticleMass(atom.index, 0 * openmm.unit.amu)
                                self.fixed_atom_indices.append(atom.index)
            elif simulation_args["fix"] == "cg":
                for res in topology.residues():
                    if res.name not in ["HOH", "Na+", "Cl-"]:
                        for atom in res.atoms():
                            if atom.name in ["N", "CA", "C"]:
                                system.setParticleMass(atom.index, 0 * openmm.unit.amu)
                                self.fixed_atom_indices.append(atom.index)
            else:
                raise ValueError("Incorrect fix supplied")

        if simulation_args["integrator_to_use"] == "ovrvo":
            integrator = custom_integrator.get_ovrvo_integrator(
                temperature, friction, dt
            )
        elif simulation_args["integrator_to_use"] == "verlet":
            integrator = custom_integrator.get_verlet_integrator(
                temperature, friction, dt
            )
        elif simulation_args["integrator_to_use"] == "overdamped":
            integrator = custom_integrator.get_overdamped_integrator(
                temperature, friction, dt
            )
        elif simulation_args["integrator_to_use"] == "overdamped_custom_noise":
            integrator = custom_integrator.get_overdamped_integrator_custom_noise(
                temperature, friction, dt
            )
        else:
            raise ValueError("Incorrect integrator supplied")

        platform = Platform.getPlatformByName(simulation_args["device"])
        self.simulation = Simulation(topology, system, integrator, platform)

        if chkpt_filename is None:
            self.simulation.context.setPositions(positions)
            if simulation_args["do_energy_minimization"]:
                self.simulation.minimizeEnergy()
            self.simulation.context.setVelocitiesToTemperature(temperature)
        else:
            self.simulation.loadCheckpoint(chkpt_filename)

        self.target_atom_indices = self._get_target_atom_indices()
        all_reporters = [
            CheckpointReporter(f"{save_filename}.chk", simulation_args["chk_freq"])
        ]
        for reporter in all_reporters:
            self.simulation.reporters.append(reporter)
        self.temperature = temperature
        self.save_filename = save_filename

    def _get_target_atom_indices(self):
        """Gets the indices of all non H2O atoms
        Returns:
            all_atom_indices: The indices of all non water atoms
        """
        all_atom_indices = []
        for residue in self.simulation.topology.residues():
            if residue.name not in ["HOH", "Na+", "Cl-", "T3P", "NA", "CL"]:
                for atom in residue.atoms():
                    all_atom_indices.append(atom.index)
        return all_atom_indices

    def update_positions_and_velocities(
        self, positions, velocities, length_units, time_units
    ):
        """Updates position and velocities of the system
        Arguments:
            positions: A numpy array of shape (n_atoms, 3) corresponding to the positions in angstroms
            velocities: A numpy array of shape (n_atoms, 3) corresponding to the velocities in angstroms/ps
            length_units (str): The units for length
            time_units (str): The units for time
        """
        length_units = getattr(openmm.unit, length_units)
        time_units = getattr(openmm.unit, time_units)
        if positions is None:
            pass
        else:
            positions = positions * length_units
            self.simulation.context.setPositions(positions)
        if velocities is None:
            pass
        elif velocities is True:
            self.simulation.context.setVelocitiesToTemperature(self.temperature)
        else:
            velocities = velocities * (length_units / time_units)
            self.simulation.context.setVelocities(velocities)

    def get_information(
        self,
        length_units,
        time_units,
        energy_units,
        as_numpy=True,
        enforce_periodic_box=True,
    ):
        """Gets information (positions, forces and PE of system)
        Arguments:
            as_numpy: A boolean of whether to return as a numpy array
            enforce_periodic_box: A boolean of whether to enforce periodic boundary conditions
        Returns:
            positions: A numpy array of shape (n_atoms, 3) corresponding to the positions in angstroms
            velocities: A numpy array of shape (n_atoms, 3) corresponding to the velocities in angstroms/ps
            forces: A numpy array of shape (n_atoms, 3) corresponding to the force in kcal/mol*angstroms
            pe: A float coressponding to the potential energy in kcal/mol
            ke: A float coressponding to the kinetic energy in kcal/mol
        """
        state = self.simulation.context.getState(
            getForces=True,
            getEnergy=True,
            getPositions=True,
            getVelocities=True,
            enforcePeriodicBox=enforce_periodic_box,
        )

        length_units = getattr(openmm.unit, length_units)
        time_units = getattr(openmm.unit, time_units)

        positions = (
            state.getPositions(asNumpy=as_numpy).in_units_of(length_units)._value
        )
        velocities = (
            state.getVelocities(asNumpy=as_numpy)
            .in_units_of(length_units / time_units)
            ._value
        )
        forces = state.getForces(asNumpy=as_numpy).in_units_of(
            openmm.unit.kilojoules_per_mole / length_units
        )

        pe = state.getPotentialEnergy()
        ke = state.getKineticEnergy()

        if energy_units == "kT":
            pe = pe * self.beta
            ke = ke * self.beta
            forces = forces * self.beta
            forces = forces._value
        else:
            energy_units = getattr(openmm.unit, energy_units)
            pe = pe.in_units_of(energy_units)._value
            ke = ke.in_units_of(energy_units)._value
            forces = forces.in_units_of(energy_units / length_units)._value

        return positions, velocities, forces, pe, ke

    def relax_energies(
        self,
        positions,
        velocities=None,
        num_relax_steps=5,
        length_units="angstroms",
        time_units="picoseconds",
        energy_units="kT",
    ):
        """Carries out num_relax_steps of integration
        Arguments:
            num_relax_steps: Number of time steps of dynamics to run
        Returns:
            positions: A numpy array of shape (n_atoms, 3) corresponding to the positions in angstroms
            pe: A float coressponding to the potential energy in kT
            ke: A float coressponding to the kinetic energy in kT
        """
        self.update_positions_and_velocities(
            positions, velocities, length_units, time_units
        )
        self.simulation.step(num_relax_steps)
        positions, velocities, forces, pe, ke = self.get_information(
            length_units=length_units, time_units=time_units, energy_units=energy_units
        )
        return positions, pe, ke, forces

    def generate_trajectory(self, num_data_points, save_freq, tw_args):
        """ """
        writer = TrajWriter(
            filename=f"{self.save_filename}.hdf5",
            target_atom_indices=self.target_atom_indices,
            num_data_points=num_data_points + 1,
            tw_args=tw_args,
        )

        writer.write(self.simulation, frame=0)
        for i in range(num_data_points):
            self.simulation.step(save_freq)
            writer.write(self.simulation, frame=i + 1)
        writer.close()


class ProteinVacuum(Protein):
    def __init__(
        self, filename, chk, simulation_args, solvent_args=None, save_filename=None
    ):
        prmtop = AmberPrmtopFile(f"{filename}.prmtop")
        system = prmtop.createSystem()
        topology = prmtop.topology

        if os.path.exists(f"{filename}.crd"):
            inpcrd = AmberInpcrdFile(f"{filename}.crd")
            positions = inpcrd.getPositions(asNumpy=True)
        elif os.path.exists(f"{filename}.inpcrd"):
            inpcrd = AmberInpcrdFile(f"{filename}.inpcrd")
            positions = inpcrd.getPositions(asNumpy=True)
        else:
            positions = np.zeros((topology._numAtoms, 3))

        if save_filename is None:
            save_filename = filename
        if chk == 0:
            chkpt_filename = None
        else:
            chkpt_filename = f"{save_filename}_{chk - 1}.chk"
        save_filename = f"{save_filename}_{chk}"

        super().__init__(
            topology=topology,
            system=system,
            positions=positions,
            simulation_args=simulation_args,
            chkpt_filename=chkpt_filename,
            save_filename=save_filename,
        )


class ProteinImplicit(Protein):
    def __init__(
        self, filename, chk, simulation_args, solvent_args, save_filename=None
    ):
        warnings.warn("Check all Implicit solvent parameters (e.g. solvent)")
        prmtop = AmberPrmtopFile(f"{filename}.prmtop")

        system = prmtop.createSystem(
            implicitSolvent=getattr(openmm.app, solvent_args["implicit_solvent"]),
            implicitSolventKappa=(
                solvent_args["implicit_solvent_kappa"]
                / getattr(
                    openmm.unit, solvent_args["implicit_solvent_kappa_length_units"]
                )
            ),
        )
        topology = prmtop.topology

        if os.path.exists(f"{filename}.crd"):
            inpcrd = AmberInpcrdFile(f"{filename}.crd")
            positions = inpcrd.getPositions(asNumpy=True)
        elif os.path.exists(f"{filename}.inpcrd"):
            inpcrd = AmberInpcrdFile(f"{filename}.inpcrd")
            positions = inpcrd.getPositions(asNumpy=True)
        else:
            positions = np.zeros((topology._numAtoms, 3))

        if save_filename is None:
            save_filename = filename
        if chk == 0:
            chkpt_filename = None
        else:
            chkpt_filename = f"{save_filename}_{chk - 1}.chk"
        save_filename = f"{save_filename}_{chk}"

        super().__init__(
            topology=topology,
            system=system,
            positions=positions,
            simulation_args=simulation_args,
            chkpt_filename=chkpt_filename,
            save_filename=save_filename,
        )


class ProteinSolvent(Protein):
    def __init__(
        self, filename, chk, simulation_args, solvent_args, save_filename=None
    ):

        prmtop = AmberPrmtopFile(f"{filename}.prmtop")
        inpcrd = AmberInpcrdFile(f"{filename}.crd")

        nonbonded_cutoff = solvent_args["nonbonded_cutoff"] * getattr(
            openmm.unit, solvent_args["nonbonded_cutoff_length_units"]
        )
        switch_width = solvent_args["switch_width"] * getattr(
            openmm.unit, solvent_args["switch_width_length_units"]
        )

        constraints = (
            getattr(openmm.app, solvent_args["constraints"])
            if solvent_args["constraints"] is not None
            else None
        )

        system = prmtop.createSystem(
            constraints=constraints,
            nonbondedMethod=getattr(openmm.app, solvent_args["nonbonded_method"]),
            nonbondedCutoff=nonbonded_cutoff,
            rigidWater=True,
            hydrogenMass=None,
        )

        topology = prmtop.topology

        system.addForce(
            MonteCarloBarostat(
                (
                    solvent_args["mcb_pressure"]
                    * getattr(openmm.unit, solvent_args["mcb_pressure_units"])
                ),
                (
                    solvent_args["mcb_temperature"]
                    * getattr(openmm.unit, solvent_args["mcb_temperature_units"])
                ),
                solvent_args["mcb_freq"],
            )
        )
        forces = {
            system.getForce(index).__class__.__name__: system.getForce(index)
            for index in range(system.getNumForces())
        }
        forces["NonbondedForce"].setUseDispersionCorrection(
            solvent_args["use_dispersion_correction"]
        )
        forces["NonbondedForce"].setEwaldErrorTolerance(solvent_args["ewald_tolerance"])
        forces["NonbondedForce"].setUseSwitchingFunction(
            solvent_args["use_switching_function"]
        )
        forces["NonbondedForce"].setSwitchingDistance((nonbonded_cutoff - switch_width))

        positions = inpcrd.getPositions(asNumpy=True)
        box_vectors = inpcrd.getBoxVectors(asNumpy=True)
        system.setDefaultPeriodicBoxVectors(*box_vectors)

        if save_filename is None:
            save_filename = filename
        if chk == 0:
            chkpt_filename = None
        else:
            chkpt_filename = f"{save_filename}_{chk - 1}.chk"
        save_filename = f"{save_filename}_{chk}"

        super().__init__(
            topology=topology,
            system=system,
            positions=positions,
            simulation_args=simulation_args,
            chkpt_filename=chkpt_filename,
            save_filename=save_filename,
        )

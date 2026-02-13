import torch
import math
import openmm.unit


class OVRVO:
    """
    https://pubs.acs.org/doi/pdf/10.1021/jp411770f
    pos in units of A
    forces in units of kcal/A
    """

    def __init__(
        self,
        u_model,
        masses,
        batch_size=1,
        dt=0.001,
        friction=10.0,
        temperature=300.0,
        length_units="angstroms",
        energy_units="kilocalories_per_mole",
        time_units="picoseconds",
        temperature_units="kelvin",
    ):
        self.u_model = u_model

        self.energy_units = energy_units
        self.length_units = length_units
        self.time_units = time_units
        self.temperature_units = temperature_units

        masses = 0.001 * torch.tensor(masses)  # Since 1 amu = 1 g/mol = 0.001 kg/mol
        unit_length_omm = 1.0 * getattr(openmm.unit, length_units)
        unit_energy_omm = 1.0 * getattr(openmm.unit, energy_units)
        unit_time_omm = 1.0 * getattr(openmm.unit, time_units)
        unit_temperature_omm = 1.0 * getattr(openmm.unit, temperature_units)
        unit_mass_omm = 1.0 * (
            (getattr(openmm.unit, "second") ** 2)
            * getattr(openmm.unit, "kilojoules")
            / (getattr(openmm.unit, "meters") ** 2)
            / getattr(openmm.unit, "mole")
        )

        friction_omm = friction / unit_time_omm
        dt_omm = dt * unit_time_omm
        temperature_omm = temperature * unit_temperature_omm
        beta_omm = 1 / (
            temperature_omm
            * openmm.unit.BOLTZMANN_CONSTANT_kB
            * openmm.unit.AVOGADRO_CONSTANT_NA
        )
        kT_omm = 1.0 / beta_omm

        masses_conversion_factor = unit_mass_omm.value_in_unit(
            (unit_energy_omm * unit_time_omm**2 / unit_length_omm**2).unit
        )
        self.masses = (
            masses * masses_conversion_factor
        )  # energy_units * time_units^2 / length_units^2
        num_atoms = len(masses)
        self.masses = self.masses.repeat_interleave(3, 0).reshape(num_atoms, 3)
        self.masses = (
            self.masses.repeat(batch_size, 1, 1)
            .reshape(batch_size * num_atoms, 3)
            .to(u_model.device)
        )

        self.temperature = temperature
        self.dt = dt_omm.value_in_unit(getattr(openmm.unit, time_units))
        self.kT = kT_omm.value_in_unit(getattr(openmm.unit, energy_units))
        self.friction_omm = friction_omm
        self.dt_omm = dt_omm
        self.t_rescale = torch.sqrt(
            (2 / (friction_omm * dt_omm))
            * torch.tanh(torch.tensor(friction_omm * dt_omm / 2))
        )  # unitless
        self.a = math.exp(-friction_omm * dt_omm)  # Unitless
        # self.a = self.a.repeat_interleave(3, 0).reshape(num_atoms, 3) # Unitless
        # self.a = self.a.repeat(batch_size, 1, 1).reshape(batch_size * num_atoms, 3)
        self.b = torch.sqrt(
            ((1 - self.a) * self.kT) / self.masses
        )  # length_units/time_units

    def ovrvo_step(self, x, v, f_prev):
        # time-step rescaling (b in paper) is set to be 1
        v = math.sqrt(self.a) * v + self.b * torch.randn_like(
            x
        )  # length_units/time_units
        v = (
            v + 0.5 * self.dt * (f_prev / self.masses) * self.t_rescale
        )  # length_units/time_units
        x = x + self.dt * v * self.t_rescale  # length_units
        f = self.u_model.get_force(x)  # energy_units/length_units
        f = f.detach()  # energy_units/length_units
        x = x.detach()  # length_units
        v = (
            v + 0.5 * self.dt * (f / self.masses) * self.t_rescale
        )  # length_units/time_units
        v = math.sqrt(self.a) * v + self.b * torch.randn_like(
            x
        )  # length_units/time_units
        return x, v, f

    def integrate(self, init_x, init_v, steps, writer, save_freq=100):
        x = init_x.to(self.u_model.device)  # length_units
        v = init_v.to(self.u_model.device)  # length_units/time_units
        f_prev = self.u_model.get_force(x)  # energy_units/length_units
        f_prev = f_prev.detach()  # energy_units/length_units
        x = x.detach()  # length_units
        for i in range(steps):
            if (i % save_freq) == 0:
                writer.write(x.cpu(), f_prev.cpu(), i // save_freq)
            x, v, f_prev = self.ovrvo_step(x, v, f_prev)
        writer.close()
        return x, v

    def update_temperature(self, temperature, temperature_units="kelvin"):
        """
        Update the temperature of the system.
        """
        assert temperature_units == self.temperature_units
        unit_temperature_omm = 1.0 * getattr(openmm.unit, temperature_units)
        temperature_omm = temperature * unit_temperature_omm
        beta_omm = 1 / (
            temperature_omm
            * openmm.unit.BOLTZMANN_CONSTANT_kB
            * openmm.unit.AVOGADRO_CONSTANT_NA
        )
        kT_omm = 1.0 / beta_omm
        self.kT = kT_omm.value_in_unit(getattr(openmm.unit, self.energy_units))
        self.b = torch.sqrt(((1 - self.a) * self.kT) / self.masses)
        self.t_rescale = torch.sqrt(
            (2 / (self.friction_omm * self.dt_omm))
            * torch.tanh(torch.tensor(self.friction_omm * self.dt_omm / 2))
        )
        self.temperature = temperature

    def heat_system(
        self,
        init_x,
        init_v,
        writer,
        save_freq,
        end_temperature,
        heating_steps,
        temperature_units="kelvin",
        time_units="picoseconds",
    ):
        assert temperature_units == self.temperature_units
        assert time_units == self.time_units
        temp_increase_per_step = (end_temperature - self.temperature) / heating_steps
        x = init_x.to(self.u_model.device)  # length_units
        v = init_v.to(self.u_model.device)  # length_units/time_units
        f_prev = self.u_model.get_force(x)  # energy_units/length_units
        f_prev = f_prev.detach()  # energy_units/length_units
        x = x.detach()  # length_units

        for i in range(heating_steps):
            if (i % save_freq) == 0:
                writer.write(x.cpu(), f_prev.cpu(), i // save_freq)
            x, v, f_prev = self.ovrvo_step(x, v, f_prev)
            self.update_temperature(
                self.temperature + temp_increase_per_step,
                temperature_units=temperature_units,
            )
        writer.close()
        return x, v

    def cool_system(
        self,
        init_x,
        init_v,
        writer,
        save_freq,
        end_temperature,
        cooling_steps,
        temperature_units="kelvin",
        time_units="picoseconds",
    ):
        assert temperature_units == self.temperature_units
        assert time_units == self.time_units
        temp_decrease_per_step = (self.temperature - end_temperature) / cooling_steps
        x = init_x.to(self.u_model.device)
        v = init_v.to(self.u_model.device)
        f_prev = self.u_model.get_force(x)
        f_prev = f_prev.detach()
        x = x.detach()

        for i in range(cooling_steps):
            if (i % save_freq) == 0:
                writer.write(x.cpu(), f_prev.cpu(), i // save_freq)
            x, v, f_prev = self.ovrvo_step(x, v, f_prev)
            self.update_temperature(
                self.temperature - temp_decrease_per_step,
                temperature_units=temperature_units,
            )
        writer.close()
        return x, v

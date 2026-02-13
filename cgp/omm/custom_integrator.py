import math
import numpy as np
from openmm import CustomIntegrator
from openmm.unit import BOLTZMANN_CONSTANT_kB, AVOGADRO_CONSTANT_NA


def _get_beta(temperature):
    beta = 1 / (temperature * BOLTZMANN_CONSTANT_kB * AVOGADRO_CONSTANT_NA)
    return beta


def get_overdamped_integrator_custom_noise(temperature, friction, dt):
    """Creates OpenMM integrator to carry out overdamped Brownian integration
    Arguments:
        temperature: temperature with OpenMM units
        friction: friction coefficient with OpenMM units
        dt: time step with OpenMM units
    Returns:
        overdamped_integrator: OpenMM Integrator
    """

    overdamped_integrator = CustomIntegrator(dt)
    overdamped_integrator.addGlobalVariable("kT", 1 / _get_beta(temperature))
    overdamped_integrator.addGlobalVariable("friction", friction)
    overdamped_integrator.addPerDofVariable("eta", 0)

    overdamped_integrator.addUpdateContextState()
    overdamped_integrator.addComputePerDof(
        "x", "x+dt*f/(m*friction) + eta*sqrt(2*kT*dt/(m*friction))"
    )
    return overdamped_integrator


def get_overdamped_integrator(temperature, friction, dt):
    """Creates OpenMM integrator to carry out overdamped Brownian integration
    Arguments:
        temperature: temperature with OpenMM units
        friction: friction coefficient with OpenMM units
        dt: time step with OpenMM units
    Returns:
        overdamped_integrator: OpenMM Integrator
    """

    overdamped_integrator = CustomIntegrator(dt)
    overdamped_integrator.addGlobalVariable("kT", 1 / _get_beta(temperature))
    overdamped_integrator.addGlobalVariable("friction", friction)

    overdamped_integrator.addUpdateContextState()
    overdamped_integrator.addComputePerDof(
        "x", "x+dt*f/(m*friction) + gaussian*sqrt(2*kT*dt/(m*friction))"
    )
    return overdamped_integrator


def get_verlet_integrator(temperature, friction, dt):
    """Creates OpenMM integrator to carry out Verlet integration
    Arguments:
        temperature: temperature with OpenMM units
        friction: friction coefficient with OpenMM units
        dt: time step with OpenMM units
    Returns:
        verlet_integrator: OpenMM Integrator
    """

    verlet_integrator = CustomIntegrator(dt)
    verlet_integrator.addComputePerDof("v", "v+0.5*dt*f/m")
    verlet_integrator.addComputePerDof("x", "x+dt*v")
    verlet_integrator.addUpdateContextState()
    verlet_integrator.addComputePerDof("v", "v+0.5*dt*f/m")
    return verlet_integrator


def get_ovrvo_integrator(temperature, friction, dt):
    """Creates OpenMM integrator to carry out ovrvo integration (Sivak, Chodera, and Crooks 2014)
    Arguments:
        temperature: temperature with OpenMM units
        friction: friction coefficient with OpenMM units
        dt: time step with OpenMM units
    Returns:
        ovrvo_integrator: OpenMM Integrator
    """

    ovrvo_integrator = CustomIntegrator(dt)
    ovrvo_integrator.setConstraintTolerance(1e-8)
    ovrvo_integrator.addGlobalVariable("a", math.exp(-friction * dt / 2))
    ovrvo_integrator.addGlobalVariable("b", np.sqrt(1 - np.exp(-2 * friction * dt / 2)))
    ovrvo_integrator.addGlobalVariable("kT", 1 / _get_beta(temperature))
    ovrvo_integrator.addPerDofVariable("x1", 0)

    ovrvo_integrator.addComputePerDof("v", "(a * v) + (b * sqrt(kT/m) * gaussian)")
    ovrvo_integrator.addConstrainVelocities()

    ovrvo_integrator.addComputePerDof("v", "v + 0.5*dt*(f/m)")
    ovrvo_integrator.addConstrainVelocities()

    ovrvo_integrator.addComputePerDof("x", "x + dt*v")
    ovrvo_integrator.addComputePerDof("x1", "x")
    ovrvo_integrator.addConstrainPositions()
    ovrvo_integrator.addComputePerDof("v", "v + (x-x1)/dt")
    ovrvo_integrator.addConstrainVelocities()
    ovrvo_integrator.addUpdateContextState()

    ovrvo_integrator.addComputePerDof("v", "v + 0.5*dt*f/m")
    ovrvo_integrator.addConstrainVelocities()
    ovrvo_integrator.addComputePerDof("v", "(a * v) + (b * sqrt(kT/m) * gaussian)")
    ovrvo_integrator.addConstrainVelocities()
    return ovrvo_integrator

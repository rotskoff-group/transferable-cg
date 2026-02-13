import h5py
import torch
import numpy as np
import random
import time


class TrajWriter:
    def __init__(self, filename, batch_size, num_atoms, num_data_points):
        self.batch_size = batch_size
        self.num_atoms = num_atoms
        self.filename = filename
        self.file = h5py.File(self.filename, "w")
        self.file.create_dataset(
            "forces", (num_data_points, batch_size, num_atoms, 3), dtype="f4"
        )
        self.file.create_dataset(
            "positions", (num_data_points, batch_size, num_atoms, 3), dtype="f4"
        )

    def write(self, positions, forces, frame):
        self.file["positions"][frame] = positions.reshape(
            self.batch_size, self.num_atoms, 3
        )
        self.file["forces"][frame] = forces.reshape(self.batch_size, self.num_atoms, 3)
        self.file.flush()

    def close(self):
        self.file.close()


def generate_trajectory(
    integrator,
    number_atoms,
    batch_size,
    num_data_points,
    save_freq,
    chk_freq,
    start_chk=0,
    save_filename="./dataset",
    init_x=None,
    init_v=None,
):
    scale = (integrator.kT / integrator.masses) ** 0.5
    if init_x is None:
        init_x = torch.randn(batch_size * number_atoms, 3)
    init_x = init_x.to(torch.float)  # make sure init_x is type float
    init_x = init_x.to(integrator.u_model.device)
    if init_v is None:
        init_v = torch.randn_like(init_x) * scale
    init_v = init_v.to(torch.float)  # make sure init_v is type float
    init_v = init_v.to(integrator.u_model.device)

    num_checkpoints = num_data_points // chk_freq
    for i in range(start_chk, num_checkpoints):
        filename_chk = f"{save_filename}_{i}.hdf5"
        writer = TrajWriter(filename_chk, batch_size, number_atoms, chk_freq)
        start = time.time()
        init_x, init_v = integrator.integrate(
            init_x, init_v, chk_freq * save_freq, writer, save_freq
        )
        end = time.time()
        print(f"Checkpoint {i} integration took {end - start} seconds.")

        ## Write checkpoint
        torch.save(
            {
                "positions": init_x.cpu(),
                "velocities": init_v.cpu(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all()
                if torch.cuda.is_available()
                else None,
                "numpy": np.random.get_state(),
                "python": random.getstate(),
                "chk_wall_time": end - start,
            },
            f"{save_filename}_{i}_chk.pt",
        )

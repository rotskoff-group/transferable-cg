## Usage
### Overview
Runs parallel coarse-grained (CG) molecular dynamics simulations using a trained model on a single GPU.

### Basic Command Structure

The `cg_sim` command uses Hydra configuration overrides. Each argument follows the pattern `config.key=value`:
```bash
cg_sim "global_args.param1=value1" "global_args.param2=value2" ...
```

### Example: Running a Simulation
```bash
cg_sim \
  "global_args.model_folder=model_folder/" \
  "global_args.save_folder_name=save_folder/" \
  "global_args.bias_force=path/to/bias_force.pt" \
  "global_args.start_indices=path/to/batch.npy" \
  "global_args.save_freq=10" \
  "global_args.chk_freq=1000" \
  "global_args.num_data_points=1000" \
  "integrator_args.friction=1" \
  "integrator_args.dt=0.2" \
  "integrator_args.temperature=300" \
  "global_args.pdb_file=path/to/reference.pdb" \
  "global_args.start_positions=path/to/start_positions.npy"
```

### Configuration Parameters

#### Global Arguments (`global_args.*`)

| Parameter | Type | Description |
|-----------|------|-------------|
| `model_folder` | path | Directory containing the trained model |
| `save_folder_name` | path | Output directory where simulation results will be saved |
| `bias_force` | path | Bias force for biased simulations (must be a TorchScript JIT model)<br>**For unbiased simulations:** Set to `null` |
| `pdb_file` | path | Reference PDB structure defining atom ordering and topology |
| `start_positions` | path | Initial atomic positions as NumPy array (.npy)<br>**Shape:** `(batch_size, num_atoms, 3)`<br>**Units:** Must match CG model units<br>**Ordering:** Must correspond to `pdb_file` atom order<br>**Default:** `null` (generates random positions with batch size 1)|
| `start_indices` | path | Batch indices (.npy) specifying which positions from `start_positions` to use<br>**Default:** `null` (uses all positions) |
| `save_freq` | int | Save trajectory frame every N steps |
| `chk_freq` | int | Save checkpoint every N frames for simulation recovery<br>**Note:** Interrupted simulations automatically restart from last checkpoint  |
| `num_data_points` | int | Number of trajectory frames to save<br>**Total simulations steps:** `num_data_points × save_freq` |


#### Integrator Arguments (`integrator_args.*`)

| Parameter | Type | Description |
|-----------|------|-------------|
| `dt` | float | Integration timestep of specified time unit |
| `friction` | float | Friction coefficient for Langevin dynamics |
| `temperature` | float | Simulation temperature |
| `length_units` | str | Length units of CG model |
| `energy_units` | str | Energy units of CG model |
| `time_units` | str | Time units of dt |
| `temperature_units` | str | Temperature units |

## Output Files

The simulation creates the following files in `save_folder_name/`:
```
save_folder_name/
├── cg_dataset_*.hdf5      # Trajectory frames
└── cg_dataset_*_chk.pt    # Checkpoints
```

| File Pattern | Format | Description |
|--------------|--------|-------------|
| `cg_dataset_*.hdf5` | HDF5 | Trajectory frames containing positions and metadata. New file every `chk_freq` frames |
| `cg_dataset_*_chk.pt` | PyTorch | Checkpoint for simulation recovery and restart. New file every `chk_freq` frames |

**Note:** The `*` in filenames is replaced with an incremental counter (e.g., `cg_dataset_0.hdf5`, `cg_dataset_1.hdf5`, etc.)
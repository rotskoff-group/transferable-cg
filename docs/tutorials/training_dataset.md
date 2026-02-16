# Atomistic Dataset Documentation

## Download

All atomistic data used to train CG models can be downloaded from:  
 **https://huggingface.co/datasets/abpark/transferable-cg**

---

## Dataset Overview

The data repository contains four trajectory datasets plus metadata files for train/test splits:

```
transferable-cg/
├── MFM/
├── MFM_100K/
├── MFM_test/
├── instantaneous_md/
├── training_cath_1000.npy
└── test_cath_50.npy
```


### Dataset Descriptions

| Dataset | Structures per Domain | Number of Domains | Purpose |
|---------|----------------------|---------|---------|
| **`MFM`** | Up to 20 | 1,000 | Mean force estimates (training)|
| **`MFM_100K`** | Up to 100 | 1,000 | Mean force estimates (training, extended) |
| **`MFM_test`** | Up to 100 | 50 | Mean force estimates (held-out test set) |
| **`instantaneous_md`** | Up to 20 simulations | 1,000 | Concatenated MD trajectories |

#### Important Notes on Mean Force Datasets

> ⚠️ **For `MFM`, `MFM_100K`, and `MFM_test` datasets:**
> 
> - Forces are recorded **only for backbone atoms** (Cα, C, N) of each residue and represent the estimated mean force on the Cα, C, and N for each structure
> - Forces for all other atoms are set to **zero** and should be **ignored**
> - Only positions/forces for Cα, C, and N atoms contain valid data

### File Descriptions

| File | Description |
|------|-------------|
| **`training_cath_1000.npy`** | List of 1000 CATH domains IDs used for training |
| **`test_cath_50.npy`** | List of 50 CATH domains held out for testing |

---

## File Structure

Each dataset folder contains subfolders organized by CATH domain ID:

```
{Dataset}/
└── {CATH_ID}/
    ├── {CATH_ID}_traj_all.hdf5
    └── {CATH_ID}.pdb
    └── subset_1000_cg.npy        # Only in instantaneous_md/
```

### Per-Domain Files

| File | Description | Datasets |
|------|-------------|----------|
| **`{CATH_ID}.pdb`** | Reference structure defining atom ordering and topology | All |
| **`{CATH_ID}_traj_all.hdf5`** | Trajectory data and simulation metadata in HDF5 format | All |
| **`subset_1000_cg.npy`** | Indices of frames used for training | `instantaneous_md` only |

---

## HDF5 File Contents

### Datasets

Each HDF5 file contains the following trajectory data:

| Dataset | Shape | Description |
|---------|-------|-------------|
| `positions` | `(n_frames, n_atoms, 3)` | Atomic coordinates |
| `forces` | `(n_frames, n_atoms, 3)` | Forces on atoms |
| `velocities` | `(n_frames, n_atoms, 3)` | Atomic velocities |
| `pe` | `(n_frames,)` | Potential energy |
| `ke` | `(n_frames,)` | Kinetic energy |

### Attributes (Metadata)

Each HDF5 file includes the following attributes:

| Attribute | Type | Description |
|-----------|------|-------------|
| `energy_units` | `str` | Units for energy values |
| `length_units` | `str` | Units for positions/distances |
| `time_units` | `str` | Units for time |
| `save_energy` | `bool` | If `False`, energy arrays contain zeros |
| `save_forces` | `bool` | If `False`, force arrays contain zeros |
| `save_positions` | `bool` | If `False`, position arrays contain zeros |
| `save_velocities` | `bool` | If `False`, velocity arrays contain zeros |

---

# CG Dataset Construction

## MFM training dataset

To create the training dataset we used to train MFM models, run the following command:

```bash
root_data_folder=${"path/to/transferable-cg"} # Path to downloaded dataset
save_folder=${"path/to/save/folder"}  # Output location

uv run make_u_dataset "global_args.root_data_folder_name=${root_data_folder}/MFM/"\
    "cg.cg_model_args.protein_name=null"\
    "global_args.root_save_folder_name=${root_save_folder}"\
    "global_args.multiple_protein_names=${root_data_folder}/training_cath_1000.npy"\
    "cg.cg_model_args.dataset_subset_indices_filename=null"\
    "global_args.is_individual_dataset_made=False"\
    "global_args.concatenate_datasets=True"

```

## FM and SM training dataset

To create the training dataset we used to train FM and SM models, run the following command in the transferable-cg codebase repo:

```bash
root_data_folder=${"path/to/transferable-cg"} # Path to downloaded dataset
save_folder=${"path/to/save/folder"}  # Output location

uv run make_u_dataset "global_args.root_data_folder_name=${root_data_folder}/instantaneous_md/"\
    "cg.cg_model_args.protein_name=null"\
    "global_args.root_save_folder_name=${root_save_folder}"\
    "global_args.multiple_protein_names=${root_data_folder}/training_cath_1000.npy"\
    "cg.cg_model_args.dataset_subset_indices_filename=subset_1000_cg.npy"\
    "global_args.is_individual_dataset_made=False"\
    "global_args.concatenate_datasets=True"

```

> **Note:** Setting `concatenate_datasets=True` creates a single large file that is not space-efficient, but we recommend this for more efficient training performance.
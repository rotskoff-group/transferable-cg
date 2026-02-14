# make_u_dataset

## Usage

### Overview
Creates coarse-grained (CG) datasets from atomistic trajectory data for training.

### Basic Command Structure

The `make_u_dataset` command uses Hydra configuration overrides. Each argument follows the pattern `config.key=value`:

```bash
uv run make_u_dataset "category.parameter=value" "category.parameter2=value2" ...
```

### Example: Creating a Dataset

```bash
uv run make_u_dataset \
  "cg.cg_model_args.protein_name=my_protein" \
  "global_args.root_data_folder_name=./raw_data/" \
  "global_args.root_save_folder_name=./processed_data/" \
  "cg.cg_model_args.cg_type=backbone"
```

## Configuration Reference

### Global Arguments (`global_args.*`)

#### Directory Configuration

| Parameter | Type | Description | Options |
|-----------|------|-------------|---------|
| `root_data_folder_name` | path | Root directory containing raw trajectory data<br>**Default:** `"./"` | Valid directory path |
| `root_save_folder_name` | path | Root directory where processed datasets will be saved<br>**Default:** `"./"` | Valid directory path |

#### Single vs. Multiple Proteins

| Parameter | Type | Description | Options |
|-----------|------|-------------|---------|
| `multiple_protein_names` | path | Path to `.npy` file containing list of protein names to process<br>**Default:** `null` (process single protein) | Valid `.npy` file path or `null` |
| `is_individual_dataset_made` | bool | Whether individual protein datasets already exist (only used with `multiple_protein_names`)<br>**Default:** `true` | `true`, `false` |

#### Dataset Concatenation

| Parameter | Type | Description | Options |
|-----------|------|-------------|---------|
| `concatenate_datasets` | bool | Concatenate all individual protein datasets into one combined dataset<br>**Default:** `false` | `true`, `false` |

---

### CG Model Configuration (`cg.*`)

### CG Model Arguments (`cg.cg_model_args.*`)

#### Basic Configuration

| Parameter | Type | Description | Options |
|-----------|------|-------------|---------|
| `cg_type` | str | Type of coarse-graining to apply<br>**Default:** `"backbone"` | `"backbone"`, `"c_alpha"` |
| `protein_name` | str | Name of the protein to process (for single protein mode)<br>**Default:** `null` | Protein identifier or `null` |
| `dataset_subset_indices_filename` | path | Path to file containing indices for dataset subsetting<br>**Default:** `null` | Valid file path or `null` |

#### Units

| Parameter | Type | Description | Options |
|-----------|------|-------------|---------|
| `cg_length_units` | str | Length units for the CG model<br>**Default:** `"angstroms"` | `"angstroms"`, `"nanometers"` |
| `cg_energy_units` | str | Energy units for the CG model<br>**Default:** `"kilocalories_per_mole"` | `"kilocalories_per_mole"`, `"kilojoules_per_mole"` |

---

### Storage Configuration (`cg.cg_model_args.store_in_dataset_args.*`)

Controls what information is stored in the generated dataset.

| Parameter | Type | Description | Options |
|-----------|------|-------------|---------|
| `store_positions` | bool | Store CG bead positions<br>**Default:** `true` | `true`, `false` |
| `store_forces` | bool | Store forces on CG bead<br>**Default:** `true` | `true`, `false` |
| `store_energies` | bool | Store potential energies<br>**Default:** `true` | `true`, `false` |
| `store_features` | bool | Store general features<br>**Default:** `false` | `true`, `false` |
| `store_bond_distance_features` | bool | Store bond distance features<br>**Default:** `true` | `true`, `false` |
| `store_bond_angle_features` | bool | Store bond angle features<br>**Default:** `true` | `true`, `false` |
| `store_dihedral_angle_features` | bool | Store dihedral angle features<br>**Default:** `true` | `true`, `false` |
| `store_nonbonded_features` | bool | Store non-bonded interaction features<br>**Default:** `true` | `true`, `false` |
| `store_atom_features` | bool | Store atom-level features<br>**Default:** `true` | `true`, `false` |

---

### Feature Arguments (`cg.cg_model_args.feature_args`)

| Parameter | Type | Description | Options |
|-----------|------|-------------|---------|
| `feature_args` | dict | Additional feature computation arguments<br>**Default:** `null` | Feature-specific arguments or `null` |

---

### Non-bonded Feature Arguments (`cg.cg_model_args.nonbonded_feature_args.*`)

| Parameter | Type | Description | Options |
|-----------|------|-------------|---------|
| `cutoff` | float | Cutoff distance for non-bonded interactions<br>**Default:** `20` | Positive float |
| `cutoff_units` | str | Units for cutoff distance<br>**Default:** `"angstroms"` | `"angstroms"`, `"nanometers"` |

---

## Processing Modes

### Mode 1: Single Protein Dataset

Process a single protein and create its dataset.

**Requirements:**
- `cg.cg_model_args.protein_name` must be set
- `multiple_protein_names` must be `null`

**Example:**
```bash
uv run make_u_dataset \
  "cg.cg_model_args.protein_name=protein_A" \
  "global_args.root_data_folder_name=./raw_data/" \
  "global_args.root_save_folder_name=./datasets/" \
  "cg.cg_model_args.cg_type=backbone"
```

**Behavior:**
- Creates dataset at `{root_save_folder_name}/{protein_name}/dataset.hdf5`
- Saves configuration at `{root_save_folder_name}/{protein_name}/dataset_config.yaml`

**Output Structure:**
```
datasets/
└── protein_A/
    ├── dataset.hdf5
    └── dataset_config.yaml
```

---

### Mode 2: Multiple Proteins (Create Individual Datasets)

Process multiple proteins and create individual datasets for each.

**Requirements:**
- `multiple_protein_names` points to `.npy` file with protein names
- `is_individual_dataset_made=false`
- `cg.cg_model_args.protein_name=null`

**Example:**
```bash
uv run make_u_dataset \
  "global_args.multiple_protein_names=./protein_list.npy" \
  "global_args.is_individual_dataset_made=false" \
  "global_args.root_data_folder_name=./raw_data/" \
  "global_args.root_save_folder_name=./datasets/" \
  "cg.cg_model_args.cg_type=backbone"
```

**Behavior:**
- Iterates through each protein in the list
- Creates individual dataset for each protein
- Saves individual configs for each protein
- Saves master config at `{root_save_folder_name}/dataset_config.yaml`

**Output Structure:**
```
datasets/
├── protein_A/
│   ├── dataset.hdf5
│   └── dataset_config.yaml
├── protein_B/
│   ├── dataset.hdf5
│   └── dataset_config.yaml
└── dataset_config.yaml
```

---

### Mode 3: Multiple Proteins (Concatenate Datasets)

Process multiple proteins and concatenate them into a single combined dataset.

**Requirements:**
- `multiple_protein_names` points to `.npy` file with protein names
- `concatenate_datasets=true`

**Example (Create and Concatenate):**
```bash
uv run make_u_dataset \
  "global_args.multiple_protein_names=./protein_list.npy" \
  "global_args.is_individual_dataset_made=false" \
  "global_args.concatenate_datasets=true" \
  "global_args.root_data_folder_name=./raw_data/" \
  "global_args.root_save_folder_name=./datasets/"
```

**Example (Concatenate Existing Datasets):**
```bash
uv run make_u_dataset \
  "global_args.multiple_protein_names=./protein_list.npy" \
  "global_args.is_individual_dataset_made=true" \
  "global_args.concatenate_datasets=true" \
  "global_args.root_data_folder_name=./datasets/" \
  "global_args.root_save_folder_name=./combined_dataset/"
```

**Behavior:**
- If `is_individual_dataset_made=false`: Creates individual datasets first, then concatenates
- If `is_individual_dataset_made=true`: Uses existing datasets from `{root_data_folder_name}/{protein_name}/dataset.hdf5`
- Creates combined dataset at `{root_save_folder_name}/dataset.hdf5`
- Saves master config at `{root_save_folder_name}/dataset_config.yaml`

**Output Structure:**
```
combined_dataset/
├── dataset.hdf5           # Combined dataset
└── dataset_config.yaml
```

---

## Notes

### Data Structure Requirements

The script expects raw trajectory data to be organized as:
```
{root_data_folder_name}/{protein_name}/
├── {protein_name}_traj_all.hdf5           
└── {protein_name}.pdb  
```
- Single protein: `{root_data_folder_name}/{protein_name}/` (containing trajectory files)
- Multiple proteins: Each protein in its own subdirectory under `root_data_folder_name`

### Protein Names File

The `.npy` file for `multiple_protein_names` should contain a NumPy array of protein name strings:
```python
import numpy as np
protein_names = np.array(['protein_A', 'protein_B', 'protein_C'])
np.save('protein_list.npy', protein_names)
```

### Output Format

- Datasets are saved in HDF5 format (`.hdf5`)
- Configuration files are saved in YAML format (`.yaml`)
- Individual and master configurations are both saved for multi-protein processing
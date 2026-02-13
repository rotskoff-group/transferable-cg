# u_test

## Usage

### Overview
Evaluates trained neural network potential models on test datasets using PyTorch Lightning. Can test on either the original training dataset's test split or on new datasets.

### Basic Command Structure

The `u_test` command uses Hydra configuration overrides. Each argument follows the pattern `config.key=value`:

```bash
u_test "category.parameter=value" "category.parameter2=value2" ...
```

### Example: Testing a Model

```bash
u_test \
  "global_args.config_path=./lightning_logs/version_0/config.yaml" \
  "global_args.ckpt_path=./lightning_logs/version_0/checkpoints/best_model.ckpt" \
  "global_args.use_training_dataset=true"
```

## Configuration Reference

### Global Arguments (`global_args.*`)

#### Required

| Parameter | Type | Description | Options |
|-----------|------|-------------|---------|
| `config_path` | path | Path to the configuration file from training<br>**Default:** `""` (must be set) | Valid `.yaml` file path from training run |
| `ckpt_path` | path | Path to model checkpoint to evaluate<br>**Default:** `""` (must be set) | Valid `.ckpt` file path |

#### Testing Mode

| Parameter | Type | Description | Options |
|-----------|------|-------------|---------|
| `use_training_dataset` | bool | Use the test split from the original training dataset<br>**Default:** `true` | `true`, `false` |
| `dataset_folder_name` | path | Directory containing new test data (only used if `use_training_dataset=false`)<br>**Default:** `null` | Valid directory path or `null` |

#### Hardware Configuration

| Parameter | Type | Description | Options |
|-----------|------|-------------|---------|
| `accelerator` | str | Hardware accelerator for testing<br>**Default:** `"cuda"` | `"cuda"`, `"cpu"`, `"mps"` |

#### Per-Protein Evaluation

| Parameter | Type | Description | Options |
|-----------|------|-------------|---------|
| `individual_protein_datasets` | path | Path to `.npy` file containing list of protein names for individual evaluation<br>**Default:** `null` | Valid `.npy` file path or `null` |
| `report_metric_per_protein_save_folder` | path | Directory to save per-protein metrics <br>**Default:** `null` | Valid directory path or `null` |

---

## Testing Modes

### Mode 1: Test on Training Dataset Split

Evaluates the model on the test split defined during training.

**Requirements:**
- The original training configuration must have `dataset.dataset_split_args.test > 0`
- `use_training_dataset=true`

**Example:**
```bash
u_test \
  "global_args.config_path=./lightning_logs/version_0/config.yaml" \
  "global_args.ckpt_path=./lightning_logs/version_0/checkpoints/best_model.ckpt" \
  "global_args.use_training_dataset=true"
```

**Behavior:**
- Uses the same dataset and split configuration from training
- Ensures reproducible test set using the same random seed
- Reports test metrics to TensorBoard

---

### Mode 2: Test on New Dataset

Evaluates the model on a completely new dataset not used during training.

**Requirements:**
- `use_training_dataset=false`
- `dataset_folder_name` must be set to a valid dataset path

**Example:**
```bash
u_test \
  "global_args.config_path=./lightning_logs/version_0/config.yaml" \
  "global_args.ckpt_path=./lightning_logs/version_0/checkpoints/best_model.ckpt" \
  "global_args.use_training_dataset=false" \
  "global_args.dataset_folder_name=./data/new_test_set/"
```

**Behavior:**
- All data in `dataset_folder_name` is used for testing (100% test split)
- Batch size is automatically set to 1
- Dataset split ratios are set to `train=0.0`, `val=0.0`, `test=1.0`

---

### Mode 3: Per-Protein Evaluation

Evaluates the model separately on individual proteins and saves per-protein metrics.

**Requirements:**
- `individual_protein_datasets` must point to a `.npy` file containing protein names
- `report_metric_per_protein_save_folder` must be set
- Dataset structure: `dataset_folder_name/{protein_name}/`

**Example:**
```bash
u_test \
  "global_args.config_path=./lightning_logs/version_0/config.yaml" \
  "global_args.ckpt_path=./lightning_logs/version_0/checkpoints/best_model.ckpt" \
  "global_args.use_training_dataset=false" \
  "global_args.dataset_folder_name=./data/proteins/" \
  "global_args.individual_protein_datasets=./data/protein_list.npy" \
  "global_args.report_metric_per_protein_save_folder=./per_protein_results/"
```

**Behavior:**
- Iterates through each protein in the list
- Tests model on each protein's dataset separately
- Saves results to `{report_metric_per_protein_save_folder}/{version_num}/mse_per_protein.npy`
- Also saves the test configuration to `{report_metric_per_protein_save_folder}/{version_num}/config.yaml`

**Output:**
- `mse_per_protein.npy`: NumPy array containing MSE force error for each protein
- `config.yaml`: Copy of the test configuration used


## Notes

### Configuration Inheritance
- The test script loads and uses the training configuration from `config_path`
- Model architecture, prior settings, and dataset processing are inherited from training
- Only testing-specific settings can be overridden

### Hardware Settings
- Testing always uses single device (1 GPU or auto CPU)
- Distributed strategies (DDP) are automatically disabled
- This ensures accurate metrics without sample replication

### Gradients During Testing
- Testing runs with `torch.enable_grad()` enabled
- This is required for models that compute forces via autodifferentiation

### Output Location
- By default, results are logged to TensorBoard in the current directory
- The version number from the training run is preserved in logging
- Per-protein results are saved as NumPy arrays for further analysis
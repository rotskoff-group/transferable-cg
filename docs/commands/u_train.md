# u_train

## Usage

### Overview
Trains neural network potential models using PyTorch Lightning.

### Basic Command Structure

The `u_train` command uses Hydra configuration overrides. Each argument follows the pattern `config.key=value`:

```bash
u_train "category.parameter=value" "category.parameter2=value2" ...
```

### Example: Training a Model

```bash
u_train \
  "dataset.dataset_folder_name=./data/training_set/" \
  "dataset.batch_size=16" \
  "train.trainer_args.max_epochs=1000" \
  "train.lightning_model_args.loss_type='force matching'" \
  "train.optimizer_args.nn_lr=0.001"
```

## Configuration Reference

### Global Arguments (`global_args.*`)

| Parameter | Type | Description | Options |
|-----------|------|-------------|---------|
| `use_nn` | bool | Enable neural network component of the potential | `true`, `false` |
| `use_prior` | bool | Enable physics-based prior (force field) component | `true`, `false` |
| `load_model` | path | Path to pretrained model checkpoint for transfer learning<br>**Default:** `null` (train from scratch) | Valid `.ckpt` file path or `null` |

---

### Dataset Arguments (`dataset.*`)

#### Required

| Parameter | Type | Description | Options |
|-----------|------|-------------|---------|
| `dataset_folder_name` | path | Directory containing training data | Valid directory path |

#### Data Loading

| Parameter | Type | Description | Options |
|-----------|------|-------------|---------|
| `model` | str | Dataset type to use<br>**Default:** `"NNDataset"` | `"NNDataset"` |
| `data_in_memory` | bool | Load entire dataset into memory<br>**Default:** `true` | `true`, `false` |
| `batch_size` | int | Training batch size<br>**Default:** `8` | Positive integer |
| `num_workers` | int | Number of data loading workers<br>**Default:** `4` | `0` to number of CPU cores |
| `dtype` | str | Data type for tensors<br>**Default:** `"float32"` | `"float32"`, `"float64"` |

#### Dataset Splitting

| Parameter | Type | Description | Options |
|-----------|------|-------------|---------|
| `dataset_split_args.train` | float | Fraction of data for training<br>**Default:** `0.8` | `0.0` to `1.0` |
| `dataset_split_args.val` | float | Fraction of data for validation<br>**Default:** `0.2` | `0.0` to `1.0` |
| `dataset_split_args.test` | float | Fraction of data for testing<br>**Default:** `0.0` | `0.0` to `1.0` |

#### Graph/Edge Configuration

| Parameter | Type | Description | Options |
|-----------|------|-------------|---------|
| `edge_args.only_cutoff_edges` | bool | Only include edges within cutoff distance<br>**Default:** `true` | `true`, `false` |
| `edge_args.only_nonbonded` | bool | Only include non-bonded interactions<br>**Default:** `false` | `true`, `false` |
| `edge_args.update_edge_indices` | bool | Dynamically update edge indices during training<br>**Default:** `false` | `true`, `false` |

#### Advanced Options

| Parameter | Type | Description | Options |
|-----------|------|-------------|---------|
| `individual_protein_datasets` | path | Path to separate protein-specific datasets<br>**Default:** `null` | Valid directory path or `null` |
| `adaption_dataset_folder_name` | path | Additional dataset for domain adaptation<br>**Default:** `null` | Valid directory path or `null` |
| `train_dataset_fraction` | float | Fraction of training set to use (intended to be used for domain adaption training)<br>**Default:** `1.0` | `0.0` to `1.0` |

| `log_every_n_steps` | int | Log metrics every N steps<br>**Default:** `500` | Positive integer |

---

### Training Arguments (`train.*`)

#### Model Configuration

| Parameter | Type | Description | Options |
|-----------|------|-------------|---------|
| `lightning_model` | str | Lightning model class to use<br>**Default:** `"UModel"` | `"UModel"` |
| `lightning_model_args.return_forces` | bool | Specifies whether the output of the neural network is forces. <br>**Default:** `false` | `true`, `false` |
| `lightning_model_args.model_temperature` | float | Temperature for score matching (K)<br>**Default:** `300` | Positive float |

#### General Loss Configuration

| Parameter | Type | Description | Options |
|-----------|------|-------------|---------|
| `lightning_model_args.loss_type` | str | Training objective<br>**Default:** `"force matching"` | `"force matching"`, `"score matching"` |
| `lightning_model_args.l1_lambda` | float | L1 regularization coefficient. Used for prior only. <br>**Default:** `0.0` | Non-negative float |

#### Score-Matching Loss Configuration
| Parameter | Type | Description | Options |
|-----------|------|-------------|---------|
| `lightning_model_args.div_method` | str | Divergence computation method for score matching<br>**Default:** `"exact"` | `"exact"`, `"hutchinson"`, `"stein"` |
| `lightning_model_args.div_samples` | int | Number of samples for trace estimators<br>**Default:** `100` | Positive integer |
| `lightning_model_args.div_epsilon` | float | Epsilon for Stein computation<br>**Default:** `0.001` | Small positive float |
| `lightning_model_args.vectorize_div` | bool | Vectorize divergence computation<br>**Default:** `true` | `true`, `false` |

#### Optimization

| Parameter | Type | Description | Options |
|-----------|------|-------------|---------|
| `lightning_model_args.layers_to_freeze` | list | Layer names to freeze during training<br>**Default:** `["prior.bonds_mean", "prior.bonds_k", "prior.angles_mean", "prior.angles_k"]` | List of layer names or `[]` |
| `lightning_model_args.optimizer` | str | Optimizer algorithm<br>**Default:** `"Adam"` | `"Adam"`, `"AdamW"`, `"SGD"` |
| `optimizer_args.nn_lr` | float | Learning rate for neural network parameters<br>**Default:** `0.001` | Positive float |
| `optimizer_args.prior_lr` | float | Learning rate for prior (force field) parameters<br>**Default:** `0.00001` | Positive float |
| `lightning_model_args.lr_scheduler` | str | Learning rate scheduler<br>**Default:** `"ReduceLROnPlateau"` | `"ReduceLROnPlateau"`, `"StepLR"`, `"CosineAnnealingLR"`, etc. |
| `lr_scheduler_args` | dict | Arguments for the learning rate scheduler<br>**Note:** Use `++` prefix to add new dictionary keys (ex. `++"train.lr_scheduler_args.patience=10"`) <br>**Default:** `{}` | Scheduler-specific arguments |

#### Monitoring & Logging

| Parameter | Type | Description | Options |
|-----------|------|-------------|---------|
| `lightning_model_args.monitor` | str | Metric to monitor for checkpointing<br>**Default:** `"val/TotalLoss"` | Any logged metric name |
| `lightning_model_args.sync_dist` | bool | Synchronize metrics across distributed processes<br>**Default:** `true` | `true`, `false` |
| `lightning_model_args.on_step` | bool | Log metrics at each step<br>**Default:** `true` | `true`, `false` |

#### Checkpointing & Resume

| Parameter | Type | Description | Options |
|-----------|------|-------------|---------|
| `resume_training_path` | path | Path to checkpoint for resuming training<br>**Default:** `null` | Valid `.ckpt` file path or `null` |

#### Trainer Configuration

| Parameter | Type | Description | Options |
|-----------|------|-------------|---------|
| `trainer_args.accelerator` | str | Hardware accelerator<br>**Default:** `"cuda"` | `"cuda"`, `"cpu"`, `"mps"`|
| `trainer_args.devices` | str/int | Number of devices to use<br>**Default:** `"auto"` | `"auto"` or positive integer|
| `trainer_args.strategy` | str | Distributed training strategy<br>**Default:** `"auto"` | `"auto"`, `"ddp"`, `"ddp_find_unused_parameters_true"`, etc. |
| `trainer_args.log_every_n_steps` | int | Log metrics every N steps<br>**Default:** `10000` | Positive integer |
| `trainer_args.max_epochs` | int | Maximum number of training epochs<br>**Default:** `25000` | Positive integer |
| `trainer_args.enable_progress_bar` | bool | Show progress bar during training<br>**Default:** `false` | `true`, `false` |
| `trainer_args.precision` | str | Numerical precision<br>**Default:** `"32-true"` | `"32-true"`, `"16-mixed"`, `"bf16-mixed"` |
| `trainer_args.accumulate_grad_batches` | int | Accumulate gradients over N batches<br>**Default:** `1` | Positive integer |

#### Random Seed

| Parameter | Type | Description | Options |
|-----------|------|-------------|---------|
| `seed_args.seed` | int | Random seed for reproducibility<br>**Default:** `42` | Any integer |
| `seed_args.workers` | bool | Set worker seed for data loaders<br>**Default:** `true` | `true`, `false` |

---

### Model Architecture (`nn.*`)

The neural network architecture is configured via Hydra config groups. Default: `schnet`

**To change architecture:**
```bash
u_train nn=schnet     # Use MACE architecture
u_train nn=esen        # Use eSEN architecture
```
See `cfgs/nn/` directory for available architectures and their parameters.

---

### Prior Configuration (`prior.*`)

The physics-based prior is configured via Hydra config groups. Default: `forcefield`

See `cfgs/prior/` directory for available priors and their parameters.

---

## Common Workflows

### Basic Training

Train a model from scratch on a new dataset:

```bash
u_train \
  "dataset.dataset_folder_name=./data/my_protein/" \
  "dataset.batch_size=16" \
  "train.trainer_args.max_epochs=1000"
```

### Force Matching Training

Train using force matching loss:

```bash
u_train \
  "dataset.dataset_folder_name=./data/my_protein/" \
  "train.lightning_model_args.loss_type='force matching'" \
  "train.optimizer_args.nn_lr=0.001"
```

### Score Matching Training

Train using score matching loss:

```bash
u_train \
  "dataset.dataset_folder_name=./data/my_protein/" \
  "train.lightning_model_args.loss_type='score matching'" \
  "train.lightning_model_args.div_method='stein'" \
  "train.lightning_model_args.div_samples=5"
```

### Transfer Learning

Fine-tune a pretrained model on new data.

**Important:** When loading a pretrained model, you must override the default configs to match the original model's architecture and settings (e.g., same `nn` type, hidden dimensions, layer counts). Only the dataset and training hyperparameters (like learning rate) should differ.
```bash
u_train \
  nn=schnet \  # Must match pretrained model architecture
  prior=forcefield \  # Must match pretrained model prior
  "dataset.dataset_folder_name=./data/new_protein/" \
  "global_args.load_model=./pretrained_models/best_model.ckpt" \
  "train.optimizer_args.nn_lr=0.0001"  # Lower LR for fine-tuning
```

**Tip:** Check the pretrained model's `config.yaml` (saved during training) to see the exact configuration used.

### Multi-GPU Training

Train on multiple GPUs:

```bash
u_train \
  "dataset.dataset_folder_name=./data/my_protein/" \
  "train.trainer_args.devices=4" \
  "train.trainer_args.strategy='ddp'" \
  "dataset.batch_size=32"
```

### Training with Domain Adaptation

Train with an additional adaptation dataset:

```bash
u_train \
  "dataset.dataset_folder_name=./data/source_data/" \
  "dataset.adaption_dataset_folder_name=./data/target_data/" \
  "dataset.batch_size=16"
```


# Training MACE Models

## Prerequisites

First, create the training dataset by following the instructions in [`docs/tutorials/training_dataset.md`](docs/tutorials/training_dataset.md).

---

## Training with MFM (Mean Force Matching)

Train a MACE model using mean force estimates with our recommended hyperparameters:

```bash
root_data_folder="path/to/MFM/dataset"  # Path to MFM training dataset

uv run u_train \
    'global_args.use_nn=True' \
    'global_args.use_moe=False' \
    'global_args.use_prior=True' \
    'nn=mace' \
    'prior=forcefield' \
    'nn.model_args.hidden_irreps=32x0e + 32x1o' \
    'nn.model_args.num_interactions=2' \
    'nn.model_args.use_cueq=True' \
    'train.lightning_model_args.return_forces=False' \
    'train.lightning_model_args.optimizer_args.nn_lr=0.001' \
    'train.lightning_model_args.optimizer_args.prior_lr=0.001' \
    'train.lightning_model_args.loss_type=force matching' \
    'train.lightning_model_args.layers_to_freeze=["prior.bonds_mean", "prior.bonds_k", "prior.angles_mean", "prior.angles_k", "prior.dihedral_a1", "prior.dihedral_a2"]' \
    'train.lightning_model_args.l1_lambda=1.0' \
    'train.trainer_args.max_epochs=500' \
    'train.lightning_model_args.on_step=False' \
    "dataset.dataset_folder_name=${root_data_folder}" \
    'dataset.dataset_split_args.train=0.8' \
    'dataset.dataset_split_args.val=0.1' \
    'dataset.dataset_split_args.test=0.1'
```

---

## Training with FM (Force Matching)

Train a MACE model using instantaneous forces from MD trajectories:

> 💡 **Note:** FM uses the **same command** as MFM. Simply change `dataset.dataset_folder_name` to point to the FM dataset created from `instantaneous_md`.

```bash
root_data_folder="path/to/FM/dataset"  # Path to FM training dataset (from instantaneous_md)

# Use the same command as MFM above, just with different dataset path
uv run u_train \
    # ... (identical parameters to MFM)
    "dataset.dataset_folder_name=${root_data_folder}"
```

---

## Training with SM (Score Matching)

Train a MACE model using score matching with divergence estimation:

```bash
root_data_folder="path/to/SM/dataset"  # Path to SM training dataset

uv run u_train \
    'global_args.use_nn=True' \
    'global_args.use_moe=False' \
    'global_args.use_prior=True' \
    'nn=mace' \
    'prior=forcefield' \
    'nn.model_args.hidden_irreps=32x0e + 32x1o' \
    'nn.model_args.num_interactions=2' \
    'nn.model_args.use_cueq=True' \
    'train.lightning_model_args.return_forces=False' \
    'train.lightning_model_args.optimizer_args.nn_lr=0.001' \
    'train.lightning_model_args.optimizer_args.prior_lr=0.001' \
    'train.lightning_model_args.loss_type=score matching' \
    'train.lightning_model_args.div_samples=1' \
    'train.lightning_model_args.div_epsilon=0.0001' \
    'train.lightning_model_args.vectorize_div=False' \
    'train.lightning_model_args.layers_to_freeze=["prior.bonds_mean", "prior.bonds_k", "prior.angles_mean", "prior.angles_k", "prior.dihedral_a1", "prior.dihedral_a2"]' \
    'train.lightning_model_args.l1_lambda=1.0' \
    'train.trainer_args.max_epochs=500' \
    'train.lightning_model_args.on_step=False' \
    "dataset.dataset_folder_name=${root_data_folder}" \
    'dataset.dataset_split_args.train=0.8' \
    'dataset.dataset_split_args.val=0.1' \
    'dataset.dataset_split_args.test=0.1'
```
# transferable-cg

## Installation

### Prerequisites

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) package manager

### Setup

1. **Install dependencies:**
   ```bash
   uv sync
   ```
    > **Note:** For CUDA support, use `uv sync --extra cuda`

2. **Install PyTorch Scatter extensions:**
   ```bash
   bash install.sh
   ```

## Usage
| Command | Description | 
| ------- | ----------- |
| uv run make_u_dataset | Creates coarse-grained (CG) datasets from atomistic trajectory data for training. |
| uv run u_train | Trains neural network potential models using PyTorch Lightning. |
| uv run u_test | Evaluates trained neural network potential models on test datasets. |
| uv run cg_sim | Runs parallel coarse-grained (CG) molecular dynamics simulations using a trained model on a single GPU.

### Documentation

- **Command reference:** See `docs/commands/` for detailed usage instructions and configuration options
- **Tutorials:** Step-by-step guides for dataset creation, model training, and running simulations are available in `docs/tutorials/`

## Data

Training and testing datasets are available on Hugging Face:  
https://huggingface.co/datasets/abpark/transferable-cg

## Pre-trained Models

Pre-trained model weights are provided in the `model_weights/` directory. Each model architecture includes four versions trained using different methods:
- **MFM** (Mean Force Matching)
- **MFM 100K** (Mean Force Matching on expanded dataset)
- **FM** (Force Matching)
- **SM** (Score Matching)

 
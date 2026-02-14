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

The installation script will automatically detect CUDA availability and install the appropriate versions.



## Usage
| Command | Description | 
| ------- | ----------- |
| uv run make_u_dataset | Creates coarse-grained (CG) datasets from atomistic trajectory data for training. |
| uv run u_train | Trains neural network potential models using PyTorch Lightning. |
| uv run u_test | Evaluates trained neural network potential models on test datasets. |
| uv run cg_sim | Runs parallel coarse-grained (CG) molecular dynamics simulations using a trained model on a single GPU.

For detailed usage instructions, configuration options, and examples, see the individual command documentation in `docs/commands/`.

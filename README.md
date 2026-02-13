# transferable-cg

## Installation

### Prerequisites

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) package manager

### Setup Instructions

#### 1. Create and Activate Environment

Create a new uv environment:
```bash
uv venv
source .venv/bin/activate  # On Linux/macOS
```

### 2. Install PyTorch with CUDA Support

First, install PyTorch (preferrably with CUDA support) by following the instructions at:
https://pytorch.org/get-started/locally/


### 3. Install Project Dependencies

Change into transferable-cg directory and install the project in editable mode:
```bash
uv pip install -e .
```

### 4. Install torch-scatter

Install `torch-scatter` with the appropriate CUDA version:
```bash
uv pip install torch-scatter -f https://data.pyg.org/whl/torch-2.7.0+${CUDA}.html
```

Replace `${CUDA}` with one of the following based on your PyTorch installation:
- `cpu` - CPU only
- `cu118` - CUDA 11.8
- `cu121` - CUDA 12.1
- `cu126` - CUDA 12.6
- `cu128` - CUDA 12.8

For more information, visit: https://pypi.org/project/torch-scatter/

## Usage
| Command | Description | 
| ------- | ----------- |
| make_u_dataset | Creates coarse-grained (CG) datasets from atomistic trajectory data for training. |
| u_train | Trains neural network potential models using PyTorch Lightning. |
| u_test | Evaluates trained neural network potential models on test datasets. |
| cg_sim | Runs parallel coarse-grained (CG) molecular dynamics simulations using a trained model on a single GPU.

For detailed usage instructions, configuration options, and examples, see the individual command documentation in `docs/commands/`.
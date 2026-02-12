# transferable-cg

## Installation

### 1. Install PyTorch with CUDA Support

First, install PyTorch (preferrably with CUDA support) by following the instructions at:
https://pytorch.org/get-started/locally/


### 2. Install Project Dependencies

Install the project in editable mode:
```bash
uv pip install -e .
```

### 3. Install torch-scatter

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

[Add usage instructions here]

## Requirements

- Python 3.x
- CUDA-capable GPU (recommended)
- PyTorch 2.7.0+
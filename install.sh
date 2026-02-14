CUDA_VERSION=$(uv run python -c "import torch; print(torch.__version__.split('+')[-1])")
uv pip install torch-scatter -f https://data.pyg.org/whl/torch-2.7.0+${CUDA_VERSION}.html
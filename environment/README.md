# Environment Boundary

The training runs were executed on Linux with a Miniconda Python 3.10
environment and CUDA-capable GPUs. `requirements.txt` records the required
Python package families, but it is not presented as an exact historical
`pip freeze`: the original server-level PyTorch, torchvision, CUDA, cuDNN, and
driver versions have not been recovered reliably.

This limitation is explicit so that the artifact does not claim byte-identical
environment reconstruction. The frozen analyzer outputs and checksums support
verification of the reported statistics without rerunning every training job.

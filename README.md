# Diffusion Posterior Sampling Benchmark
Reproducible benchmark for **diffusion posterior sampling (DPS) algorithms** on canonical inverse problems (denoising, deconvolution, imputation, and reconstruction from partial Fourier measurements), with model-based baselines and end-to-end scripts for training denoisers, parameter search, posterior sampling, and evaluation.

## TL;DR: one-minute Docker quickstart
> **Why a container?** This project depends on custom **CUDA/C++ sampling operators** from
> [`logsumexpv2`](https://github.com/zacmar/logsumexpv2) and **Triton-compiled Python kernels**.
> Compiling these reliably across different CUDA/PyTorch/toolchain versions is fragile.
> The published Docker image pins the exact PyTorch/CUDA stack and **ships with the compiled
> extensions**, so you can run the benchmark without wrestling with build environments.

1) **Requirements**
- Linux x86_64 with an NVIDIA driver (CUDA 12.x runtime compatible) and the **NVIDIA Container Toolkit** (`nvidia-docker2`).
- Disk space: a full run can consume **~171 GB**.

2) **Pull the image**
```bash
# Public image on GitHub Container Registry
docker pull ghcr.io/zacmar/logsumexpv2:v2.1
```

3) **GPU smoke test**
```bash
docker run --rm --gpus all ghcr.io/zacmar/logsumexpv2:v2.1 python - <<'PY'
import torch as th
print("CUDA available:", th.cuda.is_available())
if th.cuda.is_available():
    print("Device:", th.cuda.get_device_name(0))
PY
```

4) **Run the pipeline (minimal example)**
```bash
# Create a host directory for large artifacts
mkdir -p "$HOME/dps-benchmark-data"

# Run inside the container with your repo mounted
# (uses /data for outputs via EXPERIMENTS_ROOT)
docker run --rm -it \
  --gpus all \
  --shm-size=8g \
  -u $(id -u):$(id -g) \
  -v "$PWD":/workspace \
  -v "$HOME/dps-benchmark-data":/data \
  -w /workspace \
  -e EXPERIMENTS_ROOT=/data \
  ghcr.io/zacmar/logsumexpv2:v2.1 \
  python generate-datasets.py identity student 1
```

Then continue with the stages below using the same `docker run` pattern and replacing the last line (the Python command).

## Prerequisites & environment
- **GPU**: Some sampling routines are CUDA-accelerated and require an NVIDIA GPU.
- **Container**: We publish a Docker image specifically to avoid the pain of compiling the
  CUDA/C++ sampling operators from [`logsumexpv2`](https://github.com/zacmar/logsumexpv2) and setting up
  **Triton** across diverse environments. If you prefer a local install, you’ll need a matching
  PyTorch/CUDA toolchain **and** a working CUDA build environment.
- **Storage**: Set `EXPERIMENTS_ROOT` to a directory with sufficient space. A full run requires about **171 GB**.

```bash
export EXPERIMENTS_ROOT=/path/to/fast/storage
```

### Setup
The pipeline assumes that the environment variable `EXPERIMENTS_ROOT` points to a path that points to a storage device that has sufficient space.
A full run of the pipeline requires about 171 gigabytes of storage.

## Pipeline
The pipeline is compartmentalized into stages; each stage typically has an associated Python file with an argument parser. Where applicable, you can specify:

1. **Forward operator**: `identity`, `convolution`, `sample`, `fourier`
2. **Jump distribution** (and parameters): `gauss`, `laplace`, `student`, `bernoulli-laplace`  
   – `Bernoulli-Laplace` has **2 parameters**; the others have **1**.
3. **DPS algorithm** and **denoiser**: DPS algorithms `{cdps, diffpir, dpnp}` with denoiser `{learned, gibbs}`.

This enables straightforward parallelization across compute nodes.

### 1) Data generation
Synthesize training/validation/test signals with the specified jump distribution. For test signals, simulate the measurement process and draw gold‑standard posterior samples via Gibbs methods.

```bash
python generate-datasets.py identity student 1
```

### 2) Training of the denoisers
The training signals are used to train standard noise-conditional score networks. An example launch looks like
```
python train.py bernoulli-laplace 0.1 1
```
The output of this stage are learned denoisers for the specified jump distributions.

### 3) Parameter search (model-based + DPS algorithms)
Use the validation set to select parameters for model‑based methods and DPS algorithms. Parameter grids (defined in the scripts) are tuned for the standard distributions in `generate-datasets.py` and may need adjustment for exotic cases. We also compute model‑based estimates on the test data at the chosen parameters.

```bash
python grid-search.py convolution student 1
```

Outputs: validation MSEs over the grids (for model‑based and DPS algorithms) and the corresponding test‑set estimates for model‑based methods.

### 4) Posterior sampling with DPS algorithms
Run the DPS algorithms on the test data using the optimal parameters inferred from validation. Launches are parameterized by forward operator, jump distribution, DPS algorithm, and denoiser.

```bash
python posterior-sampling.py identity laplace 1 diffpir gibbs
```

Outputs: posterior samples produced by the DPS algorithms.

### 5) Evaluation & visualization
With gold‑standard posterior samples (Gibbs), DPS samples, and model‑based MMSE estimates on disk, evaluate and produce the figures/tables.

Main tables:
```bash
python -m postprocessing.mmse-gap-latex-table
```

Data for the main figure(s):
```bash
python -m postprocessing.posterior-figure-data
```

## Citing
If you use this benchmark in your research, please cite:
```
@misc{zach2025statisticalbenchmarkdiffusionposterior,
      title={A Statistical Benchmark for Diffusion Posterior Sampling Algorithms}, 
      author={Zach, Martin and Haouchat, Youssef and Unser, Michael},
      year={2025},
      eprint={2509.12821},
      archivePrefix={arXiv},
      primaryClass={eess.SP},
      url={https://arxiv.org/abs/2509.12821}, 
}
```
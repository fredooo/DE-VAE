# **DE-VAE: Revealing Uncertainty in Parametric and Inverse Projections with Variational Autoencoders using Differential Entropy**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Uses: venv](https://img.shields.io/badge/Environment-venv-blue)](https://docs.python.org/3/library/venv.html)
[![Paper](https://img.shields.io/badge/paper-arXiv-red)](https://arxiv.org/abs/2508.12145)
[![OSF Project](https://img.shields.io/badge/OSF-View%20Project-lightgrey)](https://osf.io/zr6xf/)

📄 **Paper:** [PDF](https://osf.io/gq8f7)

## Key Features

* Learns a parametric projection that maps high-dimensional data to a probabilistic latent space, enabling explicit modeling of uncertainty.
* Represents each data point as a full, diagomal, or isotropic Gaussian distribution.
* Provides an uncertainty-aware visualization of the projection, i.e., latent space, showing confidence around points.
* Includes an inverse model to reconstruct original data from projected points.
* Optimizes multiple losses to ensure accurate reconstruction, projection alignment, and interpretable uncertainty.


![Overview][1]

In this example, the encoder of a DE-VAE learns a parametric projection $P$ of MNIST, mapping each data point $x_i$ to a full Gaussian $\mathcal{N}(\mu, \Sigma)$, modeling the uncertainty of a UMAP projection. The decoder learns an inverse projection $P^{-1}$, taking $y_k$ and reconstructing a plausible sample $\hat{x}_k$. $P$ enables uncertainty-aware visualization of the latent space. DE-VAEs optimize the losses: $\mathcal{L}_{\text{recon}}$, ensuring reconstruction; $\mathcal{L}_{\text{proj}}$, aligning $\mu$ with points of the projection; and $\mathcal{L}_{\text{ent}}$, maximizing the variance of $\Sigma$. To show learned Gaussian distributions, we depict the 1st, 2nd, and 3rd standard deviations as ellipses around two randomly sampled points per class.

## Requirements

* **Python** ≥ 3.10 ([Python 3.10.x](https://www.python.org/downloads/release/python-3100/))
* **Virtual environment**: [venv](https://docs.python.org/3/library/venv.html) 

## How to Run

### 1. Setup Environment

```bash
# Create a virtual environment
python3 -m venv .venv

# Activate the environment
source .venv/bin/activate

# Install dependencies
pip3 install -r requirements.txt
```

### 2. Retrieve Trained Models

```bash
# Download pretrained models and datasets
sh ./retrieve_replication.sh
```

**Optional:** If you'd like to train the models yourself instead (this may take some time):

```bash
# Compute projection data
python3 projections.py

# Train VAE/AE models
python3 trainer.py
```

### 3. Visualize Results

Explore model outputs and projections visually, e.g., with:

```bash
# usage: visual.py [-h] --model MODEL
#
# Load a model from a specified path and show the latent space visualization.
#
# options:
#   -h, --help     show this help message and exit
#   --model MODEL  Path to the model file.
python3 visual.py --model ./models/vae-full-fmnist-umap-p20.00-e4.00000-s0.pt
```

### 4. Show Quantitative Results
```bash
# usage: create_tables.py [-h] [--model MODEL] [--dataset DATASET] [--projection PROJECTION] [--all-latex]
#
# Generate and print evaluation tables.
#
# options:
#  -h, --help            show this help message and exit
#  --model MODEL         model name key (e.g., 'vae-full')
#  --dataset DATASET     dataset name key (e.g., 'mnist')
#  --projection PROJECTION
#                        projection name key (e.g., 'umap')
#  --all-latex           Run full evaluation to generate all LaTeX tables
python3 create_tables.py --model vae-full --dataset mnist --projection umap
```

### 5. Experimental: Train DE-VAE Models on Own Dataset 

```bash
# usage: main.py [-h] --model MODEL --data DATA [--label LABEL] --projection PROJECTION [--l-proj L_PROJ] [--l-ent L_ENT] [--seed SEED]
# 
# Generate and print evaluation tables.
# 
# options:
#   -h, --help            show this help message and exit
#   --model MODEL         model name key (e.g., 'vae-full')
#   --data DATA           high-dimensional data as CSV file
#   --label LABEL         Column name specifing class labels
#   --projection PROJECTION
#                         2D projection data as CSV file
#   --l-proj L_PROJ       projection loss weight
#   --l-ent L_ENT         entropy loss weight
#   --seed SEED           random seed for reproducibility
python3 main.py --model vae-full --data demo.csv --label class --projection umap  
```

---

## File Overview

| File Name           | Description                                                                              |
| ------------------- | ---------------------------------------------------------------------------------------- |
| `create_tables.py`  | Generates tables summarizing model results for each dataset and projection method.       |
| `data_loader.py`    | Loads and preprocesses datasets: MNIST, FashionMNIST, KMNIST, and HAR.                   |
| `loss_functions.py` | Implements loss functions used during training of VAE/AE models.                         |
| `main.py`           | Enables training of DE-VAE models for user-defined datasets and parameters.              |
| `projections.py`    | Projects high-dimensional data to 2D using UMAP, t-SNE, and LLE.                         |
| `trainer.py`        | Training loop and utilities for training VAE/AE models.                                  |
| `vae_models.py`     | Defines the architectures for various VAE and AE model variants.                         |
| `visual.py`         | Visual tools for exploring model outputs and projections.                                |

## License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).

[1]: https://github.com/fredooo/DE-VAE/raw/main/overview.png

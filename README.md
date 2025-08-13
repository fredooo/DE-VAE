# **DE-VAE: Revealing Uncertainty in Parametric and Inverse Projections with Variational Autoencoders using Differential Entropy**

**Paper:** *To be added*

---

![Overview][1]

[1]: https://github.com/fredooo/DE-VAE/raw/main/overview.png

## How to Run

### 1. Setup Environment

```bash
# Create a virtual environment
python3 -m venv .venv

# Activate the environment
source .venv/bin/activate

# Install dependencies
pip3 install -r requirements.txt

# Download pretrained models and datasets
sh ./setup.sh
```

### 2. Train Models (Optional)

If you'd like to train the models yourself instead of using the pretrained ones:

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
python3 visual.py --model ./models/vae-full-fmnist-umap-p20.00-e4.00000-s0.pt
```

---

## File Overview

| File Name           | Description                                                                              |
| ------------------- | ---------------------------------------------------------------------------------------- |
| `create_tables.py`  | Generates LaTeX tables summarizing model results for each dataset and projection method. |
| `data_loader.py`    | Loads and preprocesses datasets: MNIST, FashionMNIST, KMNIST, and HAR.                   |
| `loss_functions.py` | Implements loss functions used during training of VAE/AE models.                         |
| `projections.py`    | Projects high-dimensional data to 2D using UMAP, t-SNE, and LLE.                         |
| `trainer.py`        | Training loop and utilities for training VAE/AE models.                                  |
| `vae_models.py`     | Defines the architectures for various VAE and AE model variants.                         |
| `visual.py`         | Visual tools for exploring model outputs and projections.                                |


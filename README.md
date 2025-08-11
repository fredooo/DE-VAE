# DE-VAE
DE-VAE: Revealing Uncertainty in Parametric and Inverse Projections with Variational Autoencoders using Differential Entropy

## Run

Create an environment:

```
python3 -m venv .venv
```

Activate the environment:

```
source .venv/bin/activate
```

Install dependencies:

```
pip3 install -r requirements.txt
```

Calculate projections:

```
python3 projection.py
```

Train models:

```
python3 trainer.py
```

## Files

| File Name           | Description                                                                                  |
|---------------------|----------------------------------------------------------------------------------------------|
| `data_loader.py`    | Handles loading and preprocessing of datasets such as MNIST, FashionMNIST, and HAR.          |
| `medoids.py`     | Computes the medoid (central point) of a set of vectors.                                     |
| `loss_functions.py` | Implements various loss functions used in training VAE/AE models.                            |
| `projection.py`     | Projects high-dimensional data into lower dimensions, possibly using UMAP, etc.              |
| `trainer.py`        | Contains the training loop and utilities for training VAE/AE models.                         |
| `vae_models.py`     | Defines the architectures for VAE and AE models, including different variants.               |
| `visual.py`         | Provides additional visualization utilities.                                                 |
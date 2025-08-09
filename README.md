# DE-VAE
DE-VAE: Revealing Uncertainty in Parametric and Inverse Projections with Variational Autoencoders using Differential Entropy

### Python Files

| File Name           | Description                                                                                  |
|---------------------|----------------------------------------------------------------------------------------------|
| `data_loader.py`    | Handles loading and preprocessing of datasets such as MNIST, FashionMNIST, and HAR.          |
| `get_medoid.py`     | Computes the medoid (central point) of a set of vectors.                                     |
| `loss_functions.py` | Implements various loss functions used in training VAE/AE models.                            |
| `plotting.py`       | Contains functions for visualizing data, model outputs, or training progress.                |
| `projection.py`     | Projects high-dimensional data into lower dimensions, possibly using UMAP, etc.              |
| `trainer.py`        | Contains the training loop and utilities for training VAE/AE models.                         |
| `vae_models.py`     | Defines the architectures for VAE and AE models, including different variants.               |
| `visual.py`         | Provides additional visualization utilities.                                                 |
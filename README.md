# DE-VAE: Revealing Uncertainty in Parametric and Inverse Projections with Variational Autoencoders using Differential Entropy

**Paper:** TODO

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
python3 projections.py
```

Train models:

```
python3 trainer.py
```

Visually explore results:

```
python3 visual.py
```

## Files

| File Name           | Description                                                                                  |
|---------------------|----------------------------------------------------------------------------------------------|
| `data_loader.py`    | Handles loading and preprocessing of datasets such as MNIST, FashionMNIST, KMNIST, and HAR.  |
| `loss_functions.py` | Implements various loss functions used in training VAE/AE models.                            |
| `projections.py`    | Projects high-dimensional data into lower dimensions, using UMAP, t-SNE, and LLE             |
| `trainer.py`        | Contains the training loop and utilities for training VAE/AE models.                         |
| `vae_models.py`     | Defines the architectures for VAE and AE models, including different variants.               |
| `visual.py`         | Provides methods to viusally expore the resutsl and additional utlities.                     |
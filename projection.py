import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.manifold import LocallyLinearEmbedding, MDS, TSNE
import torch
import umap

from data_loader import create_data_loaders, load_fashion_mnist, load_har, load_kmnist, load_mnist
from visual import plot_projection_from_loader

projection_seed = 777


def project_data(vectors, labels, output_prefix, method: str = "umap"):
    model = None
    embedding = None
    if method == "umap":
        model = umap.UMAP(random_state=projection_seed)
        embedding = np.array(model.fit_transform(vectors))
    elif method == "tsne":
        model = TSNE(n_components=2, random_state=projection_seed)
        embedding = model.fit_transform(vectors) 
    elif method == "mds":
        model = MDS(n_components=2, random_state=projection_seed, n_jobs=-1)
        embedding = model.fit_transform(vectors)
    else:
        model = LocallyLinearEmbedding(n_components=2, random_state=projection_seed)
        embedding = model.fit_transform(vectors)

    # Save coordinates and labels to CSV
    df = pd.DataFrame({
        'x': embedding[:, 0],
        'y': embedding[:, 1],
        'label': labels.numpy()
    })

    csv_path = Path(f"./preprocessed/{output_prefix}/{method}.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"Saved coordinates to: {csv_path}")

    train_loader, val_loader, test_loader = create_data_loaders(vectors, embedding, labels)
    plot_projection_from_loader(train_loader, "Train Data", f"./images/projections/{method}/{output_prefix}_train_data.png")
    plot_projection_from_loader(val_loader, "Validation Data", f"./images/projections/{method}/{output_prefix}_val_data.png")
    plot_projection_from_loader(test_loader, "Test Data", f"./images/projections/{method}/{output_prefix}_test_data.png")

    return model


def sample_umap_grid_and_inverse(umap_model, output_prefix, grid_size=7, img_shape=(28, 28)):
    # Determine min and max range from the fitted UMAP embeddings
    embedding = umap_model.embedding_
    x_min, x_max = embedding[:, 0].min(), embedding[:, 0].max()
    y_min, y_max = embedding[:, 1].min(), embedding[:, 1].max()

    # Create evenly spaced grid
    x_vals = np.linspace(x_min, x_max, grid_size)
    y_vals = np.linspace(y_min, y_max, grid_size)
    xv, yv = np.meshgrid(x_vals, y_vals)
    grid_points = np.stack([xv.ravel(), yv.ravel()], axis=1)  # Shape: (grid_size * grid_size, 2)

    npy_path = Path(f"./preprocessed/{output_prefix}/umap_grid_points.npy")
    np.save(npy_path, grid_points)
    print(f"Saved grid positions to: {npy_path}")

    # Inverse transform to image space
    inverses = umap_model.inverse_transform(grid_points)
    inverses = torch.tensor(inverses).reshape(-1, *img_shape)

    # Plot in grid
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(grid_size, grid_size))
    for i in range(grid_size * grid_size):
        row, col = divmod(i, grid_size)
        ax = axes[grid_size - 1 - row, col]
        ax.imshow(inverses[i], cmap='gray')
        ax.axis('off')
    plt.tight_layout()
    img_path = Path(f"./images/projections/umap/{output_prefix}_umap_inverse.png")
    img_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(img_path)


def process_har():
    print("Processing HAR")
    vectors, labels = load_har()
    output_prefix = "har"
    project_data(vectors, labels, output_prefix, method="lle")
    project_data(vectors, labels, output_prefix, method="tsne")
    project_data(vectors, labels, output_prefix, method="umap")


def process_mnist():
    print("Processing MNIST")
    vectors, labels = load_mnist()
    output_prefix = "mnist"
    project_data(vectors, labels, output_prefix, method="lle")
    project_data(vectors, labels, output_prefix, method="tsne")
    umap_model = project_data(vectors, labels, output_prefix, method="umap")
    sample_umap_grid_and_inverse(umap_model, output_prefix, grid_size=7)


def process_fashion_mnist():
    print("Processing Fashion MNIST")
    vectors, labels = load_fashion_mnist()
    output_prefix = "fmnist"
    project_data(vectors, labels, output_prefix, method="lle")
    project_data(vectors, labels, output_prefix, method="tsne")
    umap_model = project_data(vectors, labels, output_prefix, method="umap")
    sample_umap_grid_and_inverse(umap_model, output_prefix, grid_size=7)


def process_kmnist():
    print("Processing KMNIST")
    vectors, labels = load_kmnist()
    output_prefix = "kmnist"
    project_data(vectors, labels, output_prefix, method="lle")
    project_data(vectors, labels, output_prefix, method="tsne")
    umap_model = project_data(vectors, labels, output_prefix, method="umap")
    sample_umap_grid_and_inverse(umap_model, output_prefix, grid_size=7)


if __name__ == "__main__":
    process_har()
    process_mnist()
    process_fashion_mnist()
    process_kmnist()

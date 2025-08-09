import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import umap

from data_loader import load_fashion_mnist, load_har, load_mnist, create_data_loaders
from plotting import plot_projection_from_loader


def project_with_umap(vectors, labels, output_prefix):
    # Run UMAP
    model = umap.UMAP()
    embedding = model.fit_transform(vectors)

    # Save coordinates and labels to CSV
    df = pd.DataFrame({
        'x': embedding[:, 0],
        'y': embedding[:, 1],
        'label': labels.numpy()
    })
    df.to_csv(f"{output_prefix}_umap.csv", index=False)
    print(f"Saved coordinates to: {output_prefix}_umap.csv")

    train_loader, val_loader, test_loader = create_data_loaders(vectors, embedding, labels)
    plot_projection_from_loader(train_loader, "Train Data", f"./images/{output_prefix}_train_data.png")
    plot_projection_from_loader(val_loader, "Validation Data", f"./images/{output_prefix}_val_data.png")
    plot_projection_from_loader(test_loader, "Test Data", f"./images/{output_prefix}_test_data.png")

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
    grid_points = np.stack([xv.ravel(), yv.ravel()], axis=1)  # Shape: (grid_size², 2)

    np.save(f"{output_prefix}_grid_points.npy", grid_points)
    print(f"Saved grid positions to: {output_prefix}_umap_grid_points.npy")

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
    plt.savefig(f"./images/{output_prefix}_umap_inverse.png")


def process_har():
    print("Processing HAR")
    vectors, labels = load_har()
    output_prefix = "har"
    project_with_umap(vectors, labels, output_prefix)


def process_mnist():
    print("Processing MNIST")
    vectors, labels = load_mnist()
    output_prefix = "mnist"
    umap_model = project_with_umap(vectors, labels, output_prefix)
    sample_umap_grid_and_inverse(umap_model, output_prefix, grid_size=7)


def process_fashion_mnist():
    print("Processing Fashion MNIST")
    vectors, labels = load_fashion_mnist()
    output_prefix = "fmnist"
    umap_model = project_with_umap(vectors, labels, output_prefix)
    sample_umap_grid_and_inverse(umap_model, output_prefix, grid_size=7)


if __name__ == "__main__":
    process_har()
    process_mnist()
    process_fashion_mnist()

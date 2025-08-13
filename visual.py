import argparse
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import torch

from data_loader import create_loaders_for_dataset, model_outputs
from trainer import set_seed
from vae_models import load, VaeGaussianDiagonal, VaeGaussianFull, VaeGaussianIsotropic


def sample_classes(labels, num_points_per_label: int = 3):
    unique_labels = labels.unique()
    selected_indices = []
    for lbl in unique_labels:
        label_mask = (labels == lbl)
        idx = (label_mask).nonzero(as_tuple=True)[0]
        if len(idx) >= num_points_per_label:
            selected = idx[:num_points_per_label]
        else:
            selected = idx
        selected_indices.append(selected)
    return torch.cat(selected_indices)


def calculate_medoids(points, labels):
    medoid_indices = []
    unique_labels = torch.unique(labels)
    
    # For each label, compute medoid
    for label in unique_labels:
        label_mask = (labels == label)
        points_of_label = points[label_mask]
        
        # Compute pairwise distance matrix (Euclidean)
        dist_matrix = torch.cdist(points_of_label, points_of_label, p=2)
        
        # Sum distances for each point
        sum_distances = dist_matrix.sum(dim=1)
        
        # Get index of the point with minimal sum distance within the label subset
        medoid_idx_within_label = torch.argmin(sum_distances)
        
        # Get the original index of the medoid in the overall dataset
        original_idx = (label_mask).nonzero(as_tuple=True)[0][medoid_idx_within_label]
        
        # Append the original index to the list
        medoid_indices.append(original_idx.item())
    
    return medoid_indices


def plot_projection_from_loader(data_loader, title, filename):
    all_points_2d = []
    all_labels = []

    # Collect all batches
    for _, points_2d, labels in data_loader:
        all_points_2d.append(points_2d)
        all_labels.append(labels)

    # Concatenate everything into single tensors
    all_points_2d = torch.cat(all_points_2d, dim=0).cpu()
    all_labels = torch.cat(all_labels, dim=0).cpu()

    x = all_points_2d[:, 0]
    y = all_points_2d[:, 1]

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(x, y, c=all_labels, cmap='tab10', s=5, alpha=0.7)
    plt.colorbar(scatter, ticks=range(10), label="Digit Label")
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()

    img_path = Path(filename)
    img_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(img_path)
    #plt.show()


def plot_latent_2d(mu, label, size=5):
    sizes = torch.ones(mu.size(0)) * size
    plt.scatter(mu[:, 0], mu[:, 1], c=label, s=sizes, cmap='tab10', alpha=0.2)


def draw_ellipses_halo(mu, widths, heights, angles, alpha: float = 1.0, linewidth: float = 2, ax=None):
    if ax is None:
        ax = plt.gca()

    for i in range(len(mu)):
        ellipse = Ellipse(
            xy=mu[i],
            width=widths[i],
            height=heights[i],
            angle=angles[i],
            facecolor='none',  # No fill
            edgecolor='black',  # Black outline
            alpha=alpha,
            linewidth=linewidth
        )
        ax.add_patch(ellipse)

        # Scaled ellipse (2x dashed)
        ellipse_dashed = Ellipse(
            xy=mu[i],
            width=2 * widths[i],
            height=2 * heights[i],
            angle=angles[i],
            facecolor='none',
            edgecolor='black',
            alpha=alpha,
            linewidth=linewidth,
            linestyle='dashed'
        )
        ax.add_patch(ellipse_dashed)

        ellipse_dotted = Ellipse(
            xy=mu[i],
            width=3 * widths[i],
            height=3 * heights[i],
            angle=angles[i],
            facecolor='none',
            edgecolor='black',
            alpha=alpha,
            linewidth=linewidth,
            linestyle='dotted'
        )
        ax.add_patch(ellipse_dotted)

        # Draw black center point
        ax.plot(mu[i][0], mu[i][1], 'ko', markersize=1)


def draw_std_ellipses_filled(mu, std, scale=1.0):
    std = std.numpy()
    widths = heights = [scale * s for s in std]
    angles = [0.0] * len(std)
    draw_ellipses_halo(mu, widths, heights, angles)


def draw_diag_ellipses_filled(mu, logvar, scale=1.0):
    std = torch.exp(0.5 * logvar).numpy()
    widths = [scale * s[0] for s in std]
    heights = [scale * s[1] for s in std]
    angles = [0.0] * len(std)  # Axis-aligned
    draw_ellipses_halo(mu, widths, heights, angles)


def draw_cov_ellipses_filled(mu, L, scale=1.0):
    widths, heights, angles = [], [], []

    for i in range(len(mu)):
        cov = L[i] @ L[i].T
        eigvals, eigvecs = torch.linalg.eigh(cov)
        angle_rad = torch.atan2(eigvecs[1, 1], eigvecs[0, 1])
        angle = angle_rad.item() * 180 / torch.pi

        width, height = scale * torch.sqrt(eigvals).cpu().numpy()
        widths.append(width)
        heights.append(height)
        angles.append(angle)

    draw_ellipses_halo(mu, widths, heights, angles)


def main(model_path: str):
    print(f"Using model at: {model_path}")

    set_seed(777)
    # Load model and data
    model = load(model_path)
    model.eval()

    _, _, test_loader = create_loaders_for_dataset(model.dataset)

    _, points, labels, _, mu_out, logvar_out = model_outputs(model, test_loader)

    indices = calculate_medoids(points, labels)
    
    mu_plot = mu_out[indices]
    logvar_plot = logvar_out[indices]

    plt.figure(figsize=(10, 8))

    plot_latent_2d(mu_out, labels)

    if isinstance(model, VaeGaussianFull):
        draw_cov_ellipses_filled(mu_plot, logvar_plot, scale=1.0)
    elif isinstance(model, VaeGaussianDiagonal):
        draw_diag_ellipses_filled(mu_plot, logvar_plot, scale=1.0)
    elif isinstance(model, VaeGaussianIsotropic):
        std_plot = torch.exp(0.5 * logvar_plot[:, 0])
        draw_std_ellipses_filled(mu_plot, std_plot, scale=1.0)

    plt.axis('square')

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Latent Space")

    plt.tight_layout()
    pdf_path = Path(f"./images/latent/{model.name}.pdf")
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(pdf_path, format="pdf")
    plt.show()

    if model.dataset != "har":
        plot_decoded_umap_grid(model)


def plot_decoded_umap_grid(ae_model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ae_model.to(device)
    ae_model.eval()

    coords = np.load(f"./preprocessed/{ae_model.dataset}/umap_grid_points.npy")
    n_points = coords.shape[0]
    grid_size = int(np.sqrt(n_points))

    assert grid_size * grid_size == n_points, "Grid points must form a square grid."

    # Decode
    z = torch.tensor(coords, dtype=torch.float32).to(device)
    with torch.no_grad():
        decoded = ae_model.decoder(z).cpu()

    # Reshape decoded outputs
    decoded = decoded.view(-1, 28, 28)

    # Plot grid
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(grid_size, grid_size))
    for idx in range(n_points):
        r, c = divmod(idx, grid_size)
        ax = axes[grid_size - 1 - r, c]
        ax.imshow(decoded[idx], cmap='gray')
        ax.axis('off')
    plt.tight_layout()

    pdf_path = Path(f"./images/grid/{ae_model.name}.pdf")
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(pdf_path)
    plt.show()


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Load a model from a specified path and show the latent space visualization.")
    parser.add_argument(
        "--model", 
        type=str, 
        required=True,
        help="path to the model file."
    )

    # Parse arguments
    args = parser.parse_args()

    # Call main with the parsed argument
    main(model_path=args.model)
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
import numpy as np
import torch

from data_loader import create_loaders_for_dataset, model_outputs
from trainer import set_seed
from vae_models import load, VaeGaussianDiagonal, VaeGaussianFull, VaeGaussianIsotropic

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def plot_latent_2d(mu, label, size=5):
    sizes = torch.ones(mu.size(0)) * size
    scatter = plt.scatter(mu[:, 0], mu[:, 1], c=label, s=sizes, cmap='tab10', alpha=0.2)


def plot_point_cloud(mu, std, n_samples=1000, color="blue"):
    eps = torch.randn(n_samples, 2)
    samples = mu + std * eps
    plt.scatter(samples[:, 0], samples[:, 1], s=0.5, c=color, alpha=0.1)


def draw_ellipses(mu, widths, heights, angles, labels, alpha=0.1, linewidth=0.5, cmap='tab10', ax=None):
    if ax is None:
        ax = plt.gca()
    colormap = plt.colormaps.get_cmap(cmap)

    for i in range(len(mu)):
        color = colormap(labels[i] % 10)
        ellipse = Ellipse(
            xy=mu[i],
            width=widths[i],
            height=heights[i],
            angle=angles[i],
            facecolor=color,
            edgecolor='none',
            alpha=alpha,
            linewidth=linewidth
        )
        ax.add_patch(ellipse)


def draw_ellipses_halo(mu, widths, heights, angles, labels, alpha=1, linewidth=2, ax=None):
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


def draw_std_ellipses_filled(mu, std, labels, scale=1.0, **kwargs):
    std = std.numpy()
    widths = heights = [scale * s for s in std]
    angles = [0.0] * len(std)
    draw_ellipses_halo(mu, widths, heights, angles, labels, **kwargs)


def draw_diag_ellipses_filled(mu, logvar, labels, scale=1.0, **kwargs):
    std = torch.exp(0.5 * logvar).numpy()
    widths = [scale * s[0] for s in std]
    heights = [scale * s[1] for s in std]
    angles = [0.0] * len(std)  # Axis-aligned
    draw_ellipses_halo(mu, widths, heights, angles, labels, **kwargs)


def draw_cov_ellipses_filled(mu, L, labels, scale=1.0, **kwargs):
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

    draw_ellipses_halo(mu, widths, heights, angles, labels, **kwargs)


def main(
        model_path="./models/vae-diag-har-p5.00-e0.00100-s8.pt"
):
    set_seed(777)
    # Load model and data
    model = load(model_path)
    model.eval()

    _, _, test_loader = create_loaders_for_dataset(model.dataset)

    vectors, _, labels, recon_out, mu_out, logvar_out = model_outputs(model, test_loader)

    # Get unique labels and prepare mask
    num_points_per_label = 3
    unique_labels = labels.unique()
    selected_indices = []

    for lbl in unique_labels:
        idx = (labels == lbl).nonzero(as_tuple=True)[0]
        if len(idx) >= num_points_per_label:
            selected = idx[:num_points_per_label]
        else:
            selected = idx  # take fewer if not enough
        selected_indices.append(selected)

    indices = torch.cat(selected_indices)

    mu_plot = mu_out[indices]
    logvar_plot = logvar_out[indices]
    label_plot = labels[indices]

    plt.figure(figsize=(10, 8))

    plot_latent_2d(mu_out, labels)

    if isinstance(model, VaeGaussianFull):
        draw_cov_ellipses_filled(mu_plot, logvar_plot, label_plot, scale=1.0)
    elif isinstance(model, VaeGaussianDiagonal):
        draw_diag_ellipses_filled(mu_plot, logvar_plot, label_plot, scale=1.0)
    elif isinstance(model, VaeGaussianIsotropic):
        std_plot = torch.exp(0.5 * logvar_plot[:, 0])
        draw_std_ellipses_filled(mu_plot, std_plot, label_plot, scale=1.0)

    plt.axis('square')

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Latent Space")

    plt.tight_layout()
    plt.savefig("./images/" + model.name + "-latent.pdf", format="pdf")
    plt.show()

    if model.dataset != "har":
        show_reconstructions(vectors, recon_out)
        plot_vae_decoded_umap_grid(model)


def show_reconstructions(values, recon_out, num_images=10):
    assert values.shape == recon_out.shape
    assert values.shape[0] >= num_images

    # Randomly sample 10 consistent indices
    indices = torch.randperm(values.shape[0])[:num_images]
    originals = values[indices].view(-1, 1, 28, 28)
    reconstructions = recon_out[indices].view(-1, 1, 28, 28)

    fig, axes = plt.subplots(2, num_images, figsize=(num_images * 1.2, 2.5))
    for i in range(num_images):
        axes[0, i].imshow(originals[i, 0], cmap='gray')
        axes[1, i].imshow(reconstructions[i, 0], cmap='gray')
        axes[0, i].axis('off')
        axes[1, i].axis('off')
    axes[0, 0].set_ylabel('Original', fontsize=12)
    axes[1, 0].set_ylabel('Reconstruction', fontsize=12)
    plt.tight_layout()
    plt.show()


def plot_vae_decoded_umap_grid(ae_model):
    coords = np.load(f"./{ae_model.dataset}_grid_points.npy")
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
    plt.savefig("./images/" + ae_model.name + "-grid.png")
    plt.show()


if __name__ == "__main__":
    main()

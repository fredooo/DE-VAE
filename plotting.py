import matplotlib.pyplot as plt
import torch

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

    plt.savefig(filename)
    #plt.show()

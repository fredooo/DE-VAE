import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms


class ProjectedLabeledDataset(Dataset):
    def __init__(self, indices, vectors, points_2d, labels):
        self.indices = indices
        self.vectors = vectors
        self.points_2d = points_2d
        self.labels = labels

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        vector = self.vectors[i]
        point_2d = self.points_2d[i]
        label = self.labels[i]
        return vector, point_2d, label


def load_mnist():
    transform = transforms.ToTensor()
    mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    vectors = torch.cat([mnist_train.data, mnist_test.data], dim=0).float().div(255).view(-1, 28*28)
    labels = torch.cat([mnist_train.targets, mnist_test.targets], dim=0)
    return vectors, labels


def load_fashion_mnist():
    transform = transforms.ToTensor()
    fashion_mnist_train = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    fashion_mnist_test = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    vectors = torch.cat([fashion_mnist_train.data, fashion_mnist_test.data], dim=0).float().div(255).view(-1, 28*28)
    labels = torch.cat([fashion_mnist_train.targets, fashion_mnist_test.targets], dim=0)
    return vectors, labels


def load_har():
    train = pd.read_csv('./data/HAR/train.csv')
    test = pd.read_csv('./data/HAR/test.csv')
    data = pd.concat([train, test], ignore_index=True)
    labels = data.iloc[:, -1:]
    labels = labels.to_numpy().flatten()
    labels = np.unique(labels, return_inverse=True)[1]
    data.drop(['subject', 'Activity'], axis=1, inplace=True)
    print("HAR vectors shape", data.to_numpy().shape)
    return torch.from_numpy(data.to_numpy()).float().view(-1, 561), torch.from_numpy(labels)


def load_csv_to_tensors(csv_path):
    df = pd.read_csv(csv_path)
    points_2d_tensor = torch.tensor(df[['x', 'y']].values, dtype=torch.float32)
    labels_tensor = torch.tensor(df['label'].values, dtype=torch.long)
    return points_2d_tensor, labels_tensor


def create_data_loaders(dataset, points_2d, labels, batch_size=64):
    dataset_size = len(labels)
    val_size = int(0.1 * dataset_size)
    test_size = int(0.1 * dataset_size)
    train_size = dataset_size - val_size - test_size

    full_indices = list(range(dataset_size))
    train_indices, val_test_indices = random_split(full_indices, [train_size, val_size + test_size])
    val_indices, test_indices = random_split(val_test_indices, [val_size, test_size])

    train_dataset = ProjectedLabeledDataset(train_indices.indices, dataset, points_2d, labels)
    val_dataset = ProjectedLabeledDataset(val_indices.indices, dataset, points_2d, labels)
    test_dataset = ProjectedLabeledDataset(test_indices.indices, dataset, points_2d, labels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def create_loaders_for_dataset(dataset_name):
    if dataset_name == "mnist":
        vectors, _ = load_mnist()
    elif dataset_name == "fmnist":
        vectors, _ = load_fashion_mnist()
    elif dataset_name == "har":
        vectors, _ = load_har()
    else:
        raise ValueError("Unrecognized model type in filename")
    points_2d, labels = load_csv_to_tensors(f"./{dataset_name}_umap.csv")
    return create_data_loaders(vectors, points_2d, labels)

def model_outputs(model, loader):
    vectors = []
    labels = []
    points = []
    recon_out = []
    mu_out = []
    logvar_out = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        for v, p, l in loader:
            x = v.to(device)
            recon, mu, logvar = model(x)
            vectors.append(v)
            points.append(p)
            labels.append(l)
            recon_out.append(recon.cpu())
            mu_out.append(mu.cpu())
            logvar_out.append(logvar.cpu())

    return (
        torch.cat(vectors, dim=0),
        torch.cat(points, dim=0),
        torch.cat(labels, dim=0),
        torch.cat(recon_out, dim=0),
        torch.cat(mu_out, dim=0),
        torch.cat(logvar_out, dim=0)
    )

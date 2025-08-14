import os
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from loss_functions import loss_gaussian_full, loss_gaussian_diagonal, loss_reg_mean


class EncoderBase(nn.Module):
    def __init__(self, io_dim, latent_dim):
        super().__init__()
        self.io_dim = io_dim
        self.latent_dim = latent_dim
        self.fc1 = nn.Linear(io_dim, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4_mu = nn.Linear(256, latent_dim)

    def forward_common(self, x):
        x = x.view(-1, self.io_dim)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x


class EncoderGaussianFull(EncoderBase):
    def __init__(self, io_dim, latent_dim):
        super().__init__(io_dim, latent_dim)
        self.fc4_L_param = nn.Linear(256, latent_dim * (latent_dim + 1) // 2)

    def forward(self, x):
        x = self.forward_common(x)
        mu = self.fc4_mu(x)
        L_params = self.fc4_L_param(x)
        return mu, L_params


class EncoderDiagonalGaussian(EncoderBase):
    def __init__(self, io_dim, latent_dim):
        super().__init__(io_dim, latent_dim)
        self.fc4_logvar = nn.Linear(256, latent_dim)

    def forward(self, x):
        x = self.forward_common(x)
        mu = self.fc4_mu(x)
        logvar = self.fc4_logvar(x)
        return mu, logvar


# Encoder with scalar logvar
class EncoderGaussianIsotropic(EncoderBase):
    def __init__(self, io_dim, latent_dim):
        super().__init__(io_dim, latent_dim)
        self.fc4_logvar = nn.Linear(256, 1)  # scalar log variance

    def forward(self, x):
        x = self.forward_common(x)
        mu = self.fc4_mu(x)
        logvar_scalar = self.fc4_logvar(x)
        logvar = logvar_scalar.expand(-1, self.latent_dim)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, io_dim, latent_dim):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, io_dim)

    def forward(self, z):
        z = F.relu(self.fc1(z))
        z = F.relu(self.fc2(z))
        z = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(z))


class AeBase(nn.Module):
    def __init__(self, dataset: str, projection: str, io_dim: int, latent_dim: int, l_proj: float, l_ent: float, seed: int):
        super().__init__()
        self.dataset = dataset
        self.projection = projection
        self.io_dim = io_dim
        self.latent_dim = latent_dim
        self.l_proj = l_proj
        self.l_ent = l_ent
        self.seed = seed
        self.name = f"{self.get_model_type()}-{dataset}-{projection}-p{l_proj:.2f}-e{l_ent:.5f}-s{seed}"
        self.decoder = Decoder(self.io_dim, latent_dim)

    @staticmethod
    def reparameterization(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def get_model_type(self):
        if isinstance(self, VaeGaussianFull):
            return "vae-full"
        elif isinstance(self, VaeGaussianDiagonal):
            return "vae-diag"
        elif isinstance(self, VaeGaussianIsotropic):
            return "vae-isot"
        elif isinstance(self, AeRegMean):
            return "ae-regm"
        else:
            raise ValueError("Unknown model type")


class VaeGaussianFull(AeBase):
    def __init__(self, dataset: str, projection: str, io_dim: int, latent_dim: int, loss_recon: nn.BCELoss | nn.MSELoss, l_proj: float, l_ent: float, seed: int):
        super().__init__(dataset, projection, io_dim, latent_dim, l_proj, l_ent, seed)
        self.loss_func = lambda a, b, c, d, e: loss_gaussian_full(a, b, c, d, e, l_proj, l_ent, loss_recon=loss_recon)
        self.encoder = EncoderGaussianFull(self.io_dim, latent_dim)

    @staticmethod
    def construct_L(L_params, latent_dim):
        batch_size = L_params.size(0)
        L = torch.zeros(batch_size, latent_dim, latent_dim, device=L_params.device)
        tril_idx = torch.tril_indices(latent_dim, latent_dim)
        L[:, tril_idx[0], tril_idx[1]] = L_params
        diag_indices = torch.arange(latent_dim)
        L[:, diag_indices, diag_indices] = torch.exp(L[:, diag_indices, diag_indices])
        return L

    def forward(self, x):
        mu, L_params = self.encoder(x)
        L = self.construct_L(L_params, self.latent_dim)
        eps = torch.randn(mu.size(0), self.latent_dim, 1, device=x.device)
        z = mu.unsqueeze(2) + torch.bmm(L, eps)
        z = z.squeeze(2)
        recon = self.decoder(z)
        return recon, mu, L


class VaeGaussianDiagonal(AeBase):
    def __init__(self, dataset: str, projection: str, io_dim: int, latent_dim: int, loss_recon: nn.BCELoss | nn.MSELoss, l_proj: float, l_ent: float, seed: int):
        super().__init__(dataset, projection, io_dim, latent_dim, l_proj, l_ent, seed)
        self.loss_func = lambda recon_x, x, mu, logvar, mu_target: loss_gaussian_diagonal(recon_x, x, mu, logvar, mu_target, l_proj, l_ent, loss_recon=loss_recon)
        self.encoder = EncoderDiagonalGaussian(self.io_dim, latent_dim)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterization(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar


class VaeGaussianIsotropic(AeBase):
    def __init__(self, dataset: str, projection: str, io_dim: int, latent_dim: int, loss_recon: nn.BCELoss | nn.MSELoss, l_proj: float, l_ent: float, seed: int):
        super().__init__(dataset, projection, io_dim, latent_dim, l_proj, l_ent, seed)
        self.loss_func = lambda recon_x, x, mu, logvar, mu_target: loss_gaussian_diagonal(recon_x, x, mu, logvar, mu_target, l_proj, l_ent, loss_recon=loss_recon)
        self.encoder = EncoderGaussianIsotropic(self.io_dim, latent_dim)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterization(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar


class AeRegMean(AeBase):
    def __init__(self, dataset: str, projection: str, io_dim: int, latent_dim: int, loss_recon: nn.BCELoss | nn.MSELoss, l_proj: float, l_ent: float, seed: int):
        super().__init__(dataset, projection, io_dim, latent_dim, l_proj, l_ent, seed)
        self.loss_func = lambda recon_x, x, mu, _, mu_target: loss_reg_mean(recon_x, x, mu, mu_target, l_proj, loss_recon=loss_recon)
        self.encoder = EncoderBase(self.io_dim, latent_dim)

    def forward(self, x):
        x = self.encoder.forward_common(x)
        y = self.encoder.fc4_mu(x)
        recon = self.decoder(y)
        return recon, y, torch.full((x.size(0), 1), -float('inf'))


def save(model, dir_path="models", verbose: bool = False):
    os.makedirs(dir_path, exist_ok=True)
    filename = f"{model.name}.pt"
    filepath = os.path.join(dir_path, filename)

    if verbose:
        print(model.name, "state_dict:")
        for param_tensor in model.state_dict():
            print(param_tensor, "\t", model.state_dict()[param_tensor].size())

    torch.save(model.state_dict(), filepath)


def create_model_from_params(model_type: str, dataset: str, projection: str, l_proj: float, l_ent: float, seed: int):
    io_dim = 28*28
    if dataset == "har":
        io_dim = 561
    elif "_" in dataset:
        io_dim = int(dataset.split("_")[1])
    
    loss_recon = nn.MSELoss(reduction='sum')
    if dataset == "mnist" or dataset == "fmnist" or dataset == "kmnist":
        loss_recon = nn.BCELoss(reduction='sum')

    if model_type == "vae-full":
        model = VaeGaussianFull(dataset, projection, io_dim, 2, loss_recon, l_proj, l_ent, seed)
    elif model_type == "vae-diag":
        model = VaeGaussianDiagonal(dataset, projection, io_dim, 2, loss_recon, l_proj, l_ent, seed)
    elif model_type == "vae-isot":
        model = VaeGaussianIsotropic(dataset, projection, io_dim, 2, loss_recon, l_proj, l_ent, seed)
    elif model_type == "ae-regm":
        model = AeRegMean(dataset, projection, io_dim, 2, loss_recon, l_proj, 0.0, seed)
    else:
        raise ValueError("Unrecognized model type in filename")
    return model


def load(filepath):
    filename = os.path.basename(filepath)
    match = re.match(r"(vae-full|vae-diag|vae-isot|ae-regm)-(mnist|fmnist|kmnist|har|[a-z]+_\d+)-(umap|tsne|lle|mds|pca|isomap)-p([\d.]+)-e([\d.]+)-s(\d+)\.pt", filename)
    if not match:
        raise ValueError("Filename format is invalid")
    model_type, dataset, projection, l_proj, l_ent, seed = match.groups()
    l_proj = float(l_proj)
    l_ent = float(l_ent)
    seed = int(seed)
    model = create_model_from_params(model_type, dataset, projection, l_proj, l_ent, seed)
    model.load_state_dict(torch.load(filepath))
    return model

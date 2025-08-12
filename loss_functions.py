import math
import torch
import torch.nn as nn


def kl_divergence(mu, logvar):
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())  # / x.size(0)


def kl_divergence_cholesky(mu, L):
    diag = torch.diagonal(L, dim1=1, dim2=2)
    log_det = 2 * torch.sum(torch.log(torch.clamp(diag, min=1e-8)), dim=1)
    trace = torch.sum(L ** 2, dim=(1, 2))
    mu_term = torch.sum(mu ** 2, dim=1)
    k = L.size(1)
    kl = 0.5 * (trace + mu_term - k - log_det)
    kl = torch.sum(kl)  # / x.size(0)
    return kl


def differential_entropy(logvar):
    return 0.5 * torch.sum(1 + logvar + math.log(2 * math.pi), dim=1).sum()


def differential_entropy_cholesky(L):
    latent_dim = L.size(1)
    diag_L = torch.diagonal(L, dim1=1, dim2=2)
    log_det_cov = 2.0 * torch.sum(torch.log(diag_L + 1e-8), dim=1)
    constant = 0.5 * latent_dim * (1 + math.log(2 * math.pi))
    entropy = constant + 0.5 * log_det_cov
    return entropy.sum()


def loss_gaussian_diagonal(recon_x, x, mu, logvar, mu_target, l_proj, l_ent,
                           loss_recon: nn.BCELoss | nn.MSELoss = nn.BCELoss(reduction='sum'),
                           loss_proj=nn.MSELoss(reduction='sum')
):
    recon = loss_recon(recon_x, x)
    proj = loss_proj(mu, mu_target)
    entropy = differential_entropy(logvar)
    loss = recon + l_proj * proj - (l_ent * entropy)
    return loss, recon, proj, entropy


def loss_gaussian_full(recon_x, x, mu, L, mu_target, l_proj, l_ent,
                       loss_recon: nn.MSELoss | nn.BCELoss = nn.BCELoss(reduction='sum'),
                       loss_proj=nn.MSELoss(reduction='sum')
):
    recon = loss_recon(recon_x, x)
    proj = loss_proj(mu, mu_target)
    entropy = differential_entropy_cholesky(L)
    loss = recon + l_proj * proj - (l_ent * entropy)
    return loss, recon, proj, entropy


def loss_reg_mean(recon_x, x, mu, mu_target, l_proj=1.0,
                  loss_recon: nn.MSELoss | nn.BCELoss = nn.BCELoss(reduction='sum'),
                  loss_proj=nn.MSELoss(reduction='sum')
):
    recon = loss_recon(recon_x, x)
    proj = loss_proj(mu, mu_target)
    loss = recon + l_proj * proj
    return loss, recon, proj, torch.tensor(-float('inf'))
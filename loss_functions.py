import math
import torch
import torch.nn as nn


def differential_entropy(logvar):
    constant = torch.log(torch.tensor(2 * math.pi, device=logvar.device))
    return 0.5 * torch.sum(1 + logvar + constant, dim=1).sum()


def differential_entropy_cholesky(L):
    latent_dim = L.size(1)
    diag_L = torch.diagonal(L, dim1=1, dim2=2)
    log_det_cov = 2.0 * torch.sum(torch.log(diag_L + 1e-8), dim=1)
    constant = 0.5 * latent_dim * (1 + torch.log(torch.tensor(2 * math.pi, device=L.device)))
    entropy = constant + 0.5 * log_det_cov
    return entropy.sum()


def loss_gaussian_diagonal(recon_x, x, mu, logvar, mu_target, l_proj, l_ent,
                           loss_recon: nn.BCELoss | nn.MSELoss = None,
                           loss_proj=None
):
    if loss_recon is None:
        loss_recon = nn.BCELoss(reduction='sum')
    if loss_proj is None:
        loss_proj = nn.MSELoss(reduction='sum')
    recon = loss_recon(recon_x, x)
    proj = loss_proj(mu, mu_target)
    entropy = differential_entropy(logvar)
    loss = recon + l_proj * proj - (l_ent * entropy)
    return loss, recon, proj, entropy


def loss_gaussian_full(recon_x, x, mu, L, mu_target, l_proj, l_ent,
                       loss_recon: nn.MSELoss | nn.BCELoss = None,
                       loss_proj=None
):
    if loss_recon is None:
        loss_recon = nn.BCELoss(reduction='sum')
    if loss_proj is None:
        loss_proj = nn.MSELoss(reduction='sum')
    recon = loss_recon(recon_x, x)
    proj = loss_proj(mu, mu_target)
    entropy = differential_entropy_cholesky(L)
    loss = recon + l_proj * proj - (l_ent * entropy)
    return loss, recon, proj, entropy


def loss_reg_mean(recon_x, x, mu, mu_target, l_proj=1.0,
                  loss_recon: nn.MSELoss | nn.BCELoss = None,
                  loss_proj=None
):
    if loss_recon is None:
        loss_recon = nn.BCELoss(reduction='sum')
    if loss_proj is None:
        loss_proj = nn.MSELoss(reduction='sum')
    recon = loss_recon(recon_x, x)
    proj = loss_proj(mu, mu_target)
    loss = recon + l_proj * proj
    return loss, recon, proj, torch.tensor(-float('inf'), device=x.device)
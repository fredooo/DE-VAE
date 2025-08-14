import argparse
import gc
import numpy as np
import pandas as pd
import random
import time
import torch

from data_loader import create_loaders_for_dataset
from vae_models import create_model_from_params, save

# Hyperparameters
max_epochs = 100
lr = 1e-3

# Determine CUDA device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed=777):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def log_print(model, line):
    with open(f"./records/{model.name}.txt", "a") as f:
        print(line, file=f)
    print(line)


def loss_log_string(loss_name, values):
    return f"{loss_name} Loss = {values[0]:.2f} (recon: {values[1]:.2f}, proj: {values[2]:.2f}, ent: {values[3]:.2f}, [std: {values[4]:.2f}])"


def training_step(train_loader, optimizer, model):
    model.train()
    losses = torch.zeros(5)
    for v, p, _ in train_loader:
        x = v.to(device)
        y = p.to(device)
        optimizer.zero_grad()
        recon_x, mu, logvar = model(x)
        loss, recon, proj, ent = model.loss_func(recon_x, x, mu, logvar, y)
        loss.backward()
        avg_std = torch.exp(0.5 * logvar).sum().item()
        losses += torch.tensor([loss.item(), recon.item(), proj.item(), ent.item(), avg_std])
        optimizer.step()
    avg_losses = losses / len(train_loader.dataset)
    return avg_losses


def validation_step(val_loader, model):
    model.eval()
    losses = torch.zeros(5)
    with torch.no_grad():
        for v, p, _ in val_loader:
            x = v.to(device)
            y = p.to(device)
            recon, mu, logvar = model(x)
            loss, recon, proj, ent = model.loss_func(recon, x, mu, logvar, y)
            avg_std = torch.exp(0.5 * logvar).sum().item()
            losses += torch.tensor([loss.item(), recon.item(), proj.item(), ent.item(), avg_std])
    avg_loss = losses / len(val_loader.dataset)
    return avg_loss


def store_losses(losses_list, filename):
    data = torch.stack(losses_list)
    df = pd.DataFrame(data.numpy(), columns=["Loss", "Recon", "Proj", "Ent", "AvgStd"])
    df.index.name = "Epoch"
    df.to_csv(filename, index=True)


class ValidationSaveStop:
    def __init__(self, patience=5):
        self.best_val_loss = float('inf')
        self.counter = 0
        self.patience = patience

    def check(self, val_loss, model):
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.counter = 0
            save(model)
            return "SAVED"
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return "STOP"
            return ""


def train(model_type: str, dataset: str, projection: str, l_proj: float, l_ent: float, seed: int, data_loaders = None):
    set_seed(seed)

    # Load dataset
    if data_loaders is None:
        train_loader, val_loader, test_loader = create_loaders_for_dataset(dataset, projection)
    else:
        train_loader, val_loader, test_loader = data_loaders

    # Initialize model
    model = create_model_from_params(model_type, dataset, projection, l_proj, l_ent, seed).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training loop
    log_print(model, f"Training: {model.name}")

    save_stop = ValidationSaveStop()
    train_records = []
    val_records = []

    start = time.time()

    for epoch in range(max_epochs):
        avg_train_losses = training_step(train_loader, optimizer, model)
        train_records.append(avg_train_losses)
        train_log = loss_log_string("Train", avg_train_losses)

        avg_val_losses = validation_step(val_loader, model)
        val_records.append(avg_val_losses)
        val_log = loss_log_string("Val", avg_val_losses)

        status = save_stop.check(avg_val_losses[0], model)
        status_box = f"[{status}]" if status else ""
        log_print(model, f"Epoch {epoch + 1}: {train_log} - {val_log} {status_box}")
        if status == "STOP":
            break

    log_print(model, f"Training Time: {time.time() - start} seconds")

    store_losses(train_records, f"./records/{model.name}-train-records.csv")
    store_losses(val_records, f"./records/{model.name}-val-records.csv")

    # Evaluate on test set
    avg_test_losses = validation_step(test_loader, model)
    test = loss_log_string("Test", avg_test_losses)
    log_print(model, test)


def run_full():
    models = ["ae-regm", "vae-full", "vae-diag", "vae-isot"]
    seeds = [777, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    datasets = ["mnist"]
    projections = ["umap"]
    l_proj = 20.0
    l_ent = 5.0
    for m in models:
        for d in datasets:
            for p in projections:
                for s in seeds:
                    train(m, d, p, l_proj, l_ent, s)
                    print(f"GC cleared {gc.collect()} objects")

    datasets = ["fmnist"]
    projections = ["umap"]
    l_proj = 20.0
    l_ent = 4.0
    for m in models:
        for d in datasets:
            for p in projections:
                for s in seeds:
                    train(m, d, p, l_proj, l_ent, s)
                    print(f"GC cleared {gc.collect()} objects")

    datasets = ["kmnist"]
    projections = ["umap"]
    l_proj = 20.0
    l_ent = 1.0
    for m in models:
        for d in datasets:
            for p in projections:
                for s in seeds:
                    train(m, d, p, l_proj, l_ent, s)
                    print(f"GC cleared {gc.collect()} objects")

    datasets = ["har"]
    projections = ["umap"]
    l_proj = 5.0
    l_ent = 0.001
    for m in models:
        for d in datasets:
            for p in projections:
                for s in seeds:
                    train(m, d, p, l_proj, l_ent, s)
                    print(f"GC cleared {gc.collect()} objects")

    datasets = ["har", "mnist", "fmnist", "kmnist"]
    projections = ["pca", "tsne"]
    l_proj = 20.0
    l_ent = 1.0
    for m in models:
        for d in datasets:
            for p in projections:
                for s in seeds:
                    train(m, d, p, l_proj, l_ent, s)
                    print(f"GC cleared {gc.collect()} objects")


if __name__ == "__main__":
    run_full()
import glob
import math
import os
import pandas as pd
import re

RECORDS_DIR = "./records"  # root folder that holds the *.txt files
DATASETS = ["mnist", "fmnist", "kmnist", "har"]
MODELS = ["ae-regm", "vae-isot", "vae-diag", "vae-full"]
PROJECTIONS = ["umap",  "tsne", "pca", "mds", "isomap", "lle"]
SEEDS = range(10)

MODEL_NAMES = {
    "vae-isot": "VAE \\textbf{Isotropic} Gaussian",
    "vae-diag": "VAE \\textbf{Diagonal} Gaussian",
    "vae-full": "VAE \\textbf{Full} Gaussian",
    "ae-regm": "AE Mean Reg. (\\textbf{None})"}

DATASET_NAMES = {
    "har": "HAR",
    "mnist": "MNIST",
    "fmnist": "Fashion-MNIST",
    "kmnist": "KMNIST"}

PROJECTION_NAMES = {
    "pca": "PCA",
    "mds": "MDS",
    "tsne": "t-SNE",
    "umap": "UMAP",
    "isomap": "Isomap",
    "lle": "LLE"}
    

num_pattern = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|[+-]?inf|nan"


def safe_float(text: str):
    try:
        val = float(text)
        if math.isinf(val) or math.isnan(val):
            return None
        return val
    except ValueError:
        return None


def parse_file(path: str):
    training_time = None
    recon = proj = ent = None
    epoch_count = 0

    with open(path, "r") as fh:
        lines = fh.readlines()

    # get number of epochs
    epoch_count = sum(1 for ln in lines if ln.startswith("Epoch "))

    # training time
    for ln in lines:
        if ln.startswith("Training Time:"):
            m = re.search(num_pattern, ln, re.I)
            training_time = safe_float(m.group(0)) if m else None
            break

    # final test losses (scan backwards; first match is the last "Test Loss")
    for ln in reversed(lines):
        if ln.startswith("Test Loss"):
            m = re.search(
                rf"recon:\s*({num_pattern}),\s*proj:\s*({num_pattern}),\s*ent:\s*({num_pattern})",
                ln,
                re.I,
            )
            if m:
                recon = safe_float(m.group(1))
                proj = safe_float(m.group(2))
                ent = safe_float(m.group(3))
            break

    return dict(
        TrainingTime=training_time,
        Recon=recon,
        Proj=proj,
        Ent=ent,
        Epochs=epoch_count,
    )


def summarise(df: pd.DataFrame):
    row = []
    for col in df.columns:
        vals = df[col].dropna().to_numpy(dtype=float)
        if vals.size:
            row.append(f"${vals.mean():.3f} \\pm {vals.std(ddof=0):.3f}$")
        else:
            row.append("---")
    return row


def create_single_table(model, dataset, projection):
    # gather one DataFrame with 10 rows (one per seed)
    rows = {}
    for seed in SEEDS:
        pattern = f"{model}-{dataset}-p*-e*-s{seed}.txt"
        matches = glob.glob(os.path.join(RECORDS_DIR, pattern))
        if not matches:
            continue  # file missing - skip seed
        rows[seed] = parse_file(matches[0])

    if not rows:
        # No files for this (dataset, model); skip silently
        return None

    df = pd.DataFrame.from_dict(rows, orient="index")
    
    # add aggregation row
    row = summarise(df)
    df.loc[len(df)] = row

    df["Seed"] = df.index.astype(str)
    df.loc[len(df) - 1, "Seed"] = "$\\mu \\pm \\sigma$"

    df = df[["Seed", "TrainingTime", "Recon", "Proj", "Ent", "Epochs"]]

    df.rename(columns={
        "TrainingTime": "Training Time (s)",
        "Recon": "$\\altmathcal{L}_{\\text{recon}}$",
        "Proj": "$\\altmathcal{L}_{\\text{proj}}$",
        "Ent": "$\\altmathcal{L}_{\\text{ent}}$"
    }, inplace=True)
    
    # pretty LaTeX
    latex_table = df.to_latex(
        index=False,
        float_format="%.3f",
        na_rep="---",
        escape=False,
        column_format="lrrrrr",
        caption=f"{MODEL_NAMES[model]} on {DATASET_NAMES[dataset]} with {PROJECTION_NAMES[projection]}: Test metrics and training time (10 seeds)",
        label=f"tab:{model}-{dataset}-{projection}",
        position="!h"
    )

    latex_table = latex_table.replace(
        "\\begin{table}[!h]",
        "\\begin{table}[!h]\n\\centering"
    ).replace(
        "\n$\\mu \\pm \\sigma$",
        "\n\\midrule\n$\\mu \\pm \\sigma$"
    )

    return latex_table


def run_full():
    for dataset in DATASETS:
        for projection in PROJECTIONS:
            print("\\subsubsection{" + DATASET_NAMES[dataset] + " with " + PROJECTION_NAMES[projection] + "}")
            for model in MODELS:
                latex_table = create_single_table(model, dataset, projection)
                print(latex_table)
            print("\\clearpage\n")


if __name__ == "__main__":
    run_full()
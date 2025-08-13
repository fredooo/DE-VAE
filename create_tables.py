import argparse
import glob
import math
import os
import pandas as pd
import re
from tabulate import tabulate

RECORDS_DIR = "./records"  # root folder that holds the *.txt files
DATASETS = ["mnist", "fmnist", "kmnist", "har"]
MODELS = ["ae-regm", "vae-isot", "vae-diag", "vae-full"]
PROJECTIONS = ["umap",  "tsne", "pca", "mds", "isomap", "lle"]
SEEDS = range(10)

MODEL_NAMES_LATEX = {
    "vae-isot": "VAE \\textbf{Isotropic} Gaussian",
    "vae-diag": "VAE \\textbf{Diagonal} Gaussian",
    "vae-full": "VAE \\textbf{Full} Gaussian",
    "ae-regm": "AE Mean Reg. (\\textbf{None})"}

MODEL_NAMES_TERMINAL = {
    "vae-isot": "VAE Isotropic Gaussian",
    "vae-diag": "VAE Diagonal Gaussian",
    "vae-full": "VAE Full Gaussian",
    "ae-regm": "AE Mean Reg. (None)"}

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
            row.append(f"{vals.mean():.3f} ± {vals.std(ddof=0):.3f}")
        else:
            row.append("---")
    return row


def create_single_table(model, dataset, projection):
    # gather one DataFrame with 10 rows (one per seed)
    rows = {}
    for seed in SEEDS:
        pattern = f"{model}-{dataset}-{projection}-p*-e*-s{seed}.txt"
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
    df.loc[len(df) - 1, "Seed"] = "μ ± σ"

    df = df[["Seed", "TrainingTime", "Recon", "Proj", "Ent", "Epochs"]]

    return df


def df_to_latex_str(df, model, dataset, projection):
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
        column_format="cccccc",
        caption=f"{MODEL_NAMES_LATEX[model]} on {DATASET_NAMES[dataset]} with {PROJECTION_NAMES[projection]}: Test metrics and training time (10 seeds)",
        label=f"tab:{model}-{dataset}-{projection}",
        position="!h"
    )

    latex_table = latex_table.replace(
        "\\begin{table}[!h]",
        "\\begin{table}[!h]\n\\centering"
    ).replace(
        "\nμ ± σ",
        "\n\\midrule\n$\\mu \\pm \\sigma$"
    ).replace(
        "±",
        "$\\pm$"
    )

    return latex_table


def run_full():
    for dataset in DATASETS:
        for projection in PROJECTIONS:
            out = ""
            for model in MODELS:
                df = create_single_table(model, dataset, projection)
                if df is not None:
                    latex_table = df_to_latex_str(df, model, dataset, projection)
                    out += latex_table
            if len(out) > 0:
                print("\\subsubsection{" + DATASET_NAMES[dataset] + " with " + PROJECTION_NAMES[projection] + "}\n")
                print(out)
                print("\\clearpage\n")

def print_df_on_terminal(df, model, dataset, projection):
    if df is None:
        print("No data available for the given combination.")
        return
    df.rename(columns={
        "TrainingTime": "Training Time (s)",
        "Recon": "L_recon",
        "Proj": "L_proj",
        "Ent": "L_ent"
    }, inplace=True)
    pretty = tabulate(df, headers='keys', tablefmt='pretty', floatfmt=".3f")
    print(f"{MODEL_NAMES_TERMINAL[model]} on {DATASET_NAMES[dataset]} with {PROJECTION_NAMES[projection]}:")
    print(pretty)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate and print evaluation tables.")

    parser.add_argument("--model", help="model name key (e.g., 'vae-full')")
    parser.add_argument("--dataset", help="dataset name key (e.g., 'mnist')")
    parser.add_argument("--projection", help="projection name key (e.g., 'umap')")
    parser.add_argument("--all-latex", action="store_true", help="Run full evaluation to generate all LaTeX tables")

    args = parser.parse_args()

    if args.all_latex:
        run_full()

    elif args.model and args.dataset and args.projection:
        df = create_single_table(args.model, args.dataset, args.projection)
        print_df_on_terminal(df, args.model, args.dataset, args.projection)

    else:
        print("Warning - No output. You must provide either:")
        print("1. All of --model, --dataset, and --projection")
        print("OR")
        print("2. Just --all-latex")
import glob
import math
import os
import pandas as pd
import re

RECORDS_DIR = "./records"  # root folder that holds the *.txt files
DATASETS = {"har", "mnist", "fmnist"}
MODELS = {"vae-isot", "vae-diag", "vae-full", "ae-regm"}
SEEDS = range(10)

MODEL_NAMES = {
    "vae-isot": "VAE \\textbf{Full} Gaussian",
    "vae-diag": "VAE \\textbf{Diagonal} Gaussian",
    "vae-full": "VAE \\textbf{Full} Gaussian",
    "ae-regm": "AE Mean Reg. (\textbf{None})"}

DATASET_NAMES = {
    "har": "HAR",
    "mnist": "MNIST",
    "fmnist": "Fashion-MNIST",
    "kmnist": "KMNIST"}

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
    out = {}
    for col in df.columns:
        vals = df[col].dropna().to_numpy(dtype=float)
        if vals.size:
            out[col] = f"${vals.mean():.3f} \\pm {vals.std(ddof=0):.3f}$"
        else:
            out[col] = "--"
    return pd.Series(out, name="$\\mu \\pm \\sigma$")


def run_full():
    for dataset in sorted(DATASETS):
        for model in sorted(MODELS):
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
                continue

            df = pd.DataFrame.from_dict(rows, orient="index")
            
            df["Seed"] = df.index

            df = df[["Seed", "TrainingTime", "Recon", "Proj", "Ent", "Epochs"]]

            # add aggregate row
            df = pd.concat([df, summarise(df)], axis=0)

            df.rename(columns={
                    "TrainingTime": "Training Time (s)",
                    "Recon": "$\\altmathcal{L}_{\\text{recon}}$",
                    "Proj": "$\\altmathcal{L}_{\\text{proj}}$",
                    "Ent": "$\\altmathcal{L}_{\\text{ent}}$"}, inplace=True)
            
            # pretty LaTeX
            latex_table = df.to_latex(
                index=False,
                formatters={"Seed": lambda x: f"{math.floor(x)}"},
                float_format="%.3f",
                na_rep="---",
                escape=False,
                column_format="lrrrrrr",
                caption=f"{MODEL_NAMES[model]} on {DATASET_NAMES[dataset]}: Test metrics and training time (10 seeds)",
                label=f"tab:{model}-{dataset}",
                position="!h"
            )

            print(latex_table)
            print("\n")


if __name__ == "__main__":
    run_full()
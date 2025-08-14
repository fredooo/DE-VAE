import argparse
from pathlib import Path
import torch

from data_loader import create_data_loaders, load_user_csv
from projections import project_data
from trainer import train


def main(model: str, data_path: str, label_col, projection, l_proj, l_ent, seed: int):
    name, vectors, labels = load_user_csv(data_path, label_col)
    _, embedding = project_data(vectors, labels, name, projection)
    points_2d_tensor = torch.tensor(embedding, dtype=torch.float32)
    loaders = create_data_loaders(vectors, points_2d_tensor, labels, batch_size=64)
    train(model, name, projection, l_proj, l_ent, seed, data_loaders=loaders)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate and print evaluation tables.")

    parser.add_argument("--model", required=True, help="model name key (e.g., 'vae-full')")
    parser.add_argument("--data", required=True, help="high-dimensional data as CSV file")
    parser.add_argument("--label", help="Column name specifing class labels")
    parser.add_argument("--projection", required=True, help="2D projection data as CSV file")

    parser.add_argument("--l-proj", type=float, default=1.0, help="projection loss weight")
    parser.add_argument("--l-ent", type=float, default=1.0, help="entropy loss weight")
    parser.add_argument("--seed", type=int, default=777, help="random seed for reproducibility")
    args = parser.parse_args()

    main(args.model, args.data, args.label, args.projection, args.l_proj, args.l_ent, args.seed)


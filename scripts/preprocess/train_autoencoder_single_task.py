#!/usr/bin/env python3
"""Train or refresh a single-task D-MoLE autoencoder artifact."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from internvl.model.autoencoder import AutoEncoder


QUANTILES = [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]


def l2_normalize(embeddings: torch.Tensor) -> torch.Tensor:
    norm = embeddings.norm(p=2, dim=1, keepdim=True)
    return embeddings / norm.clamp_min(1e-12)


def train_autoencoder(
    model: AutoEncoder,
    embeddings: torch.Tensor,
    *,
    device: torch.device,
    num_epochs: int,
    batch_size: int,
    patience: int,
) -> AutoEncoder:
    dataset = TensorDataset(embeddings)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-3
    )

    best_loss = float("inf")
    stagnant_epochs = 0
    model.train()

    for epoch in range(num_epochs):
        total_loss = 0.0
        for batch, in dataloader:
            batch = batch.to(device)
            optimizer.zero_grad()
            outputs = model(batch)
            loss = criterion(outputs, batch)
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"autoencoder loss became non-finite at epoch {epoch + 1}"
                )
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        total_loss /= max(1, len(dataloader))
        scheduler.step()
        print(f"epoch {epoch + 1}/{num_epochs} loss={total_loss:.8f}", flush=True)

        if total_loss < best_loss:
            best_loss = total_loss
            stagnant_epochs = 0
        else:
            stagnant_epochs += 1
            if stagnant_epochs >= patience:
                print(
                    f"early stop after epoch {epoch + 1} with best_loss={best_loss:.8f}",
                    flush=True,
                )
                break

    return model


def write_quantile_table(
    *,
    task_name: str,
    losses: torch.Tensor,
    csv_path: Path,
) -> None:
    quantile_values = torch.quantile(
        losses, torch.tensor(QUANTILES, device=losses.device)
    ).detach()
    row = {
        "min": losses.min().item(),
        "Q10": quantile_values[0].item(),
        "Q25": quantile_values[1].item(),
        "Q50": quantile_values[2].item(),
        "Q75": quantile_values[3].item(),
        "Q90": quantile_values[4].item(),
        "Q95": quantile_values[5].item(),
        "Q99": quantile_values[6].item(),
        "max": losses.max().item(),
        "mean": losses.mean().item(),
        "std": losses.std().item(),
    }

    if csv_path.exists():
        df = pd.read_csv(csv_path, index_col=0)
    else:
        df = pd.DataFrame(
            columns=["min", "Q10", "Q25", "Q50", "Q75", "Q90", "Q95", "Q99", "max", "mean", "std"]
        )

    df.loc[task_name] = row
    df = df.sort_index()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-name", required=True)
    parser.add_argument("--embedding-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument(
        "--threshold-quantile",
        type=float,
        default=0.95,
        help="Quantile used to materialize threshold.txt for task routing.",
    )
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help="Retrain even if autoencoder.pt already exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    task_dir = args.output_dir / args.task_name
    task_dir.mkdir(parents=True, exist_ok=True)

    embeddings = torch.load(args.embedding_path, map_location=device).to(torch.float32)
    if not torch.isfinite(embeddings).all():
        nan_count = int(torch.isnan(embeddings).sum().item())
        inf_count = int(torch.isinf(embeddings).sum().item())
        raise RuntimeError(
            "embedding tensor contains non-finite values before autoencoder training: "
            f"nan_count={nan_count}, inf_count={inf_count}"
        )
    embeddings = l2_normalize(embeddings).to(device)
    input_dim = embeddings.shape[1]

    autoencoder_path = task_dir / "autoencoder.pt"
    quantile_csv_path = args.output_dir / "reconstruction_loss_quantiles.csv"
    threshold_path = task_dir / "threshold.txt"

    model = AutoEncoder(input_dim=input_dim, hidden_dim=args.hidden_dim).to(device)
    if autoencoder_path.exists() and not args.force_retrain:
        print(f"loading autoencoder from {autoencoder_path}", flush=True)
        state_dict = torch.load(autoencoder_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print(f"training autoencoder for {args.task_name}", flush=True)
        model = train_autoencoder(
            model,
            embeddings,
            device=device,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            patience=args.patience,
        )
        torch.save(model.state_dict(), autoencoder_path)
        print(f"saved autoencoder to {autoencoder_path}", flush=True)

    model.eval()
    losses = model.compute_reconstruction_loss(embeddings).detach()
    if not torch.isfinite(losses).all():
        nan_count = int(torch.isnan(losses).sum().item())
        inf_count = int(torch.isinf(losses).sum().item())
        raise RuntimeError(
            "reconstruction losses contain non-finite values: "
            f"nan_count={nan_count}, inf_count={inf_count}"
        )
    threshold = torch.quantile(
        losses, torch.tensor(args.threshold_quantile, device=losses.device)
    ).item()
    threshold_path.write_text(f"{threshold:.10f}\n", encoding="utf-8")
    write_quantile_table(
        task_name=args.task_name,
        losses=losses,
        csv_path=quantile_csv_path,
    )
    print(f"wrote threshold to {threshold_path}", flush=True)
    print(f"updated quantile table at {quantile_csv_path}", flush=True)


if __name__ == "__main__":
    main()

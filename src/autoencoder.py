from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass(frozen=True)
class AEConfig:
    input_dim: int
    latent_dim: int = 16
    hidden_dim: int = 256
    lr: float = 1e-3
    batch_size: int = 64
    epochs: int = 50
    seed: int = 42


class MLPAutoEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(int(input_dim), int(hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim), int(latent_dim)),
        )
        self.decoder = nn.Sequential(
            nn.Linear(int(latent_dim), int(hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim), int(input_dim)),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon, z


def train_autoencoder(
    X: np.ndarray,
    cfg: AEConfig,
    device: str | None = None,
    verbose: bool = False,
) -> MLPAutoEncoder:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = MLPAutoEncoder(cfg.input_dim, cfg.hidden_dim, cfg.latent_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg.lr))

    x_tensor = torch.tensor(X, dtype=torch.float32)
    loader = DataLoader(TensorDataset(x_tensor), batch_size=int(cfg.batch_size), shuffle=True)

    model.train()
    for epoch in range(int(cfg.epochs)):
        epoch_loss = 0.0
        n_batches = 0
        for (xb,) in loader:
            xb = xb.to(device)
            recon, _z = model(xb)
            loss = nn.functional.mse_loss(recon, xb, reduction="mean")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            epoch_loss += float(loss.detach().cpu().item())
            n_batches += 1

        if verbose and (epoch == 0 or (epoch + 1) % 10 == 0 or (epoch + 1) == int(cfg.epochs)):
            denom = max(1, n_batches)
            print(f"[AE] epoch {epoch + 1}/{int(cfg.epochs)} loss={epoch_loss / denom:.6f}")

    return model


def encode_ae_latents(
    model: MLPAutoEncoder,
    X: np.ndarray,
    *,
    batch_size: int = 512,
    device: str | None = None,
) -> np.ndarray:
    if int(batch_size) <= 0:
        raise ValueError("batch_size must be positive")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    def _encode_on(device_name: str) -> np.ndarray:
        model.eval()
        model.to(device_name)

        x_tensor = torch.tensor(X, dtype=torch.float32)
        loader = DataLoader(
            TensorDataset(x_tensor),
            batch_size=int(batch_size),
            shuffle=False,
            pin_memory=(device_name == "cuda"),
        )

        out: list[np.ndarray] = []
        with torch.no_grad():
            for (xb,) in loader:
                xb = xb.to(device_name, non_blocking=True)
                _recon, z = model(xb)
                out.append(z.detach().cpu().numpy())
        return np.concatenate(out, axis=0)

    try:
        return _encode_on(device)
    except torch.OutOfMemoryError:
        if device != "cuda":
            raise
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return _encode_on("cpu")

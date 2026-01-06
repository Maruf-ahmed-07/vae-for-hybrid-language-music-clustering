from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass(frozen=True)
class VAEConfig:
    input_dim: int
    latent_dim: int = 16
    hidden_dim: int = 256
    lr: float = 1e-3
    batch_size: int = 64
    epochs: int = 30
    beta: float = 1.0
    seed: int = 42


class MLPVAE(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.mu(h), self.logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


def vae_loss(
    x: torch.Tensor,
    recon: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0,
) -> torch.Tensor:
    recon_loss = nn.functional.mse_loss(recon, x, reduction="mean")
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kld


def train_vae(
    X: np.ndarray,
    cfg: VAEConfig,
    device: str | None = None,
) -> MLPVAE:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = MLPVAE(cfg.input_dim, cfg.hidden_dim, cfg.latent_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    x_tensor = torch.tensor(X, dtype=torch.float32)
    loader = DataLoader(TensorDataset(x_tensor), batch_size=cfg.batch_size, shuffle=True)

    model.train()
    for _ in range(cfg.epochs):
        for (xb,) in loader:
            xb = xb.to(device)
            recon, mu, logvar = model(xb)
            loss = vae_loss(xb, recon, mu, logvar, beta=cfg.beta)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

    return model


def encode_latents(model: MLPVAE, X: np.ndarray, device: str | None = None) -> np.ndarray:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model.eval()
    with torch.no_grad():
        x_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        mu, _logvar = model.encode(x_tensor)
        return mu.cpu().numpy()

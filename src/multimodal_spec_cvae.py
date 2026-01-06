from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass(frozen=True)
class MultiModalSpecCVAEConfig:
    n_mels: int = 64
    n_frames: int = 256
    lyrics_dim: int = 256
    cond_dim: int = 8  # number of genres

    latent_dim: int = 16
    hidden_channels: int = 128
    lyrics_hidden_dim: int = 256

    lr: float = 1e-3
    batch_size: int = 32
    epochs: int = 60
    beta: float = 1.0

    audio_recon_weight: float = 1.0
    lyrics_recon_weight: float = 1.0

    seed: int = 42


class MultiModalSpecCVAE(nn.Module):
    """Conditional multimodal VAE.

    Audio input: (B, 1, n_mels, n_frames)
    Lyrics input: (B, lyrics_dim)
    Condition: (B, cond_dim) one-hot genre

    Encoder: q(z | audio, lyrics, cond)
    Decoder: p(audio, lyrics | z, cond)

    Shared latent z reconstructs both modalities conditioned on genre.
    """

    def __init__(
        self,
        n_mels: int,
        n_frames: int,
        lyrics_dim: int,
        cond_dim: int,
        latent_dim: int,
        hidden_channels: int,
        lyrics_hidden_dim: int,
    ):
        super().__init__()

        if (int(n_mels) % 8) != 0 or (int(n_frames) % 8) != 0:
            raise ValueError("n_mels and n_frames must be divisible by 8")

        self.n_mels = int(n_mels)
        self.n_frames = int(n_frames)
        self.lyrics_dim = int(lyrics_dim)
        self.cond_dim = int(cond_dim)
        self.latent_dim = int(latent_dim)

        c1 = int(hidden_channels)
        c2 = int(hidden_channels) * 2
        c3 = int(hidden_channels) * 4

        self.audio_encoder = nn.Sequential(
            nn.Conv2d(1, c1, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(c1, c2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(c2, c3, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 1, self.n_mels, self.n_frames)
            enc = self.audio_encoder(dummy)
            self._enc_c = int(enc.shape[1])
            self._enc_h = int(enc.shape[2])
            self._enc_w = int(enc.shape[3])
            self._audio_flat = int(enc.numel())

        self.lyrics_encoder = nn.Sequential(
            nn.Linear(self.lyrics_dim, int(lyrics_hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(lyrics_hidden_dim), int(lyrics_hidden_dim)),
            nn.ReLU(),
        )

        joint_in = int(self._audio_flat + lyrics_hidden_dim + self.cond_dim)
        joint_hidden = max(256, int(lyrics_hidden_dim))
        self.joint = nn.Sequential(
            nn.Linear(joint_in, joint_hidden),
            nn.ReLU(),
        )
        self.mu = nn.Linear(joint_hidden, self.latent_dim)
        self.logvar = nn.Linear(joint_hidden, self.latent_dim)

        # Decoder uses (z, cond)
        zc_dim = int(self.latent_dim + self.cond_dim)

        self.audio_decoder_fc = nn.Linear(zc_dim, self._audio_flat)
        self.audio_decoder = nn.Sequential(
            nn.ConvTranspose2d(c3, c2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(c2, c1, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(c1, 1, kernel_size=4, stride=2, padding=1),
        )

        self.lyrics_decoder = nn.Sequential(
            nn.Linear(zc_dim, int(lyrics_hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(lyrics_hidden_dim), self.lyrics_dim),
        )

    def encode(self, x_audio: torch.Tensor, x_lyrics: torch.Tensor, c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        a = self.audio_encoder(x_audio).reshape(x_audio.shape[0], -1)
        l = self.lyrics_encoder(x_lyrics)
        h = self.joint(torch.cat([a, l, c], dim=1))
        return self.mu(h), self.logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor, c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        zc = torch.cat([z, c], dim=1)
        a = self.audio_decoder_fc(zc).reshape(z.shape[0], self._enc_c, self._enc_h, self._enc_w)
        recon_audio = self.audio_decoder(a)
        recon_lyrics = self.lyrics_decoder(zc)
        return recon_audio, recon_lyrics

    def forward(
        self, x_audio: torch.Tensor, x_lyrics: torch.Tensor, c: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x_audio, x_lyrics, c)
        z = self.reparameterize(mu, logvar)
        recon_audio, recon_lyrics = self.decode(z, c)
        return recon_audio, recon_lyrics, mu, logvar


def multimodal_spec_cvae_loss(
    x_audio: torch.Tensor,
    x_lyrics: torch.Tensor,
    recon_audio: torch.Tensor,
    recon_lyrics: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    *,
    beta: float,
    audio_recon_weight: float,
    lyrics_recon_weight: float,
) -> torch.Tensor:
    audio_loss = nn.functional.mse_loss(recon_audio, x_audio, reduction="mean")
    lyrics_loss = nn.functional.mse_loss(recon_lyrics, x_lyrics, reduction="mean")
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return audio_recon_weight * audio_loss + lyrics_recon_weight * lyrics_loss + float(beta) * kld


def train_multimodal_spec_cvae(
    X_audio: np.ndarray,
    X_lyrics: np.ndarray,
    C: np.ndarray,
    cfg: MultiModalSpecCVAEConfig,
    device: str | None = None,
    verbose: bool = False,
) -> MultiModalSpecCVAE:
    if X_audio.ndim != 4:
        raise ValueError("X_audio must be (N, 1, n_mels, n_frames)")
    if X_lyrics.ndim != 2:
        raise ValueError("X_lyrics must be (N, D)")
    if C.ndim != 2:
        raise ValueError("C must be (N, cond_dim)")
    if X_audio.shape[0] != X_lyrics.shape[0] or X_audio.shape[0] != C.shape[0]:
        raise ValueError("X_audio, X_lyrics, and C must have same N")

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = MultiModalSpecCVAE(
        n_mels=int(cfg.n_mels),
        n_frames=int(cfg.n_frames),
        lyrics_dim=int(cfg.lyrics_dim),
        cond_dim=int(cfg.cond_dim),
        latent_dim=int(cfg.latent_dim),
        hidden_channels=int(cfg.hidden_channels),
        lyrics_hidden_dim=int(cfg.lyrics_hidden_dim),
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=float(cfg.lr))

    a = torch.tensor(X_audio, dtype=torch.float32)
    l = torch.tensor(X_lyrics, dtype=torch.float32)
    c = torch.tensor(C, dtype=torch.float32)

    loader = DataLoader(TensorDataset(a, l, c), batch_size=int(cfg.batch_size), shuffle=True)

    model.train()
    for epoch in range(int(cfg.epochs)):
        epoch_loss = 0.0
        n_batches = 0
        for xb_a, xb_l, xb_c in loader:
            xb_a = xb_a.to(device)
            xb_l = xb_l.to(device)
            xb_c = xb_c.to(device)

            recon_a, recon_l, mu, logvar = model(xb_a, xb_l, xb_c)
            loss = multimodal_spec_cvae_loss(
                xb_a,
                xb_l,
                recon_a,
                recon_l,
                mu,
                logvar,
                beta=float(cfg.beta),
                audio_recon_weight=float(cfg.audio_recon_weight),
                lyrics_recon_weight=float(cfg.lyrics_recon_weight),
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            epoch_loss += float(loss.detach().cpu().item())
            n_batches += 1

        if verbose and (epoch == 0 or (epoch + 1) % 10 == 0 or (epoch + 1) == int(cfg.epochs)):
            denom = max(1, n_batches)
            print(f"[mmSpecCVAE] epoch {epoch + 1}/{int(cfg.epochs)} loss={epoch_loss / denom:.6f}")

    return model


def encode_multimodal_spec_cvae_latents(
    model: MultiModalSpecCVAE,
    X_audio: np.ndarray,
    X_lyrics: np.ndarray,
    C: np.ndarray,
    *,
    batch_size: int = 256,
    device: str | None = None,
) -> np.ndarray:
    if X_audio.ndim != 4:
        raise ValueError("X_audio must be (N, 1, n_mels, n_frames)")
    if X_lyrics.ndim != 2:
        raise ValueError("X_lyrics must be (N, D)")
    if C.ndim != 2:
        raise ValueError("C must be (N, cond_dim)")
    if X_audio.shape[0] != X_lyrics.shape[0] or X_audio.shape[0] != C.shape[0]:
        raise ValueError("X_audio, X_lyrics, and C must have same N")

    if int(batch_size) <= 0:
        raise ValueError("batch_size must be positive")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    def _encode_on(device_name: str) -> np.ndarray:
        model.eval()
        model.to(device_name)

        a = torch.tensor(X_audio, dtype=torch.float32)
        l = torch.tensor(X_lyrics, dtype=torch.float32)
        c = torch.tensor(C, dtype=torch.float32)

        pin = device_name == "cuda"
        loader = DataLoader(
            TensorDataset(a, l, c),
            batch_size=int(batch_size),
            shuffle=False,
            pin_memory=pin,
        )

        out: list[np.ndarray] = []
        with torch.no_grad():
            for xb_a, xb_l, xb_c in loader:
                xb_a = xb_a.to(device_name, non_blocking=True)
                xb_l = xb_l.to(device_name, non_blocking=True)
                xb_c = xb_c.to(device_name, non_blocking=True)
                mu, _logvar = model.encode(xb_a, xb_l, xb_c)
                out.append(mu.detach().cpu().numpy())
        return np.concatenate(out, axis=0)

    try:
        return _encode_on(device)
    except torch.OutOfMemoryError:
        if device != "cuda":
            raise
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return _encode_on("cpu")


def reconstruct_examples(
    model: MultiModalSpecCVAE,
    X_audio: np.ndarray,
    X_lyrics: np.ndarray,
    C: np.ndarray,
    indices: np.ndarray,
    *,
    device: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return reconstructions for selected indices.

    Returns:
      - recon_audio: (B, 1, n_mels, n_frames)
      - recon_lyrics: (B, lyrics_dim)
    """

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    idx = np.asarray(indices).astype(int)
    a = torch.tensor(X_audio[idx], dtype=torch.float32)
    l = torch.tensor(X_lyrics[idx], dtype=torch.float32)
    c = torch.tensor(C[idx], dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        model.to(device)
        a = a.to(device)
        l = l.to(device)
        c = c.to(device)
        mu, logvar = model.encode(a, l, c)
        z = model.reparameterize(mu, logvar)
        ra, rl = model.decode(z, c)
        return ra.detach().cpu().numpy(), rl.detach().cpu().numpy()

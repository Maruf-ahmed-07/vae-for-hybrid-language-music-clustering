from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class MFCCConfig:
    sr: int = 22050
    duration_sec: float = 30.0
    n_mfcc: int = 40
    hop_length: int = 512
    n_fft: int = 2048


def extract_mfcc_sequence(audio_path: str | Path, cfg: MFCCConfig, n_frames: int) -> np.ndarray:
    """Extract a fixed-length MFCC sequence.

    Returns shape (n_mfcc, n_frames). If the extracted MFCC has fewer than n_frames,
    it is padded with zeros; if longer, it is truncated.
    """

    import librosa  # local import to keep module import light

    audio_path = Path(audio_path)
    y, _sr = librosa.load(
        audio_path,
        sr=cfg.sr,
        mono=True,
        duration=cfg.duration_sec,
    )

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=cfg.sr,
        n_mfcc=cfg.n_mfcc,
        hop_length=cfg.hop_length,
        n_fft=cfg.n_fft,
    ).astype(np.float32)  # (n_mfcc, T)

    T = mfcc.shape[1]
    if T >= n_frames:
        return mfcc[:, :n_frames]

    out = np.zeros((cfg.n_mfcc, n_frames), dtype=np.float32)
    out[:, :T] = mfcc
    return out


def extract_logmel_sequence(
    audio_path: str | Path,
    cfg: MFCCConfig,
    n_frames: int,
    n_mels: int = 64,
) -> np.ndarray:
    """Extract a fixed-length log-mel spectrogram sequence.

    Returns shape (n_mels, n_frames). If fewer than n_frames, pads with zeros; if longer, truncates.
    """

    import librosa  # local import

    audio_path = Path(audio_path)
    y, _sr = librosa.load(
        audio_path,
        sr=cfg.sr,
        mono=True,
        duration=cfg.duration_sec,
    )

    S = librosa.feature.melspectrogram(
        y=y,
        sr=cfg.sr,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        n_mels=int(n_mels),
        power=2.0,
    )  # (n_mels, T)
    S_db = librosa.power_to_db(S, ref=np.max).astype(np.float32)

    T = S_db.shape[1]
    if T >= n_frames:
        return S_db[:, :n_frames]

    out = np.zeros((int(n_mels), n_frames), dtype=np.float32)
    out[:, :T] = S_db
    return out


def extract_mfcc_stats(audio_path: str | Path, cfg: MFCCConfig) -> np.ndarray:
    """Extract a fixed-size MFCC feature vector from an audio file.

    Returns: shape (2 * n_mfcc,) = concat(mean(MFCC), std(MFCC)) over time.

    Note: mp3 decoding may require system codecs/ffmpeg depending on your setup.
    """

    import librosa  # local import to keep module import light

    audio_path = Path(audio_path)
    y, _sr = librosa.load(
        audio_path,
        sr=cfg.sr,
        mono=True,
        duration=cfg.duration_sec,
    )

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=cfg.sr,
        n_mfcc=cfg.n_mfcc,
        hop_length=cfg.hop_length,
        n_fft=cfg.n_fft,
    )  # (n_mfcc, T)

    mean = mfcc.mean(axis=1)
    std = mfcc.std(axis=1)
    feat = np.concatenate([mean, std], axis=0).astype(np.float32)
    return feat

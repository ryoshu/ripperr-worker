"""Settings from the environment, and the model identity reported with results.

Variable names match ripperr's where the meaning is the same, so one `.env`
can describe both.
"""

from __future__ import annotations

import os
import platform
import socket
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

# Speaker vectors from Senko's CAM++ model. Ripperr only matches profiles
# within one space, so a different embedding model must use a different name.
SENKO_EMBEDDING_SPACE = "senko-campplus"


def is_apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine().lower() in {"arm64", "aarch64"}


def resolve_asr_backend(backend: str = "auto") -> str:
    if backend == "auto":
        return "mlx" if is_apple_silicon() else "faster-whisper"
    if backend not in {"mlx", "faster-whisper"}:
        raise ValueError("RIPPERR_ASR_BACKEND must be auto, mlx, or faster-whisper")
    return backend


def resolve_diarization_backend(backend: str = "auto") -> str:
    if backend == "auto":
        return "senko" if is_apple_silicon() else "pyannote"
    if backend not in {"senko", "pyannote"}:
        raise ValueError("RIPPERR_DIARIZATION_BACKEND must be auto, senko, or pyannote")
    return backend


def runtime_version(package: str) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def _token() -> str | None:
    token = os.environ.get("RIPPERR_API_TOKEN")
    if token:
        return token.strip()
    path = Path(os.environ.get("RIPPERR_TOKEN_FILE", "~/.config/ripperr-worker/token")).expanduser()
    return path.read_text().strip() if path.exists() else None


@dataclass
class Settings:
    server: str = field(default_factory=lambda: os.environ.get("RIPPERR_SERVER", "").rstrip("/"))
    token: str | None = field(default_factory=_token)
    worker: str = field(default_factory=lambda: os.environ.get("RIPPERR_WORKER_NAME") or socket.gethostname())
    poll_seconds: float = field(default_factory=lambda: float(os.environ.get("RIPPERR_POLL_SECONDS", "60")))

    asr_backend: str = field(default_factory=lambda: os.environ.get("RIPPERR_ASR_BACKEND", "auto"))
    diarization_backend: str = field(
        default_factory=lambda: os.environ.get("RIPPERR_DIARIZATION_BACKEND", "auto")
    )
    device: str = field(default_factory=lambda: os.environ.get("RIPPERR_DEVICE", "auto"))
    asr_model: str | None = field(default_factory=lambda: os.environ.get("RIPPERR_ASR_MODEL"))
    diarization_model: str = field(default_factory=lambda: os.environ.get(
        "RIPPERR_DIARIZATION_MODEL", "pyannote/speaker-diarization-community-1"
    ))
    diarization_token: str | None = field(default_factory=lambda: (
        os.environ.get("RIPPERR_HF_TOKEN") or os.environ.get("HF_TOKEN")
    ))
    language: str | None = field(default_factory=lambda: os.environ.get("RIPPERR_LANGUAGE") or None)

    def __post_init__(self) -> None:
        self.asr_backend = resolve_asr_backend(self.asr_backend)
        self.diarization_backend = resolve_diarization_backend(self.diarization_backend)
        if self.asr_model is None:
            self.asr_model = (
                "mlx-community/whisper-large-v3-turbo" if self.asr_backend == "mlx" else "large-v3"
            )

    # The two configs below use the same fields ripperr's own cache keys did
    # before the split (including `model` for Senko, which ignores it), so a
    # worker result for an already-processed episode lands on the same cache key.
    def asr_config(self) -> dict[str, object]:
        return {
            "backend": self.asr_backend,
            "runtime": runtime_version("mlx-whisper" if self.asr_backend == "mlx" else "faster-whisper"),
            "model": self.asr_model,
            "device": self.device,
            "language": self.language,
        }

    def diarization_config(self) -> dict[str, object]:
        return {
            "backend": self.diarization_backend,
            "runtime": runtime_version("senko" if self.diarization_backend == "senko" else "pyannote-audio"),
            "model": self.diarization_model,
            "device": self.device,
        }

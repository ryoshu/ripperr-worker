"""Transcription adapters with one normalized Whisper-shaped output.

We only care about one thing here beyond the text: word-level timestamps. Those
are what make the speaker merge accurate — segment-level timing is too coarse,
since a Whisper segment routinely spans a speaker change.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from .config import resolve_asr_backend


def backend_name(backend: str = "auto") -> str:
    return resolve_asr_backend(backend)


def transcribe(
    wav_path: Path,
    model: str,
    language: str | None = None,
    backend: str = "auto",
    device: str = "auto",
) -> dict[str, Any]:
    backend = backend_name(backend)
    if backend == "faster-whisper":
        return _faster_whisper(wav_path, model, language, device)
    return _mlx(wav_path, model, language)


def _mlx(wav_path: Path, model: str, language: str | None) -> dict[str, Any]:
    import mlx_whisper  # imported lazily so the CLI works without MLX installed

    kwargs: dict[str, Any] = {
        "path_or_hf_repo": model,
        "word_timestamps": True,
        "condition_on_previous_text": False,  # limits runaway hallucination loops
    }
    if language:
        kwargs["language"] = language

    return mlx_whisper.transcribe(str(wav_path), **kwargs)


def _faster_whisper(
    wav_path: Path, model: str, language: str | None, device: str
) -> dict[str, Any]:
    if device not in {"auto", "cpu", "cuda"}:
        raise ValueError("RIPPERR_DEVICE must be auto, cpu, or cuda")
    compute_type = "float16" if device == "cuda" else "int8" if device == "cpu" else "default"
    whisper = _faster_model(model, device, compute_type)
    kwargs: dict[str, Any] = {
        "word_timestamps": True,
        "condition_on_previous_text": False,
    }
    if language:
        kwargs["language"] = language
    segments, _ = whisper.transcribe(str(wav_path), **kwargs)
    return {"segments": [_faster_segment(segment) for segment in segments]}


@lru_cache(maxsize=4)
def _faster_model(model: str, device: str, compute_type: str) -> Any:
    from faster_whisper import WhisperModel

    return WhisperModel(model, device=device, compute_type=compute_type)


def _faster_segment(segment: Any) -> dict[str, Any]:
    words = []
    for word in getattr(segment, "words", None) or []:
        start, end, text = word.start, word.end, word.word
        if start is not None and end is not None and text:
            words.append({"start": float(start), "end": float(end), "word": text})
    return {
        "start": float(segment.start),
        "end": float(segment.end),
        "text": segment.text,
        "words": words,
    }

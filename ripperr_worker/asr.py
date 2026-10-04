"""Transcription adapters with one normalized Whisper-shaped output.

We only care about one thing here beyond the text: word-level timestamps. Those
are what make the speaker merge accurate — segment-level timing is too coarse,
since a Whisper segment routinely spans a speaker change.
"""

from __future__ import annotations

import os
import tempfile
import wave
from functools import lru_cache
from pathlib import Path
from typing import Any

from .config import WHISTLE_MODEL, resolve_asr_backend


def backend_name(backend: str = "auto") -> str:
    return resolve_asr_backend(backend)


def transcribe(
    wav_path: Path,
    model: str,
    language: str | None = None,
    backend: str = "auto",
    device: str = "auto",
    keywords: list[str] | None = None,
) -> dict[str, Any]:
    backend = backend_name(backend)
    if backend == "faster-whisper":
        return _faster_whisper(wav_path, model, language, device)
    if backend == "whistle":
        return _whistle(wav_path, model, language, device, keywords or [])
    return _mlx(wav_path, model, language)


def _whistle(
    wav_path: Path, model: str, language: str | None, device: str, keywords: list[str]
) -> dict[str, Any]:
    """Run Whistle in 30-second windows and stitch word times to file time."""
    if device not in {"auto", "cpu"}:
        raise ValueError("Whistle currently supports only RIPPERR_DEVICE=auto or cpu")

    # The evaluation and worker should not emit Cactus usage telemetry unless
    # a future caller explicitly sets the variable before starting the worker.
    os.environ.setdefault("NEEDLE_TELEMETRY", "0")
    import needle  # imported lazily so non-Whistle installs remain lightweight

    weights = None if model == WHISTLE_MODEL else model
    rate = 16000
    chunk_frames = 30 * rate
    stride_frames = 28 * rate  # 2 seconds of overlap between adjacent chunks
    stitched: list[dict[str, Any]] = []

    with wave.open(str(wav_path), "rb") as source:
        params = source.getparams()
        if (params.nchannels, params.sampwidth, params.framerate) != (1, 2, rate):
            raise ValueError("Whistle requires 16 kHz, mono, 16-bit PCM WAV input")
        total_frames = params.nframes
        if total_frames <= 0:
            return {"segments": []}

        with tempfile.TemporaryDirectory(prefix="ripperr-whistle-") as tmp:
            chunk_path = Path(tmp) / "chunk.wav"
            frame_start = 0
            index = 0
            while frame_start < total_frames:
                frame_count = min(chunk_frames, total_frames - frame_start)
                source.setpos(frame_start)
                frames = source.readframes(frame_count)
                with wave.open(str(chunk_path), "wb") as chunk:
                    chunk.setparams(params)
                    chunk.writeframes(frames)

                result = needle.transcribe(
                    str(chunk_path), language=language, keywords=keywords or None,
                    word_timestamps=True, weights=weights,
                )
                offset = frame_start / rate
                duration = frame_count / rate
                # The midpoint boundary gives each overlapping region one owner.
                keep_from = 0.0 if index == 0 else 1.0
                keep_until = duration if frame_start + frame_count == total_frames else duration - 1.0
                chunk_words = []
                for word in result.get("words") or []:
                    start, end, text = word.get("start"), word.get("end"), word.get("word")
                    if start is None or end is None or not text:
                        continue
                    start, end = float(start), float(end)
                    midpoint = (start + end) / 2
                    if keep_from <= midpoint < keep_until:
                        chunk_words.append({"start": start + offset, "end": end + offset, "word": str(text)})

                if not chunk_words:
                    if (result.get("text") or "").strip():
                        raise RuntimeError("Whistle returned transcript text without timestamped words")
                else:
                    stitched.extend(chunk_words)

                # A final window that starts every 28 seconds would be
                # redundant when the preceding 30-second window already
                # reaches the end of a shorter file.
                if frame_start + chunk_frames >= total_frames:
                    break
                frame_start += stride_frames
                index += 1

    stitched.sort(key=lambda word: (word["start"], word["end"]))
    if not stitched:
        return {"segments": []}
    return {
        "segments": [{
            "start": stitched[0]["start"],
            "end": stitched[-1]["end"],
            "text": " ".join(word["word"] for word in stitched).strip(),
            "words": stitched,
        }]
    }


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

from io import BytesIO
from types import SimpleNamespace
from urllib.error import HTTPError

import pytest

from ripperr_worker import asr, config, diarize
from ripperr_worker.config import Settings
from ripperr_worker.worker import build_result, work_one


def _settings(**kw) -> Settings:
    return Settings(server="http://ripperr", token="t", worker="test",
                    asr_backend="mlx", diarization_backend="senko", **kw)


def test_auto_selects_linux_backends(monkeypatch):
    monkeypatch.setattr(config.platform, "system", lambda: "Linux")
    monkeypatch.setattr(config.platform, "machine", lambda: "x86_64")
    assert asr.backend_name() == "faster-whisper"
    assert diarize.backend_name() == "pyannote"


def test_unknown_backend_is_explicit():
    with pytest.raises(ValueError, match="RIPPERR_ASR_BACKEND"):
        asr.backend_name("nope")
    with pytest.raises(ValueError, match="RIPPERR_DIARIZATION_BACKEND"):
        diarize.backend_name("nope")


def test_whistle_config_and_keywords(monkeypatch):
    monkeypatch.setenv("RIPPERR_ASR_KEYWORDS", "Ben Johnson, Kalif Raymond\nDetroit Lions")
    settings = Settings(asr_backend="whistle", diarization_backend="pyannote")
    assert settings.asr_model == config.WHISTLE_MODEL
    assert settings.asr_keywords == ["Ben Johnson", "Kalif Raymond", "Detroit Lions"]
    assert settings.asr_config() == {
        "backend": "whistle", "runtime": config.runtime_version("cactus-needle"),
        "model": config.WHISTLE_MODEL, "device": "auto", "language": None,
        "keywords": "Ben Johnson, Kalif Raymond, Detroit Lions",
    }
    configs = (settings.asr_config(), settings.diarization_config())
    assert all(v is None or isinstance(v, (str, int, float)) for c in configs for v in c.values())


def test_whistle_chunks_and_stitches_overlaps(tmp_path, monkeypatch):
    import sys
    import wave

    calls = []

    def fake_transcribe(path, **kwargs):
        with wave.open(path, "rb") as audio:
            duration = audio.getnframes() / audio.getframerate()
        calls.append((duration, kwargs))
        words = [{"start": 0.5, "end": 0.8, "word": "overlap"}]
        if duration > 2:
            words.append({"start": 1.5, "end": 1.8, "word": "owned"})
        if duration >= 30:
            words.append({"start": 29.5, "end": 29.8, "word": "tail"})
        return {"text": "fake", "words": words}

    monkeypatch.setitem(sys.modules, "needle", SimpleNamespace(transcribe=fake_transcribe))
    wav_path = tmp_path / "hour.wav"
    with wave.open(str(wav_path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(16000)
        out.writeframes(b"\0\0" * (60 * 16000))

    result = asr._whistle(wav_path, config.WHISTLE_MODEL, "en", "cpu", ["Detroit Lions"])
    assert [duration for duration, _ in calls] == [30, 30, 4]
    assert [word["word"] for word in result["segments"][0]["words"]] == [
        "overlap", "owned", "owned", "owned",
    ]
    assert [word["start"] for word in result["segments"][0]["words"]] == [0.5, 1.5, 29.5, 57.5]
    assert all(kwargs["word_timestamps"] for _, kwargs in calls)
    assert all(kwargs["keywords"] == ["Detroit Lions"] for _, kwargs in calls)


def test_whistle_requires_timestamped_words(tmp_path, monkeypatch):
    import sys
    import wave

    monkeypatch.setitem(sys.modules, "needle", SimpleNamespace(
        transcribe=lambda *args, **kwargs: {"text": "lost timing", "words": []}
    ))
    wav_path = tmp_path / "short.wav"
    with wave.open(str(wav_path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(16000)
        out.writeframes(b"\0\0" * 16000)
    with pytest.raises(RuntimeError, match="without timestamped words"):
        asr._whistle(wav_path, config.WHISTLE_MODEL, None, "cpu", [])


def test_whistle_does_not_create_a_redundant_short_tail_chunk(tmp_path, monkeypatch):
    import sys
    import wave

    calls = []

    def fake_transcribe(path, **kwargs):
        calls.append(path)
        return {"text": "last second", "words": [{"start": 28.2, "end": 28.7, "word": "last"}]}

    monkeypatch.setitem(sys.modules, "needle", SimpleNamespace(transcribe=fake_transcribe))
    wav_path = tmp_path / "short-tail.wav"
    with wave.open(str(wav_path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(16000)
        out.writeframes(b"\0\0" * (29 * 16000))

    result = asr._whistle(wav_path, config.WHISTLE_MODEL, None, "cpu", [])
    assert len(calls) == 1
    assert result["segments"][0]["words"] == [{"start": 28.2, "end": 28.7, "word": "last"}]


def test_faster_whisper_segments_match_contract():
    segment = SimpleNamespace(start=1, end=3, text=" hello", words=[SimpleNamespace(start=1, end=2, word=" hello")])
    assert asr._faster_segment(segment) == {
        "start": 1.0, "end": 3.0, "text": " hello", "words": [{"start": 1.0, "end": 2.0, "word": " hello"}],
    }


def test_result_body_is_schema_1_and_drops_whisper_extras():
    mlx_output = {"text": "hi", "language": "en", "segments": [{
        "id": 0, "start": 0.0, "end": 1.0, "text": " hi", "tokens": [1, 2], "avg_logprob": -0.1,
        "words": [{"start": 0.0, "end": 0.4, "word": " hi", "probability": 0.9}],
    }]}
    segments = [{"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"}]
    body = build_result(_settings(), "L", mlx_output, segments, {"SPEAKER_00": [0.5]})

    assert body["schema"] == 1 and body["lease_id"] == "L"
    assert body["asr"]["segments"] == [
        {"start": 0.0, "end": 1.0, "text": " hi", "words": [{"start": 0.0, "end": 0.4, "word": " hi"}]},
    ]
    # Same fields ripperr keyed its caches by before the split; changing them
    # re-keys every cached result.
    assert set(body["asr"]["config"]) == {"backend", "runtime", "model", "device", "language"}
    assert set(body["diarization"]["config"]) == {"backend", "runtime", "model", "device"}
    assert body["diarization"]["embeddings"] == {"space": "senko-campplus", "speakers": {"SPEAKER_00": [0.5]}}
    assert "embeddings" not in build_result(_settings(), "L", mlx_output, segments, None)["diarization"]


class FakeClient:
    def __init__(self, result_error: int | None = None):
        self.claims = [{"guid": "g1", "lease_id": "L", "duration_s": 3.0}]
        self.result_error, self.results, self.failures = result_error, [], []

    def claim(self, worker):
        return self.claims.pop() if self.claims else None

    def audio(self, guid, lease_id, dest):
        dest.write_bytes(b"RIFF")

    def result(self, guid, body):
        if self.result_error:
            raise HTTPError("u", self.result_error, "x", {}, BytesIO(b"{}"))
        self.results.append((guid, body))
        return {"guid": guid, "revision": 1}

    def fail(self, guid, lease_id, error):
        self.failures.append((guid, lease_id, error))


class FakeModels:
    def __init__(self, boom: bool = False):
        self.boom = boom

    def transcribe(self, wav):
        if self.boom:
            raise RuntimeError("metal out of memory\ntraceback...")
        return {"segments": [{"start": 0.0, "end": 1.0, "text": "hi", "words": []}]}

    def diarize(self, wav):
        return [{"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"}], None


def test_work_one_uploads_reports_failures_and_drops_lost_leases():
    quiet = lambda _: None  # noqa: E731
    client = FakeClient()
    assert work_one(client, FakeModels(), _settings(), quiet) is True
    assert [guid for guid, _ in client.results] == ["g1"]
    assert work_one(client, FakeModels(), _settings(), quiet) is False  # nothing left

    client = FakeClient()
    assert work_one(client, FakeModels(boom=True), _settings(), quiet) is True
    assert client.failures == [("g1", "L", "RuntimeError: metal out of memory")]

    client = FakeClient(result_error=409)
    assert work_one(client, FakeModels(), _settings(), quiet) is True  # dropped, loop goes on
    with pytest.raises(HTTPError):
        work_one(FakeClient(result_error=500), FakeModels(), _settings(), quiet)


def test_senko_is_pinned_to_a_commit():
    import re
    import tomllib
    from pathlib import Path

    pyproject = Path(__file__).parent.parent / "pyproject.toml"
    deps = tomllib.loads(pyproject.read_text())["project"]["optional-dependencies"]["senko"]
    senko = next(d for d in deps if d.startswith("senko"))
    assert re.search(r"git\+https://\S+@[0-9a-f]{40}$", senko), senko

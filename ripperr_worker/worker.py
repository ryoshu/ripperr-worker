"""Claim an episode from ripperr, run ASR and diarization, upload the result.

The protocol is ripperr's docs/worker-contract.md (schema 1). This process
keeps no state: if it dies or the machine sleeps, the lease expires and
ripperr hands the episode to the next claim.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from . import asr, diarize
from .config import SENKO_EMBEDDING_SPACE, Settings

SCHEMA = 1
Log = Callable[[str], None]


class Client:
    """Ripperr's /v1/work endpoints over plain HTTP(S)."""

    def __init__(self, server: str, token: str):
        self.server, self.token = server, token

    def _open(self, method: str, path: str, body: dict | None = None, timeout: float = 60):
        data = json.dumps(body).encode() if body is not None else None
        headers = {"Authorization": f"Bearer {self.token}"}
        if data is not None:
            headers["Content-Type"] = "application/json"
        return urlopen(Request(self.server + path, data=data, headers=headers, method=method), timeout=timeout)

    def claim(self, worker: str) -> dict[str, Any] | None:
        with self._open("POST", "/v1/work/claim", {"worker": worker, "schema": SCHEMA}) as response:
            return None if response.status == 204 else json.load(response)

    def audio(self, guid: str, lease_id: str, dest: Path) -> None:
        path = f"/v1/work/{quote(guid, safe='')}/audio?lease_id={quote(lease_id, safe='')}"
        with self._open("GET", path, timeout=600) as response, dest.open("wb") as out:
            shutil.copyfileobj(response, out)

    def result(self, guid: str, body: dict[str, Any]) -> dict[str, Any]:
        with self._open("PUT", f"/v1/work/{quote(guid, safe='')}/result", body, timeout=600) as response:
            return json.load(response)

    def fail(self, guid: str, lease_id: str, error: str) -> None:
        with self._open("POST", f"/v1/work/{quote(guid, safe='')}/fail", {"lease_id": lease_id, "error": error}):
            pass


class Models:
    """Model backends, loaded once and reused across episodes."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._diarizer: diarize.Diarizer | None = None

    @property
    def diarizer(self) -> diarize.Diarizer:
        if self._diarizer is None:
            s = self.settings
            self._diarizer = diarize.Diarizer(
                device=s.device, backend=s.diarization_backend, model=s.diarization_model,
                token=s.diarization_token, warmup=True, quiet=True,
            )
        return self._diarizer

    def transcribe(self, wav: Path) -> dict[str, Any]:
        s = self.settings
        return asr.transcribe(wav, s.asr_model, s.language, s.asr_backend, s.device)

    def diarize(self, wav: Path) -> tuple[list[dict[str, Any]], dict[str, list[float]] | None]:
        if self.diarizer.backend == "senko":
            return self.diarizer.run_with_embeddings(wav)
        return self.diarizer.run(wav), None


def build_result(
    settings: Settings,
    lease_id: str,
    asr_result: dict[str, Any],
    segments: list[dict[str, Any]],
    embeddings: dict[str, list[float]] | None,
) -> dict[str, Any]:
    """The schema-1 result body. Whisper's extra per-segment fields (tokens,
    log-probs, ...) are dropped; ripperr only reads times and text."""
    diarization: dict[str, Any] = {"config": settings.diarization_config(), "segments": segments}
    if embeddings:
        diarization["embeddings"] = {"space": SENKO_EMBEDDING_SPACE, "speakers": embeddings}
    return {
        "schema": SCHEMA,
        "lease_id": lease_id,
        "asr": {
            "config": settings.asr_config(),
            "segments": [
                {
                    "start": seg.get("start"),
                    "end": seg.get("end"),
                    "text": seg.get("text") or "",
                    "words": [
                        {"start": w.get("start"), "end": w.get("end"), "word": w.get("word") or w.get("text") or ""}
                        for w in seg.get("words") or []
                    ],
                }
                for seg in asr_result.get("segments", [])
            ],
        },
        "diarization": diarization,
    }


def work_one(client: Client, models: Models, settings: Settings, log: Log = print) -> bool:
    """Claim and finish one episode. Returns False when nothing is claimable."""
    claim = client.claim(settings.worker)
    if claim is None:
        return False
    guid, lease_id = claim["guid"], claim["lease_id"]
    log(f"[{guid}] claimed ({claim.get('duration_s') or '?'} s of audio)")
    with tempfile.TemporaryDirectory(prefix="ripperr-worker-") as tmp:
        wav = Path(tmp) / "audio.16k.wav"
        try:
            client.audio(guid, lease_id, wav)
            started = time.monotonic()
            asr_result = models.transcribe(wav)
            segments, embeddings = models.diarize(wav)
            log(f"  models done in {time.monotonic() - started:.0f} s")
        except HTTPError:
            raise  # a lease or server problem, not this episode's fault
        except Exception as exc:  # noqa: BLE001 - report it; ripperr marks the episode failed
            log(f"  ! failed: {exc}")
            client.fail(guid, lease_id, f"{type(exc).__name__}: {exc}".splitlines()[0][:500])
            return True
        try:
            done = client.result(guid, build_result(settings, lease_id, asr_result, segments, embeddings))
        except HTTPError as exc:
            if exc.code != 409:
                raise
            log("  lease lost (expired and re-claimed, or already applied); dropping result")
            return True
    log(f"  merged at revision {done['revision']}")
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ripperr-worker", description=__doc__.splitlines()[0])
    parser.add_argument("--once", action="store_true", help="exit when nothing is claimable")
    args = parser.parse_args(argv)

    settings = Settings()
    if not settings.server or not settings.token:
        print("set RIPPERR_SERVER and RIPPERR_API_TOKEN (or RIPPERR_TOKEN_FILE)", file=sys.stderr)
        return 2
    client, models = Client(settings.server, settings.token), Models(settings)
    log = lambda line: print(line, flush=True)  # noqa: E731
    backoff = settings.poll_seconds
    while True:
        try:
            if work_one(client, models, settings, log):
                backoff = settings.poll_seconds
                continue
            if args.once:
                return 0
        except HTTPError as exc:
            if exc.code in (400, 401, 503):  # misconfiguration; retrying will not help
                print(f"ripperr refused the request ({exc.code}): {exc.read().decode(errors='replace')}",
                      file=sys.stderr)
                return 1
            log(f"server error {exc.code}; retrying in {backoff:.0f} s")
        except (URLError, OSError) as exc:
            log(f"ripperr unreachable ({exc}); retrying in {backoff:.0f} s")
            backoff = min(backoff * 2, 900)
        if args.once:
            return 1
        time.sleep(backoff)


if __name__ == "__main__":
    raise SystemExit(main())

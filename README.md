# ripperr-worker

The compute half of [ripperr](https://github.com/ryoshu/ripperr). It claims a
prepared episode from a ripperr server, runs speech recognition and speaker
diarization on it, uploads the result, and repeats. It keeps no database and
no state between episodes; ripperr does everything else (feeds, downloads,
merge, glossary, speaker identity, the API).

The protocol is ripperr's
[docs/worker-contract.md](https://github.com/ryoshu/ripperr/blob/main/docs/worker-contract.md)
(schema 1). The worker only polls; ripperr never connects to it, so it can run
on a laptop that sleeps. An unfinished episode's lease expires and it goes back
in the queue.

## Setup

```bash
uv venv --python 3.13 && source .venv/bin/activate
uv pip install -e ".[apple]"        # Apple Silicon: mlx-whisper + Senko (macOS 14+)
# uv pip install -e ".[linux]"      # Linux: faster-whisper + pyannote
```

## Run

```bash
export RIPPERR_SERVER=https://ripperr-host.example.ts.net
export RIPPERR_API_TOKEN=...        # or put it in ~/.config/ripperr-worker/token (mode 600)
ripperr-worker                      # loop forever; --once exits when the queue is empty
```

| Variable | Effect |
| --- | --- |
| `RIPPERR_SERVER` | ripperr base URL (required) |
| `RIPPERR_API_TOKEN` / `RIPPERR_TOKEN_FILE` | bearer token, or a file holding it (default `~/.config/ripperr-worker/token`) |
| `RIPPERR_WORKER_NAME` | name recorded on leases (default: hostname) |
| `RIPPERR_POLL_SECONDS` | wait between empty polls (default 60; doubles up to 15 min while ripperr is unreachable) |
| `RIPPERR_ASR_BACKEND`, `RIPPERR_ASR_MODEL`, `RIPPERR_DIARIZATION_BACKEND`, `RIPPERR_DIARIZATION_MODEL`, `RIPPERR_DEVICE`, `RIPPERR_LANGUAGE`, `RIPPERR_HF_TOKEN` | model choices, as in ripperr |

Each result carries the backend, runtime version and model it came from.
Ripperr caches it under that identity, and only accepts speaker embeddings
from its configured space (`senko-campplus` for Senko), so mixing a Linux
worker with a Mac one cannot corrupt speaker profiles.

Run the tests with `uv pip install pytest && python -m pytest tests`.

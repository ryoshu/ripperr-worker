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
# uv pip install -e ".[linux,whistle]" # Linux: optional Cactus Whistle + pyannote
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
| `RIPPERR_ASR_BACKEND`, `RIPPERR_ASR_MODEL`, `RIPPERR_ASR_KEYWORDS`, `RIPPERR_DIARIZATION_BACKEND`, `RIPPERR_DIARIZATION_MODEL`, `RIPPERR_DEVICE`, `RIPPERR_LANGUAGE`, `RIPPERR_HF_TOKEN` | model choices, as in ripperr; `whistle` is an opt-in ASR backend |

Each result carries the backend, runtime version and model it came from.
Ripperr caches it under that identity, and only accepts speaker embeddings
from its configured space (`senko-campplus` for Senko), so mixing a Linux
worker with a Mac one cannot corrupt speaker profiles.

Run the tests with `uv pip install pytest && python -m pytest tests`.

## Docker on an always-on host

The CPU image targets Linux/amd64 and includes faster-whisper, pyannote, and
the optional Whistle backend. The Compose defaults select Whistle; set
`RIPPERR_ASR_BACKEND=faster-whisper` in `.env` to use the existing Linux ASR
instead. Whistle processes WAV audio in overlapping 30-second windows and
stitches word timestamps back onto the episode timeline. A custom
`RIPPERR_ASR_MODEL` for Whistle must be a local `.cact` weights path; the
default downloads Cactus's published weights on first use.

On the Mac Pro, use the Docker CLI connected to its Colima engine:

```bash
cp .env.docker.example .env
chmod 600 .env
# Edit .env with the ripperr worker token and Hugging Face token.
/opt/local/bin/docker compose build
/opt/local/bin/docker compose up -d
/opt/local/bin/docker compose logs -f worker
```

The worker polls the server over HTTPS and publishes no ports. Set
`RIPPERR_SERVER` to an address reachable from inside Colima (the example uses
the Mac Pro's Tailscale Serve URL). The named `model-cache` volume stores
Whistle and Hugging Face model files across restarts. The Compose project is
named `ripperr-worker`, so its lifecycle is separate from other Compose
projects such as the existing PostgreSQL service. `docker compose down` keeps
the cache; `docker compose down -v` deletes it.

The deployment needs a Ripperr worker API token and a Hugging Face token that
can access `pyannote/speaker-diarization-community-1`. Keep those values only
in the local `.env` file. The Compose service is limited to three CPUs and 6 GB
of memory to leave room for other containers on a small Colima VM.

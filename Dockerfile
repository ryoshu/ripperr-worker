FROM python:3.11-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HOME=/models/home \
    HF_HOME=/models/home/.cache/huggingface

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates ffmpeg libgomp1 libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY ripperr_worker ./ripperr_worker
RUN python -m pip install --index-url https://download.pytorch.org/whl/cpu \
    "torch==2.8.0+cpu" "torchaudio==2.8.0+cpu" "torchcodec==0.7.0"
RUN python -m pip install --upgrade pip \
    && python -m pip install ".[linux,whistle]"

RUN groupadd --gid 10001 worker \
    && useradd --uid 10001 --gid worker --home-dir /models/home --no-create-home --shell /usr/sbin/nologin worker \
    && mkdir -p /models/home \
    && chown -R worker:worker /models

USER 10001:10001
ENTRYPOINT ["ripperr-worker"]

FROM runtime AS test
USER root
RUN python -m pip install pytest
COPY tests ./tests
RUN python -m pytest -q

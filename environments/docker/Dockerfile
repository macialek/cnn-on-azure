# syntax=docker/dockerfile:1.3.1

# ************************************************
# Base image
# ************************************************
FROM python:3.9-slim-buster as base

ARG DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TZ=Etc/UTC


# ************************************************
# Builder image
# ************************************************
FROM base as builder
ARG DEBIAN_FRONTEND=noninteractive
RUN apt update -y \
    && apt install --no-install-recommends -y \
        python3-dev \
        wget \
        curl \
        build-essential \
        cmake \
        libopenblas-dev \
        liblapack-dev \
        libjpeg-dev \
        libssl-dev \
        libffi-dev \
        cargo \
        ssh \
        git \
        gzip \
    && apt clean \
    && rm -rf /var/lib/apt/lists/*

# POETRY installation
ENV POETRY_HOME=/etc/poetry \
    POETRY_VERSION=1.1.13 \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_NO_INTERACTION=1
RUN curl -sSL https://install.python-poetry.org | python -

# dependencies installation via POETRY
COPY ./pyproject.toml ./poetry.lock ./
RUN $POETRY_HOME/bin/poetry install --no-dev --no-root


# ************************************************
# Final image
# ************************************************
FROM base as production
COPY --from=builder /usr/local /usr/local

# syntax=docker/dockerfile:1.3.1

# ************************************************
# Base image
# ************************************************
FROM ubuntu:20.04 as base

ARG DEBIAN_FRONTEND=noninteractive

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    LANG=C.UTF-8 \
    TZ=Etc/UTC

RUN apt-get update -y \
    && apt-get install --no-install-recommends -y \
        python3 \
        python3-pip \
        python3.8-venv \
        python-is-python3 \
        curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* 

# ************************************************
# Builder image
# ************************************************
FROM base as builder
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update -y \
    && apt-get install --no-install-recommends -y \
        python3-dev \
        wget \
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
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# POETRY installation
WORKDIR /app
ENV POETRY_HOME=/etc/poetry \
    POETRY_VERSION=1.1.13 \
    POETRY_VIRTUALENVS_CREATE=true \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_NO_INTERACTION=1
RUN curl -sSL https://install.python-poetry.org | python -

# dependencies installation via POETRY
COPY ./pyproject.toml ./poetry.lock ./
RUN $POETRY_HOME/bin/poetry install --no-dev --no-root --no-ansi


# ************************************************
# Final image
# ************************************************
FROM base as production
COPY --from=builder /app/.venv /usr/local

# syntax=docker/dockerfile:1.3.1

# ************************************************
# Base image
# ************************************************
FROM python:3.9 as base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1


# ************************************************
# Builder image
# ************************************************
FROM base as builder
RUN apt update -y \
    && apt install --no-install-recommends -y \
        curl build-essential cmake libopenblas-dev liblapack-dev libjpeg-dev ssh git zip unzip

# POETRY installation
ENV POETRY_HOME=/etc/poetry \
    POETRY_VERSION=1.1.13 \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_NO_INTERACTION=1
RUN curl -sSL https://install.python-poetry.org | python3 -

# dependencies installation via POETRY
COPY ./pyproject.toml ./poetry.lock ./
RUN $POETRY_HOME/bin/poetry install --no-dev --no-root


# ************************************************
# Final image
# ************************************************
FROM base as production
COPY --from=builder /usr/local /usr/local

# syntax=docker/dockerfile:1
# Keep this syntax directive! It's used to enable Docker BuildKit


################################
# PYTHON-BASE
# Sets up all our shared environment variables
################################
FROM python:3.11-slim as python-base

ENV \
    # python
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # pip
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    # poetry https://python-poetry.org/docs/configuration/#using-environment-variables
    POETRY_VERSION=1.8.2 \
    # make poetry install to this location
    POETRY_HOME="/opt/poetry" \
    # do not ask any interactive question
    POETRY_NO_INTERACTION=1 \
    # paths
    # this is where our requirements + virtual environment will live
    PYSETUP_PATH="/opt/pysetup" \
    VIRTUAL_ENV_PATH="/opt/pysetup/.venv"

# prepend poetry and venv to path
# install poetry - respects $POETRY_VERSION & $POETRY_HOME
# The --mount will mount the buildx cache directory to where
# Poetry and Pip store their cache so that they can re-use it
ENV PATH="$POETRY_HOME/bin:$VIRTUAL_ENV_PATH/bin:$PATH"


################################
# Builder-BASE
# Sets up all our builds tools
# `builder-base` stage is used to build deps + create our virtual environment
################################
FROM python-base as builder-base
RUN apt-get update && apt-get install --no-install-recommends -y curl build-essential python3-dev gcc g++ cmake libgomp1
# Add  `--mount=type=cache,target=/root/.cache` right after RUN if buildkit available to speedup the process
RUN curl -sSL https://install.python-poetry.org | python -
ENV PATH="$POETRY_HOME/bin:$VIRTUAL_ENV_PATH/bin:$PATH"

# copy project requirement files here to ensure they will be cached.
WORKDIR $PYSETUP_PATH
COPY ./ ./

# install runtime deps to $VIRTUAL_ENV
# Add  `--mount=type=cache,target=/root/.cache` If buildkit available to speedup the process
RUN poetry install

################################
# Production
# `production` image used for runtime
################################
FROM python-base as production
RUN apt-get update && apt-get install --no-install-recommends -y curl build-essential python3-dev gcc g++ cmake libgomp1

COPY --from=builder-base $PYSETUP_PATH $PYSETUP_PATH
WORKDIR $PYSETUP_PATH
CMD [".venv/bin/cli", "run"]

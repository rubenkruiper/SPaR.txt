FROM python:3.13-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 - \
    && ln -s /root/.local/bin/poetry /usr/local/bin/poetry

# Copy dependency manifests first so this layer is cached independently of code changes
WORKDIR /app
COPY pyproject.toml poetry.lock ./

# Install runtime dependencies only (no dev extras)
RUN poetry config virtualenvs.create false \
    && poetry install --only main --no-interaction --no-ansi

# To rebuild from here without re-running pip install, pass a build arg:
#   docker build --build-arg ONLY_CODE=$(date +%s) -t spar .
ARG ONLY_CODE=unknown
RUN echo "$ONLY_CODE"

# Copy the rest of the application
COPY . .

EXPOSE 8501
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

CMD ["uvicorn", "spar_api:SPaR_api", "--host", "0.0.0.0", "--port", "8501"]

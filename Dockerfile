# Serving image for the newline-fixer HTTP API.
#
# The trained model is baked in from artifacts/encoder, so train (or copy) it
# before building — artifacts/ is gitignored, not part of the repo:
#   uv run python scripts/train_encoder.py --data data/docs --out artifacts/encoder
#   docker build -t newlinefix . && docker run -p 8000:8000 newlinefix
#
# Linux resolves CPU-only torch wheels (see [tool.uv.sources] in pyproject.toml),
# keeping the image an order of magnitude smaller than a CUDA build.

FROM ghcr.io/astral-sh/uv:python3.14-bookworm-slim

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

# Dependency layer first: cached until pyproject.toml/uv.lock change.
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-install-project --no-dev

COPY src ./src
COPY artifacts/encoder ./artifacts/encoder
RUN uv sync --locked --no-dev

EXPOSE 8000
CMD ["uv", "run", "--no-sync", "uvicorn", "newlinefix.api:app", "--host", "0.0.0.0", "--port", "8000"]

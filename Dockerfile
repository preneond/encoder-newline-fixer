# Serving image for the newline-fixer HTTP API.
#
#   docker build -t newlinefix . && docker run -p 8000:8000 newlinefix
#
# No model is baked in: at startup the API resolves its model source (see
# newlinefix.api.default_model_source) and downloads the published checkpoint
# preneond/newlinefix-encoder from the Hugging Face Hub on first boot. To serve
# a local checkpoint instead, mount it and point NEWLINEFIX_MODEL_DIR at it:
#   docker run -p 8000:8000 -v ./artifacts:/app/artifacts newlinefix
#
# Linux resolves CPU-only torch wheels (see [tool.uv.sources] in pyproject.toml),
# keeping the image an order of magnitude smaller than a CUDA build.

FROM ghcr.io/astral-sh/uv:python3.14-bookworm-slim

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

# Dependency layer first: cached until pyproject.toml/uv.lock change.
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-install-project --no-dev

# README.md is the project's declared readme; uv_build needs it to install the package.
COPY README.md ./
COPY src ./src
RUN uv sync --locked --no-dev

EXPOSE 8000
CMD ["uv", "run", "--no-sync", "uvicorn", "newlinefix.api:app", "--host", "0.0.0.0", "--port", "8000"]

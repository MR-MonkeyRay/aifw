# syntax=docker/dockerfile:1
ARG BASE_IMAGE=python:3.13-slim

# ============================================================================
# Builder stage: install dependencies with uv (10-100x faster than pip)
# ============================================================================
FROM ${BASE_IMAGE} AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /opt/aifw

# Install uv (fast Python package installer)
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl ca-certificates && \
    curl -LsSf https://astral.sh/uv/install.sh | sh && \
    rm -rf /var/lib/apt/lists/*

ENV PATH="/root/.local/bin:/root/.cargo/bin:${PATH}"

# Copy requirements first for better cache
COPY cli/python/requirements.txt /opt/aifw/cli/python/requirements.txt
COPY cli/python/services/requirements.txt /opt/aifw/cli/python/services/requirements.txt
COPY libs/aifw-py/requirements.txt /opt/aifw/libs/aifw-py/requirements.txt

# Install dependencies with uv into /opt/venv
RUN uv venv /opt/venv && \
    . /opt/venv/bin/activate && \
    uv pip install --no-cache -r /opt/aifw/cli/python/requirements.txt && \
    uv pip install --no-cache -r /opt/aifw/libs/aifw-py/requirements.txt

# ============================================================================
# Runtime stage: minimal image with only runtime dependencies
# ============================================================================
FROM ${BASE_IMAGE} AS runtime

# OCI labels for metadata
LABEL org.opencontainers.image.title="OneAIFW CLI" \
      org.opencontainers.image.description="AI Framework CLI with PII masking and multi-provider support" \
      org.opencontainers.image.source="https://github.com/funstory-ai/aifw" \
      org.opencontainers.image.vendor="FunStory AI" \
      org.opencontainers.image.licenses="MIT"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    AIFW_WORK_DIR=/data/aifw \
    XDG_CONFIG_HOME=/data/config \
    AIFW_MODELS_BASE=/opt/aifw/ner-models \
    PATH="/opt/venv/bin:${PATH}"

WORKDIR /opt/aifw

# System deps (git for model cloning)
RUN apt-get update && \
    apt-get install -y --no-install-recommends git ca-certificates curl && \
    rm -rf /var/lib/apt/lists/* && \
    useradd -r -s /bin/false aifw

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Copy CLI / services code and aifw-py library into image
COPY cli/python /opt/aifw/cli/python
COPY libs/aifw-py /opt/aifw/libs/aifw-py

# Copy assets metadata (whitelist - exclude sensitive files)
COPY assets/aifw.yaml assets/oneaifw_assets_hashes.json /opt/aifw/assets/

# Clone models from GitHub (pinned version for reproducibility)
RUN set -e; \
    mkdir -p "${AIFW_MODELS_BASE}"; \
    if [ -d "/opt/aifw/OneAIFW-Assets/models" ]; then \
      cp -R /opt/aifw/OneAIFW-Assets/models/* "${AIFW_MODELS_BASE}/"; \
    else \
      git clone --depth 1 --branch v0.3.1 https://github.com/funstory-ai/OneAIFW-Assets.git /opt/aifw-assets && \
      cp -R /opt/aifw-assets/models/* "${AIFW_MODELS_BASE}/" && \
      rm -rf /opt/aifw-assets; \
    fi

# Copy prebuilt Zig core shared library if provided by CI (zig-out/lib)
# Note: zig-out/lib must exist in build context (CI provides it, local builds need placeholder)
COPY zig-out/lib /opt/aifw/zig-out/lib

# Create runtime dirs with minimal permissions
RUN mkdir -p ${AIFW_WORK_DIR} /var/log/aifw && \
    chown -R aifw:aifw ${AIFW_WORK_DIR} /var/log/aifw /opt/aifw && \
    chmod -R 750 ${AIFW_WORK_DIR} /var/log/aifw

# Runtime entrypoint: prepare work dir/config and PYTHONPATH
RUN printf '#!/bin/sh\nset -e\n: "${AIFW_WORK_DIR:=/data/aifw}"\nif [ ! -f "${AIFW_WORK_DIR}/aifw.yaml" ] && [ -f "/opt/aifw/assets/aifw.yaml" ]; then\n  cp /opt/aifw/assets/aifw.yaml "${AIFW_WORK_DIR}/aifw.yaml";\nfi\nexport PYTHONPATH="/opt/aifw/cli/python:/opt/aifw/libs/aifw-py:/opt/aifw:${PYTHONPATH:-}"\nexec "$@"\n' > /usr/local/bin/aifw-entrypoint.sh && \
    chmod +x /usr/local/bin/aifw-entrypoint.sh

# Simple CLI wrapper so `aifw` can be used inside the container
RUN printf '#!/bin/sh\nexec python /opt/aifw/cli/python/aifw.py "$@"\n' > /usr/local/bin/aifw && \
    chmod +x /usr/local/bin/aifw

EXPOSE 8844

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8844/api/health || exit 1

USER aifw
ENTRYPOINT ["/usr/local/bin/aifw-entrypoint.sh"]

WORKDIR /opt/aifw/cli/python
CMD ["python", "-m", "uvicorn", "services.app.main:app", "--host", "0.0.0.0", "--port", "8844"]

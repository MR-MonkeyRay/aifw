ARG BASE_IMAGE=python:3.13-slim
FROM ${BASE_IMAGE}

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    AIFW_WORK_DIR=/data/aifw \
    XDG_CONFIG_HOME=/data/config \
    AIFW_MODELS_BASE=/opt/aifw/ner-models

WORKDIR /opt/aifw

# Copy requirements first for better cache
COPY web/requirements.txt /opt/aifw/web/requirements.txt
COPY cli/python/requirements.txt /opt/aifw/cli/python/requirements.txt
COPY libs/aifw-py/requirements.txt /opt/aifw/libs/aifw-py/requirements.txt

# System deps and Python deps
RUN apt-get update && \
    apt-get install -y --no-install-recommends git ca-certificates curl && \
    rm -rf /var/lib/apt/lists/* && \
    useradd -r -s /bin/false aifw && \
    pip install --upgrade pip && \
    pip install --no-cache-dir -r /opt/aifw/web/requirements.txt && \
    pip install --no-cache-dir -r /opt/aifw/cli/python/requirements.txt && \
    pip install --no-cache-dir -r /opt/aifw/libs/aifw-py/requirements.txt && \
    python -m pip cache purge || true

# Copy web application files
COPY web/*.py /opt/aifw/web/
COPY web/templates/ /opt/aifw/web/templates/
COPY web/static/ /opt/aifw/web/static/

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
COPY zig-out/lib /opt/aifw/zig-out/lib

# Create runtime dirs with minimal permissions
RUN mkdir -p ${AIFW_WORK_DIR} /var/log/aifw && \
    chown -R aifw:aifw ${AIFW_WORK_DIR} /var/log/aifw /opt/aifw && \
    chmod -R 750 ${AIFW_WORK_DIR} /var/log/aifw

# Entrypoint: prepare work dir and default config if missing
RUN printf '#!/bin/sh\nset -e\n: "${AIFW_WORK_DIR:=/data/aifw}"\nif [ ! -f "${AIFW_WORK_DIR}/aifw.yaml" ] && [ -f "/opt/aifw/assets/aifw.yaml" ]; then\n  cp /opt/aifw/assets/aifw.yaml "${AIFW_WORK_DIR}/aifw.yaml";\nfi\nexport PYTHONPATH="/opt/aifw:${PYTHONPATH:-}"\nexec "$@"\n' > /usr/local/bin/aifw-entrypoint.sh && \
    chmod +x /usr/local/bin/aifw-entrypoint.sh

ENV PYTHONPATH=/opt/aifw
EXPOSE 5001

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:5001/api/health || exit 1

USER aifw
ENTRYPOINT ["/usr/local/bin/aifw-entrypoint.sh"]

WORKDIR /opt/aifw/web
CMD ["python", "app.py"]

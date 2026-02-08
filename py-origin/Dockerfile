ARG BASE_IMAGE=python:3.13-slim
FROM ${BASE_IMAGE}

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    AIFW_WORK_DIR=/data/aifw \
    XDG_CONFIG_HOME=/data/config

WORKDIR /opt/aifw

# Build-time profile to control spaCy models
ARG SPACY_PROFILE=minimal

# Copy requirements first for better cache (context is repo root)
COPY py-origin/services/requirements.txt /opt/aifw/services/requirements.txt
COPY py-origin/cli/requirements.txt /opt/aifw/cli/requirements.txt

# System deps and Python deps
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/* && \
    useradd -r -s /bin/false aifw && \
    pip install --upgrade pip && \
    pip install --no-cache-dir -r /opt/aifw/services/requirements.txt && \
    python -m pip cache purge || true

# Install spaCy models per profile
RUN set -e; \
    python -m spacy download en_core_web_sm; \
    python -m spacy download zh_core_web_sm || true; \
    if [ "$SPACY_PROFILE" = "fr" ] || [ "$SPACY_PROFILE" = "multi" ]; then python -m spacy download fr_core_news_sm || true; fi; \
    if [ "$SPACY_PROFILE" = "de" ] || [ "$SPACY_PROFILE" = "multi" ]; then python -m spacy download de_core_news_sm || true; fi; \
    if [ "$SPACY_PROFILE" = "ja" ] || [ "$SPACY_PROFILE" = "multi" ]; then python -m spacy download ja_core_news_sm || true; fi; \
    if [ "$SPACY_PROFILE" = "multi" ]; then python -m spacy download xx_ent_wiki_sm || true; fi; \
    find /usr/local/lib -type d -name '__pycache__' -prune -exec rm -rf {} + || true && \
    find /usr/local/lib -type f -name '*.pyc' -delete || true

# Copy only necessary project files to minimize image size (context is repo root)
COPY py-origin/cli/*.py /opt/aifw/cli/
COPY py-origin/aifw/*.py /opt/aifw/aifw/
COPY py-origin/services/app/*.py /opt/aifw/services/app/
COPY py-origin/services/app/*.json /opt/aifw/services/app/
COPY py-origin/services/fake_llm/*.py /opt/aifw/services/fake_llm/

# Copy default config template (whitelist - exclude sensitive files)
COPY assets/aifw.yaml /opt/aifw/assets/

# Create runtime dirs with minimal permissions
RUN mkdir -p ${AIFW_WORK_DIR} /var/log/aifw && \
    chown -R aifw:aifw ${AIFW_WORK_DIR} /var/log/aifw /opt/aifw && \
    chmod -R 750 ${AIFW_WORK_DIR} /var/log/aifw

# Entrypoint: prepare work dir and default config if missing
RUN printf '#!/bin/sh\nset -e\n: "${AIFW_WORK_DIR:=/data/aifw}"\nif [ ! -f "${AIFW_WORK_DIR}/aifw.yaml" ] && [ -f "/opt/aifw/assets/aifw.yaml" ]; then\n  cp /opt/aifw/assets/aifw.yaml "${AIFW_WORK_DIR}/aifw.yaml";\nfi\nexport PYTHONPATH="/opt/aifw:${PYTHONPATH:-}"\nexec "$@"\n' > /usr/local/bin/aifw-entrypoint.sh && \
    chmod +x /usr/local/bin/aifw-entrypoint.sh

ENV PYTHONPATH=/opt/aifw
EXPOSE 8844

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8844/api/health || exit 1

USER aifw
ENTRYPOINT ["/usr/local/bin/aifw-entrypoint.sh"]

CMD ["/bin/bash"]

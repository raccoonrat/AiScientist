FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    bash \
    ca-certificates \
    curl \
    git \
    python3 \
    python3-pip \
    python3-venv \
    unzip \
    zip \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md /opt/aisci/
COPY src /opt/aisci/src
RUN python3 -m pip install --no-cache-dir /opt/aisci

COPY docker/paper-entrypoint.sh /usr/local/bin/aisci-paper-entrypoint
RUN chmod +x /usr/local/bin/aisci-paper-entrypoint

WORKDIR /home/submission
ENTRYPOINT ["/usr/local/bin/aisci-paper-entrypoint"]

FROM hub.byted.org/ubuntu:24.04

ENV WORKSPACE_BASE=/home
ENV PAPER_DIR=/home/paper
ENV SUBMISSION_DIR=/home/submission
ENV LOGS_DIR=/home/logs
ENV AGENT_DIR=/home/agent
ENV HF_HUB_ETAG_TIMEOUT=120
ENV HF_HUB_DOWNLOAD_TIMEOUT=600
ENV DEBIAN_FRONTEND=noninteractive

RUN mkdir -p /etc/pip && printf '[global]\nindex-url = https://bytedpypi.byted.org/simple/\nextra-index-url = https://pypi.org/simple/\ntrusted-host = bytedpypi.byted.org\n' > /etc/pip.conf

RUN mkdir -p ${PAPER_DIR} ${SUBMISSION_DIR} ${LOGS_DIR} ${AGENT_DIR} /workspace/logs

RUN apt-get update && apt-get install -y \
    bash \
    build-essential \
    ca-certificates \
    curl \
    ffmpeg \
    gettext \
    git \
    libgl1 \
    libsm6 \
    libxext6 \
    nano \
    openssh-server \
    p7zip-full \
    python-is-python3 \
    python3 \
    python3-dev \
    python3-pip \
    python3-venv \
    sudo \
    tmux \
    unzip \
    vim \
    wget \
    zip \
    && rm -rf /var/lib/apt/lists/*

ENV JULIA_VERSION=1.10.7
RUN curl -fsSL "https://julialang-s3.julialang.org/bin/linux/x64/1.10/julia-${JULIA_VERSION}-linux-x86_64.tar.gz" | \
    tar -xz -C /usr/local && \
    ln -s /usr/local/julia-${JULIA_VERSION}/bin/julia /usr/local/bin/julia

RUN python3 -m pip install --break-system-packages --no-cache-dir jupyter ipykernel

RUN cd ${SUBMISSION_DIR} && git init && mkdir -p .git/hooks
RUN git config --global user.email "agent@example.com" && \
    git config --global user.name "agent"

COPY pyproject.toml README.md /opt/aisci/
COPY src /opt/aisci/src
RUN python3 -m pip install --break-system-packages --no-cache-dir /opt/aisci

COPY docker/paper-entrypoint.sh /usr/local/bin/aisci-paper-entrypoint
RUN chmod +x /usr/local/bin/aisci-paper-entrypoint

WORKDIR /home/submission
ENTRYPOINT ["/usr/local/bin/aisci-paper-entrypoint"]

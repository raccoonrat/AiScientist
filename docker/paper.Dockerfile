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

WORKDIR /work


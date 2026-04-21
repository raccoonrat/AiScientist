#!/usr/bin/env bash
set -euo pipefail

HTTP_PROXY_VALUE="${http_proxy:-${HTTP_PROXY:-}}"
HTTPS_PROXY_VALUE="${https_proxy:-${HTTPS_PROXY:-}}"
NO_PROXY_VALUE="${no_proxy:-${NO_PROXY:-}}"

# Allow SOCKS-only setups: many tools read ALL_PROXY when HTTP_PROXY is unset.
if [ -z "${HTTP_PROXY_VALUE}" ]; then
  HTTP_PROXY_VALUE="${all_proxy:-${ALL_PROXY:-}}"
fi
if [ -z "${HTTPS_PROXY_VALUE}" ]; then
  HTTPS_PROXY_VALUE="${HTTP_PROXY_VALUE}"
fi

BUILD_ARGS=(
  --network host
  --no-cache
  --platform=linux/amd64
)

if [ -n "${HTTP_PROXY_VALUE}" ]; then
  BUILD_ARGS+=(
    --build-arg "http_proxy=${HTTP_PROXY_VALUE}"
    --build-arg "https_proxy=${HTTPS_PROXY_VALUE}"
    --build-arg "no_proxy=${NO_PROXY_VALUE},.ubuntu.com"
    --build-arg "HTTP_PROXY=${HTTP_PROXY_VALUE}"
    --build-arg "HTTPS_PROXY=${HTTPS_PROXY_VALUE}"
    --build-arg "NO_PROXY=${NO_PROXY_VALUE},.ubuntu.com"
  )
  if [ -n "${all_proxy:-${ALL_PROXY:-}}" ]; then
    AP="${all_proxy:-${ALL_PROXY:-}}"
    BUILD_ARGS+=(--build-arg "ALL_PROXY=${AP}" --build-arg "all_proxy=${AP}")
  fi
fi

docker build "${BUILD_ARGS[@]}" -t aisci-paper:latest -f docker/paper-agent.Dockerfile .

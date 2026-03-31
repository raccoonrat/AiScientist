#!/usr/bin/env bash
set -euo pipefail

mkdir -p /home/paper /home/submission /home/agent /home/logs
exec "$@"

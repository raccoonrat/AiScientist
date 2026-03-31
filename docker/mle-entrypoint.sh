#!/usr/bin/env bash
set -euo pipefail

mkdir -p /home/data /home/code /home/submission /home/agent /home/logs
exec "$@"

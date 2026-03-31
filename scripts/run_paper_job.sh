#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export AISCI_REPO_ROOT="${AISCI_REPO_ROOT:-${REPO_ROOT}}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"

if [[ -x "${REPO_ROOT}/.venv/bin/aisci" ]]; then
  AISCI_BIN="${REPO_ROOT}/.venv/bin/aisci"
else
  AISCI_BIN="uv run aisci"
fi

usage() {
  cat <<'EOF'
Usage:
  scripts/run_paper_job.sh /abs/path/to/paper.pdf [extra aisci paper run args]
  scripts/run_paper_job.sh --pdf /abs/path/to/paper.pdf [extra aisci paper run args]
  scripts/run_paper_job.sh --paper-bundle-zip /abs/path/to/paper_bundle.zip [extra args]
  scripts/run_paper_job.sh --paper-md /abs/path/to/paper.md [extra args]

Examples:
  scripts/run_paper_job.sh /abs/path/to/paper.pdf
  scripts/run_paper_job.sh /abs/path/to/paper.pdf --rubric-path /abs/path/to/rubric.json
  scripts/run_paper_job.sh --paper-bundle-zip /abs/path/to/paper_bundle.zip --gpus 1

This is only a thin convenience wrapper over:
  aisci paper run --wait ...
EOF
}

args=("$@")

if (($# == 0)); then
  usage
  exit 2
fi

case "${1:-}" in
  -h|--help)
    usage
    exit 0
    ;;
esac

if [[ "${1}" != --* ]]; then
  args=(--pdf "$1" "${@:2}")
fi

has_mode_flag=0
for arg in "${args[@]}"; do
  case "${arg}" in
    --wait|--detach)
      has_mode_flag=1
      break
      ;;
  esac
done

if ((has_mode_flag == 0)); then
  args+=(--wait)
fi

cd "${REPO_ROOT}"
if [[ "${AISCI_BIN}" == "uv run aisci" ]]; then
  uv run aisci paper run "${args[@]}"
else
  "${AISCI_BIN}" paper run "${args[@]}"
fi

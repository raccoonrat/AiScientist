# AiScientist

Independent AI Scientist workbench for `paper` and `mle` jobs.

## Quick Start

```bash
uv sync
uv run aisci --help
uv run aisci serve --host 127.0.0.1 --port 8080
```

## What v1 Implements

- Unified SQLite-backed job store and filesystem layout under `jobs/<job_id>/`
- Shared Docker runtime API with per-mode default profiles
- `paper` and `mle` job staging adapters with prompt-pack artifacts
- CLI commands and a minimal Web workbench for jobs, details, artifacts, and export

The current adapters focus on staging, artifact generation, and runtime unification.
They do not yet reproduce the full upstream multi-subagent execution loops from
`paperbench` and `mle-bench`.

## Layout

- `src/aisci_core`: shared models, job store, worker, export
- `src/aisci_runtime_docker`: unified Docker runtime API
- `src/aisci_domain_paper`: paper-mode staging and validation
- `src/aisci_domain_mle`: mle-mode staging and validation
- `src/aisci_app`: CLI and Web app

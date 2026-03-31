# AiScientist

Independent AI Scientist workbench for `paper` and `mle` jobs.

## Quick Start

```bash
uv sync
uv run aisci --help
uv run aisci serve --host 127.0.0.1 --port 8080
```

## Paper Mode Prerequisites

`paper` jobs will not start unless all of the following are true:

- Python `>=3.12`
- Docker daemon is reachable from the host
- `OPENAI_API_KEY` or `AZURE_OPENAI_API_KEY` is set in the host environment

Optional but recommended:

- `AISCI_MAX_STEPS`: overrides the default paper loop step budget (`80`)
- `AISCI_REMINDER_FREQ`: overrides the default reminder interval (`5`)
- `AISCI_REPO_ROOT`: override repo root detection if you are not running commands from the repo root

Sanity-check the local environment before running:

```bash
uv run aisci paper doctor
```

## Paper Job Runtime Model

Current `paper` runs use a single Docker work environment:

- Default Dockerfile: `docker/paper-agent.Dockerfile`
- Main loop: runs `python -m aisci_domain_paper.orchestrator`
- Final validation: if enabled, starts a fresh container from the same image and runs `bash reproduce.sh`

Inside the container the canonical paths are:

- `/home/paper`: staged paper inputs
- `/home/submission`: working repo and final `reproduce.sh`
- `/home/agent`: analysis and planning artifacts
- `/home/logs`: agent/runtime logs

## Paper CLI

The main entrypoint is:

```bash
uv run aisci paper run [OPTIONS]
```

At least one of the following inputs must be provided:

- `--pdf PATH`: path to a paper PDF
- `--paper-bundle-zip PATH`: zip extracted into `/home/paper`
- `--paper-md PATH`: markdown copy of the paper

Other `paper run` options exposed by the current CLI:

- `--llm-profile TEXT`: default `gpt-5.4-responses`
- `--gpus INT`: default `0`
- `--time-limit TEXT`: default `24h`; parsed from units like `30m`, `8h`, `1d12h`
- `--inputs-zip PATH`: extra context bundle extracted into `/home/paper`
- `--rubric-path PATH`: copied to `/home/paper/rubric.json`
- `--blacklist-path PATH`: copied to `/home/paper/blacklist.txt`
- `--addendum-path PATH`: copied to `/home/paper/addendum.md`
- `--submission-seed-repo-zip PATH`: extracted into `/home/submission`
- `--dockerfile PATH`: advanced override; if omitted, the default paper work-environment image is used
- `--supporting-materials PATH`: repeatable option; each file is copied into `/home/paper`
- `--run-final-validation` / `--skip-final-validation`: default is enabled
- `--detach` / `--wait`: default is detached

Current `paper run` CLI defaults that are not user-configurable from the CLI:

- `objective="paper reproduction job"`
- `enable_online_research=True`

If you need to change those values today, use the Web form or construct the `JobSpec` in Python.

## Recommended Paper Usage

Minimal run with a PDF:

```bash
export OPENAI_API_KEY=...

uv run aisci paper run \
  --pdf /abs/path/to/paper.pdf \
  --wait
```

A more complete run with staged context:

```bash
uv run aisci paper run \
  --pdf /abs/path/to/paper.pdf \
  --inputs-zip /abs/path/to/context_bundle.zip \
  --rubric-path /abs/path/to/rubric.json \
  --addendum-path /abs/path/to/addendum.md \
  --submission-seed-repo-zip /abs/path/to/seed_repo.zip \
  --supporting-materials /abs/path/to/notes.md \
  --supporting-materials /abs/path/to/diagram.png \
  --time-limit 12h \
  --llm-profile gpt-5.4-responses \
  --wait
```

If you prefer background execution:

```bash
uv run aisci paper run --pdf /abs/path/to/paper.pdf
```

Detached mode returns JSON containing the new `job_id`. Use that `job_id` for inspection commands below.

## Inspecting Runs

List jobs:

```bash
uv run aisci jobs list
```

Show one job with events and recorded artifacts:

```bash
uv run aisci jobs show <job_id>
```

List available logs for a run:

```bash
uv run aisci logs list <job_id>
```

Tail logs:

```bash
uv run aisci logs tail <job_id> --kind main
uv run aisci logs tail <job_id> --kind conversation
uv run aisci logs tail <job_id> --kind agent
uv run aisci logs tail <job_id> --kind subagent
uv run aisci logs tail <job_id> --kind validation
uv run aisci logs tail <job_id> --kind all
```

List recorded artifacts:

```bash
uv run aisci artifacts ls <job_id>
```

Export a job bundle:

```bash
uv run aisci export <job_id>
```

## Re-Run Validation Or Resume

Start a fresh self-check job from an existing paper run:

```bash
uv run aisci paper validate <job_id> --wait
```

Resume from an existing paper job spec:

```bash
uv run aisci paper resume <job_id> --wait
```

## Web UI

Start the local Web workbench:

```bash
uv run aisci serve --host 127.0.0.1 --port 8080
```

Then open `http://127.0.0.1:8080/`.

The Web form currently exposes a few `paper` fields that the CLI does not, including:

- `objective`
- `enable_online_research`

## Output Layout

Each run writes under `jobs/<job_id>/`:

- `input/`: copied raw inputs
- `workspace/paper/`: staged paper materials
- `workspace/submission/`: working repo and final `reproduce.sh`
- `workspace/agent/`: summaries, plans, logs, self-check outputs
- `logs/`: job log, agent log, conversation log, subagent logs
- `artifacts/`: persisted `validation_report.json` and exported bundle metadata
- `export/`: zip bundle for the run

Useful files to inspect first for `paper` runs:

- `jobs/<job_id>/workspace/agent/paper_analysis/summary.md`
- `jobs/<job_id>/workspace/agent/prioritized_tasks.md`
- `jobs/<job_id>/workspace/submission/reproduce.sh`
- `jobs/<job_id>/workspace/agent/final_self_check.md`
- `jobs/<job_id>/logs/agent.log`

## What v1 Implements

- Unified SQLite-backed job store and filesystem layout under `jobs/<job_id>/`
- Shared Docker runtime API with per-mode default profiles
- Upstream-aligned `paper` AI Scientist loop: `read_paper -> prioritize_tasks -> implement -> run_experiment -> clean_reproduce_validation -> submit`
- `mle` job staging adapter with prompt-pack artifacts and validation plumbing
- CLI commands and a minimal Web workbench for jobs, details, artifacts, and export

The paper mode now carries the default upstream AI Scientist execution path from
`paperbench`. Experimental or unhooked upstream modules are intentionally not
part of the alignment target. The `mle` mode remains focused on staging,
artifact generation, and runtime unification rather than a full upstream loop.

## Layout

- `src/aisci_core`: shared models, job store, worker, export
- `src/aisci_runtime_docker`: unified Docker runtime API
- `src/aisci_domain_paper`: paper-mode staging and validation
- `src/aisci_domain_mle`: mle-mode staging and validation
- `src/aisci_app`: CLI and Web app

from __future__ import annotations

import os
from pathlib import Path

from aisci_core.models import JobPaths


def repo_root() -> Path:
    return Path(os.environ.get("AISCI_REPO_ROOT", Path.cwd())).resolve()


def var_root() -> Path:
    return repo_root()


def jobs_root() -> Path:
    return var_root() / "jobs"


def state_root() -> Path:
    return var_root() / ".aisci"


def database_path() -> Path:
    return state_root() / "state" / "jobs.db"


def resolve_job_paths(job_id: str) -> JobPaths:
    root = jobs_root() / job_id
    return JobPaths(
        root=root,
        input_dir=root / "input",
        workspace_dir=root / "workspace",
        logs_dir=root / "logs",
        artifacts_dir=root / "artifacts",
        export_dir=root / "export",
        state_dir=root / "state",
    )


def ensure_job_dirs(job_paths: JobPaths) -> JobPaths:
    for path in (
        jobs_root(),
        state_root() / "state",
        job_paths.root,
        job_paths.input_dir,
        job_paths.workspace_dir,
        job_paths.logs_dir,
        job_paths.artifacts_dir,
        job_paths.export_dir,
        job_paths.state_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)
    return job_paths


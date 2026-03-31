from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from aisci_core.models import RuntimeProfile, WorkspaceLayout


@dataclass(frozen=True)
class DockerProfile:
    name: str
    dockerfile_path: Path
    context_dir: Path
    image_tag: str
    default_command: list[str] = field(default_factory=lambda: ["sleep", "infinity"])


@dataclass(frozen=True)
class SessionMount:
    source: Path
    target: str
    read_only: bool = False


@dataclass(frozen=True)
class SessionSpec:
    job_id: str
    workspace_layout: WorkspaceLayout
    profile: DockerProfile
    runtime_profile: RuntimeProfile
    mounts: tuple[SessionMount, ...]
    workdir: str
    entry_command: tuple[str, ...] = ()
    env: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class ContainerSession:
    container_name: str
    image_tag: str
    profile: DockerProfile
    runtime_profile: RuntimeProfile
    workspace_layout: WorkspaceLayout
    mounts: tuple[SessionMount, ...] = ()
    workdir: str = "/workspace"


@dataclass(frozen=True)
class DockerExecutionResult:
    command: list[str]
    exit_code: int
    stdout: str
    stderr: str

    @property
    def combined_output(self) -> str:
        if self.stdout and self.stderr:
            return f"{self.stdout}\n{self.stderr}"
        return self.stdout or self.stderr

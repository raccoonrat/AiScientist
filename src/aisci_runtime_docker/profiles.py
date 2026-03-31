from __future__ import annotations

import hashlib
from pathlib import Path

from aisci_runtime_docker.models import DockerProfile
from aisci_core.paths import repo_root


def _tag_from_file(prefix: str, dockerfile_path: Path) -> str:
    digest = hashlib.sha256(dockerfile_path.read_bytes()).hexdigest()[:12]
    return f"aisci-{prefix}:{digest}"


def default_paper_profile() -> DockerProfile:
    dockerfile_path = repo_root() / "docker" / "paper-agent.Dockerfile"
    if not dockerfile_path.exists():
        dockerfile_path = repo_root() / "docker" / "paper.Dockerfile"
    return DockerProfile(
        name="paper-default",
        dockerfile_path=dockerfile_path,
        context_dir=repo_root(),
        image_tag=_tag_from_file("paper", dockerfile_path),
    )


def default_mle_profile() -> DockerProfile:
    dockerfile_path = repo_root() / "docker" / "mle-agent.Dockerfile"
    if not dockerfile_path.exists():
        dockerfile_path = repo_root() / "docker" / "mle.Dockerfile"
    return DockerProfile(
        name="mle-default",
        dockerfile_path=dockerfile_path,
        context_dir=repo_root(),
        image_tag=_tag_from_file("mle", dockerfile_path),
    )

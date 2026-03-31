from __future__ import annotations

import hashlib
import json
import secrets
import shutil
import subprocess
from pathlib import Path

from aisci_core.models import (
    JobPaths,
    NetworkPolicy,
    RuntimeProfile,
    ValidationReport,
    WorkspaceLayout,
)
from aisci_runtime_docker.models import (
    ContainerSession,
    DockerExecutionResult,
    DockerProfile,
    SessionMount,
    SessionSpec,
)


class DockerRuntimeError(RuntimeError):
    pass


class AgentSessionManager:
    def __init__(self):
        self._docker = shutil.which("docker") or "docker"

    def can_use_docker(self) -> bool:
        if shutil.which(self._docker) is None and shutil.which("docker") is None:
            return False
        try:
            return self._run([self._docker, "info"], check=False).exit_code == 0
        except Exception:
            return False

    def build_profile(self, profile: DockerProfile, runtime_profile: RuntimeProfile) -> str:
        dockerfile_path = (
            Path(runtime_profile.dockerfile_path).resolve()
            if runtime_profile.dockerfile_path
            else profile.dockerfile_path
        )
        digest = hashlib.sha256(
            (
                dockerfile_path.read_text(encoding="utf-8")
                + json.dumps(runtime_profile.model_dump(mode="json"), sort_keys=True)
            ).encode("utf-8")
        ).hexdigest()[:12]
        image_tag = f"{profile.image_tag.split(':', 1)[0]}:{digest}"
        self._run(
            [
                self._docker,
                "build",
                "-f",
                str(dockerfile_path),
                "-t",
                image_tag,
                str(profile.context_dir),
            ]
        )
        return image_tag

    def ensure_layout(self, job_paths: JobPaths, layout: WorkspaceLayout) -> None:
        for path in self._layout_paths(job_paths, layout).values():
            path.mkdir(parents=True, exist_ok=True)

    def create_session_spec(
        self,
        job_id: str,
        job_paths: JobPaths,
        profile: DockerProfile,
        runtime_profile: RuntimeProfile,
        *,
        layout: WorkspaceLayout,
        entry_command: list[str] | tuple[str, ...] = (),
        env: dict[str, str] | None = None,
        workdir: str | None = None,
    ) -> SessionSpec:
        mounts = tuple(self._layout_mounts(job_paths, layout))
        return SessionSpec(
            job_id=job_id,
            workspace_layout=layout,
            profile=profile,
            runtime_profile=runtime_profile,
            mounts=mounts,
            workdir=workdir or self._default_workdir(layout),
            entry_command=tuple(entry_command),
            env=tuple(sorted((env or {}).items())),
        )

    def start_session(self, spec: SessionSpec, image_tag: str) -> ContainerSession:
        container_name = f"aisci-{spec.job_id}-{secrets.token_hex(4)}"
        command = [
            self._docker,
            "run",
            "-d",
            "--name",
            container_name,
            "-w",
            spec.workdir,
            *self._network_args(spec.runtime_profile.network_policy),
            *self._gpu_args(spec.runtime_profile),
        ]
        for mount in spec.mounts:
            source = str(mount.source.resolve())
            target = mount.target
            suffix = ":ro" if mount.read_only else ""
            command.extend(["-v", f"{source}:{target}{suffix}"])
        for key, value in spec.env:
            command.extend(["-e", f"{key}={value}"])
        command.append(image_tag)
        command.extend(spec.entry_command or tuple(spec.profile.default_command))
        self._run(command)
        return ContainerSession(
            container_name=container_name,
            image_tag=image_tag,
            profile=spec.profile,
            runtime_profile=spec.runtime_profile,
            workspace_layout=spec.workspace_layout,
            mounts=spec.mounts,
            workdir=spec.workdir,
        )

    def exec(
        self,
        session: ContainerSession,
        command: str,
        *,
        workdir: str | None = None,
        env: dict[str, str] | None = None,
        check: bool = False,
    ) -> DockerExecutionResult:
        cmd = [self._docker, "exec", "-w", workdir or session.workdir]
        for key, value in (env or {}).items():
            cmd.extend(["-e", f"{key}={value}"])
        cmd.extend([session.container_name, "bash", "-lc", command])
        return self._run(cmd, check=check)

    def run_validation(
        self,
        spec: SessionSpec,
        image_tag: str,
        validation_command: str,
        *,
        workdir: str | None = None,
    ) -> ValidationReport:
        started = ValidationReport(
            status="skipped",
            summary="placeholder",
            runtime_profile_hash=self._runtime_hash(spec.runtime_profile),
            container_image=image_tag,
        ).started_at
        session = self.start_session(spec, image_tag)
        try:
            result = self.exec(session, validation_command, workdir=workdir, check=False)
            status = "passed" if result.exit_code == 0 else "failed"
            summary = (
                "Validation completed successfully."
                if result.exit_code == 0
                else "Validation command failed."
            )
            return ValidationReport(
                status=status,
                summary=summary,
                details={
                    "command": validation_command,
                    "exit_code": result.exit_code,
                    "output": result.combined_output,
                },
                runtime_profile_hash=self._runtime_hash(spec.runtime_profile),
                container_image=image_tag,
                started_at=started,
            )
        finally:
            self.cleanup(session)

    def cleanup(self, session: ContainerSession) -> None:
        self._run([self._docker, "rm", "-f", session.container_name], check=False)

    def collect_artifacts(self, job_paths: JobPaths) -> list[Path]:
        return [path for path in sorted(job_paths.artifacts_dir.rglob("*")) if path.is_file()]

    def _layout_paths(self, job_paths: JobPaths, layout: WorkspaceLayout) -> dict[str, Path]:
        if layout == WorkspaceLayout.PAPER:
            return {
                "paper": job_paths.workspace_dir / "paper",
                "submission": job_paths.workspace_dir / "submission",
                "agent": job_paths.workspace_dir / "agent",
                "logs": job_paths.workspace_dir / "logs",
            }
        return {
            "data": job_paths.workspace_dir / "data",
            "code": job_paths.workspace_dir / "code",
            "submission": job_paths.workspace_dir / "submission",
            "agent": job_paths.workspace_dir / "agent",
            "logs": job_paths.workspace_dir / "logs",
        }

    def _layout_mounts(self, job_paths: JobPaths, layout: WorkspaceLayout) -> list[SessionMount]:
        paths = self._layout_paths(job_paths, layout)
        mounts = [SessionMount(job_paths.logs_dir, "/home/logs")]
        if layout == WorkspaceLayout.PAPER:
            mounts.extend(
                [
                    SessionMount(paths["paper"], "/home/paper"),
                    SessionMount(paths["submission"], "/home/submission"),
                    SessionMount(paths["agent"], "/home/agent"),
                    SessionMount(paths["logs"], "/workspace/logs"),
                ]
            )
        else:
            mounts.extend(
                [
                    SessionMount(paths["data"], "/home/data"),
                    SessionMount(paths["code"], "/home/code"),
                    SessionMount(paths["submission"], "/home/submission"),
                    SessionMount(paths["agent"], "/home/agent"),
                    SessionMount(paths["logs"], "/workspace/logs"),
                ]
            )
        return mounts

    def _default_workdir(self, layout: WorkspaceLayout) -> str:
        if layout == WorkspaceLayout.PAPER:
            return "/home/submission"
        return "/home/code"

    def _run(self, command: list[str], check: bool = True) -> DockerExecutionResult:
        completed = subprocess.run(command, text=True, capture_output=True)
        result = DockerExecutionResult(
            command=command,
            exit_code=completed.returncode,
            stdout=(completed.stdout or "").strip(),
            stderr=(completed.stderr or "").strip(),
        )
        if check and result.exit_code != 0:
            raise DockerRuntimeError(
                result.combined_output or f"Docker command failed: {' '.join(command)}"
            )
        return result

    def _network_args(self, policy: NetworkPolicy) -> list[str]:
        if policy == NetworkPolicy.HOST:
            return ["--network", "host"]
        if policy == NetworkPolicy.NONE:
            return ["--network", "none"]
        return []

    def _gpu_args(self, runtime_profile: RuntimeProfile) -> list[str]:
        if runtime_profile.gpu_ids:
            return ["--gpus", f"device={','.join(runtime_profile.gpu_ids)}"]
        if runtime_profile.gpu_count > 0:
            return ["--gpus", str(runtime_profile.gpu_count)]
        return []

    def _runtime_hash(self, runtime_profile: RuntimeProfile) -> str:
        return hashlib.sha256(
            json.dumps(runtime_profile.model_dump(mode="json"), sort_keys=True).encode("utf-8")
        ).hexdigest()[:12]

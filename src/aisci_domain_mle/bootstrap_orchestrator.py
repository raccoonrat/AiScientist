from __future__ import annotations

import json
import shutil
from pathlib import Path

from aisci_agent_runtime.llm_profiles import resolve_llm_profile
from aisci_agent_runtime.trace import AgentTraceWriter
from aisci_core.models import ArtifactRecord, JobRecord, RunPhase
from aisci_domain_mle.candidate_registry import CandidateRegistry
from aisci_domain_mle.constants import MAIN_AGENT_WORKSPACE_REFERENCE


class BootstrapMLEOrchestrator:
    """Fallback path when Docker is unavailable.

    This preserves the previous staged-artifact behavior so local unit tests can
    run without a container, while production runs use the real upstream loop.
    """

    def run(self, job: JobRecord, job_paths) -> list[ArtifactRecord]:
        data_dir = job_paths.workspace_dir / "data"
        submission_dir = job_paths.workspace_dir / "submission"
        agent_dir = job_paths.workspace_dir / "agent"
        analysis_dir = agent_dir / "analysis"
        logs_dir = job_paths.logs_dir
        subagent_logs = logs_dir / "subagent_logs"
        for path in (
            data_dir,
            submission_dir,
            agent_dir,
            analysis_dir,
            logs_dir,
            subagent_logs,
        ):
            path.mkdir(parents=True, exist_ok=True)

        trace = AgentTraceWriter(logs_dir)
        llm_profile = resolve_llm_profile(job.llm_profile)
        trace.event(
            "agent_step",
            "Bootstrap MLE orchestrator started.",
            phase=RunPhase.ANALYZE.value,
            payload={
                "provider": llm_profile.provider,
                "model": llm_profile.model,
                "api_mode": llm_profile.api_mode,
            },
        )
        summary_path = analysis_dir / "summary.md"
        summary_path.write_text(self._build_analysis_summary(job, data_dir), encoding="utf-8")
        prioritized_tasks = agent_dir / "prioritized_tasks.md"
        prioritized_tasks.write_text(self._build_priorities(job), encoding="utf-8")
        impl_log = agent_dir / "impl_log.md"
        exp_log = agent_dir / "exp_log.md"
        impl_log.write_text("# Implementation Log\n\n", encoding="utf-8")
        exp_log.write_text("# Experiment Log\n\n", encoding="utf-8")
        for name in ("analysis_001", "prioritization_001", "implementation_001", "experiment_001"):
            (subagent_logs / name).mkdir(parents=True, exist_ok=True)

        submission_path = submission_dir / "submission.csv"
        if not submission_path.exists():
            sample_submission = self._find_first(data_dir, "sample_submission.csv")
            if sample_submission is not None:
                shutil.copy2(sample_submission, submission_path)
            else:
                submission_path.write_text("id,target\n", encoding="utf-8")

        registry = CandidateRegistry(
            registry_path=submission_dir / "submission_registry.jsonl",
            candidates_dir=submission_dir / "candidates",
        )
        registry.append(
            "system_snapshot",
            objective=job.objective,
            llm_profile=job.llm_profile,
            workspace_reference=MAIN_AGENT_WORKSPACE_REFERENCE,
        )
        snapshot = registry.snapshot_submission(
            submission_path,
            reason="bootstrap_seed",
            method_summary="Seeded submission for no-docker fallback mode.",
            metrics=None,
            eval_protocol="shape_only",
        )
        registry.select_champion(
            snapshot.snapshot,
            rationale="Only seeded candidate available in bootstrap mode.",
            metrics=None,
            eval_protocol="shape_only",
        )

        champion_report = job_paths.artifacts_dir / "champion_report.md"
        champion_report.write_text(
            "# Champion Report\n\n"
            f"- Champion path: `{snapshot.snapshot}`\n"
            f"- Registry: `{registry.registry_path}`\n",
            encoding="utf-8",
        )
        state_path = job_paths.artifacts_dir / "mle_bootstrap_state.json"
        state_path.write_text(
            json.dumps(
                {
                    "job_id": job.id,
                    "llm_profile": llm_profile.__dict__,
                    "submission_path": str(submission_path),
                    "candidate_path": str(snapshot.snapshot),
                    "registry_path": str(registry.registry_path),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        trace.write_state(
            phase=RunPhase.FINALIZE.value,
            submission_path=str(submission_path),
            candidate_path=str(snapshot.snapshot),
        )
        trace.event(
            "agent_step",
            "Bootstrap MLE orchestrator completed.",
            phase=RunPhase.FINALIZE.value,
            payload={"candidate_path": str(snapshot.snapshot)},
        )
        artifacts = []
        for artifact_type, path, phase, metadata in (
            ("analysis_summary", summary_path, RunPhase.ANALYZE, {}),
            ("prioritized_tasks", prioritized_tasks, RunPhase.PRIORITIZE, {}),
            ("impl_log", impl_log, RunPhase.IMPLEMENT, {}),
            ("exp_log", exp_log, RunPhase.VALIDATE, {}),
            ("submission_registry", registry.registry_path, RunPhase.FINALIZE, {}),
            ("candidate_snapshot", snapshot.snapshot, RunPhase.FINALIZE, {"reason": "bootstrap_seed"}),
            ("champion_report", champion_report, RunPhase.FINALIZE, {}),
            ("bootstrap_state", state_path, RunPhase.FINALIZE, {}),
        ):
            artifacts.append(
                ArtifactRecord(
                    artifact_type=artifact_type,
                    path=str(path),
                    phase=phase,
                    size_bytes=path.stat().st_size,
                    metadata=metadata,
                )
            )
        return artifacts

    def _build_analysis_summary(self, job: JobRecord, data_dir: Path) -> str:
        files = sorted(str(path.relative_to(data_dir)) for path in data_dir.rglob("*") if path.is_file())
        contracts = [
            item
            for item in files
            if item.endswith("sample_submission.csv")
            or item.endswith("eval.py")
            or item.endswith("eval_cmd.txt")
        ]
        return (
            "# MLE Analysis Summary\n\n"
            f"- Objective: {job.objective}\n"
            f"- LLM Profile: `{job.llm_profile}`\n"
            f"- Files Staged: {len(files)}\n"
            f"- Validation Contracts: {', '.join(contracts) if contracts else 'none detected'}\n\n"
            "## Workspace Reference\n\n"
            f"{MAIN_AGENT_WORKSPACE_REFERENCE}\n"
        )

    def _build_priorities(self, job: JobRecord) -> str:
        return (
            "# Prioritized Tasks\n\n"
            "## P0\n\n"
            "- Analyze data shape, metric, and output contract.\n"
            "- Keep `submission.csv` valid against `sample_submission.csv` at every major step.\n"
            "- Record every concrete candidate in `submission_registry.jsonl`.\n\n"
            "## P1\n\n"
            "- Build baseline training and inference loop inside `/home/code`.\n"
            "- Use `implement(full)` breadth-first before deep optimization.\n"
        )

    def _find_first(self, root: Path, filename: str) -> Path | None:
        for path in root.rglob(filename):
            return path
        return None

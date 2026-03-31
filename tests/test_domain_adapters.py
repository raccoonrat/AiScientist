from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile

from pypdf import PdfWriter

from aisci_core.models import (
    JobRecord,
    JobStatus,
    JobType,
    MLESpec,
    PaperSpec,
    RunPhase,
    RuntimeProfile,
    WorkspaceLayout,
)
from aisci_core.paths import ensure_job_dirs, resolve_job_paths
from aisci_domain_mle.adapter import MLEDomainAdapter
from aisci_domain_paper.adapter import PaperDomainAdapter
from aisci_runtime_docker.profiles import default_mle_profile
from aisci_runtime_docker.runtime import DockerRuntimeManager


def _make_pdf(path: Path) -> None:
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    with path.open("wb") as handle:
        writer.write(handle)


def _paper_job(tmp_path: Path, pdf_path: Path) -> JobRecord:
    now = __import__("datetime").datetime.now().astimezone()
    return JobRecord(
        id="paper-job",
        job_type=JobType.PAPER,
        status=JobStatus.PENDING,
        phase=RunPhase.INGEST,
        objective="paper objective",
        llm_profile="test",
        runtime_profile=RuntimeProfile(
            run_final_validation=False,
            workspace_layout=WorkspaceLayout.PAPER,
        ),
        mode_spec=PaperSpec(pdf_path=str(pdf_path)),
        created_at=now,
        updated_at=now,
    )


def _mle_job(tmp_path: Path, bundle_path: Path) -> JobRecord:
    now = __import__("datetime").datetime.now().astimezone()
    return JobRecord(
        id="mle-job",
        job_type=JobType.MLE,
        status=JobStatus.PENDING,
        phase=RunPhase.INGEST,
        objective="mle objective",
        llm_profile="test",
        runtime_profile=RuntimeProfile(
            run_final_validation=False,
            workspace_layout=WorkspaceLayout.MLE,
        ),
        mode_spec=MLESpec(workspace_bundle_zip=str(bundle_path)),
        created_at=now,
        updated_at=now,
    )


def test_paper_adapter_stages_artifacts(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("AISCI_REPO_ROOT", str(tmp_path))
    (tmp_path / "docker").mkdir(parents=True, exist_ok=True)
    (tmp_path / "docker" / "paper.Dockerfile").write_text("FROM scratch\n", encoding="utf-8")
    pdf_path = tmp_path / "sample.pdf"
    _make_pdf(pdf_path)
    adapter = PaperDomainAdapter(DockerRuntimeManager())
    result = adapter.run(_paper_job(tmp_path, pdf_path))
    assert result["validation_report"].status == "skipped"
    artifact_types = {item.artifact_type for item in result["artifacts"]}
    assert "paper_analysis" in artifact_types
    assert "prioritized_tasks" in artifact_types
    assert "capabilities" in artifact_types
    assert "self_check_report" in artifact_types
    job_paths = ensure_job_dirs(resolve_job_paths("paper-job"))
    assert (job_paths.workspace_dir / "submission" / "reproduce.sh").exists()
    assert (job_paths.workspace_dir / "agent" / "paper_analysis" / "summary.md").exists()
    assert (job_paths.workspace_dir / "agent" / "capabilities.json").exists()
    assert (job_paths.workspace_dir / "agent" / "final_self_check.md").exists()


def test_mle_adapter_stages_summary_and_submission(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("AISCI_REPO_ROOT", str(tmp_path))
    (tmp_path / "docker").mkdir(parents=True, exist_ok=True)
    (tmp_path / "docker" / "mle.Dockerfile").write_text("FROM scratch\n", encoding="utf-8")
    bundle = tmp_path / "workspace.zip"
    with ZipFile(bundle, "w") as zf:
        zf.writestr("description.md", "# task")
        zf.writestr("sample_submission.csv", "id,target\n1,0\n")
    adapter = MLEDomainAdapter(DockerRuntimeManager())
    result = adapter.run(_mle_job(tmp_path, bundle))
    assert result["validation_report"].status == "skipped"
    artifact_types = {item.artifact_type for item in result["artifacts"]}
    assert "champion_report" in artifact_types
    assert "submission_registry" in artifact_types
    assert "candidate_snapshot" in artifact_types
    job_paths = ensure_job_dirs(resolve_job_paths("mle-job"))
    registry_text = (job_paths.workspace_dir / "submission" / "submission_registry.jsonl").read_text(
        encoding="utf-8"
    )
    assert "champion_selected" in registry_text
    assert (job_paths.workspace_dir / "submission" / "submission.csv").exists()


def test_runtime_session_spec_uses_canonical_mounts(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("AISCI_REPO_ROOT", str(tmp_path))
    (tmp_path / "docker").mkdir(parents=True, exist_ok=True)
    (tmp_path / "docker" / "mle.Dockerfile").write_text("FROM scratch\n", encoding="utf-8")
    runtime = DockerRuntimeManager()
    job_paths = ensure_job_dirs(resolve_job_paths("layout-job"))
    runtime.ensure_layout(job_paths, WorkspaceLayout.MLE)
    spec = runtime.create_session_spec(
        "layout-job",
        job_paths,
        default_mle_profile(),
        RuntimeProfile(workspace_layout=WorkspaceLayout.MLE),
        layout=WorkspaceLayout.MLE,
        workdir="/home/code",
    )
    targets = {mount.target for mount in spec.mounts}
    assert "/home/data" in targets
    assert "/home/code" in targets
    assert "/home/submission" in targets
    assert spec.workdir == "/home/code"

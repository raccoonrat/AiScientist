from __future__ import annotations

import json
from pathlib import Path

from aisci_core.models import JobSpec, JobType, MLESpec, PaperSpec, RuntimeProfile, RunPhase
from aisci_core.paths import ensure_job_dirs, resolve_job_paths
from aisci_core.runner import JobRunner
from aisci_core.store import JobStore


def test_store_create_list_and_events(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs.db")
    paper = store.create_job(
        JobSpec(
            job_type=JobType.PAPER,
            objective="paper test",
            llm_profile="test-profile",
            runtime_profile=RuntimeProfile(),
            mode_spec=PaperSpec(pdf_path="/tmp/paper.pdf"),
        )
    )
    mle = store.create_job(
        JobSpec(
            job_type=JobType.MLE,
            objective="mle test",
            llm_profile="test-profile",
            runtime_profile=RuntimeProfile(),
            mode_spec=MLESpec(workspace_bundle_zip="/tmp/workspace.zip"),
        )
    )
    assert store.get_job(paper.id).job_type == JobType.PAPER
    assert len(store.list_jobs()) == 2
    event = store.append_event(paper.id, "status", RunPhase.INGEST, "created")
    events = store.list_events(paper.id)
    assert event.id == events[0].id
    assert store.get_job(mle.id).mode_spec.workspace_bundle_zip == "/tmp/workspace.zip"
    assert store.get_job(paper.id).mode_spec.enable_online_research is True


def test_runner_ingests_conversation_jsonl(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("AISCI_REPO_ROOT", str(tmp_path))
    store = JobStore(tmp_path / "jobs.db")
    runner = JobRunner(store=store)
    job = store.create_job(
        JobSpec(
            job_type=JobType.PAPER,
            objective="paper test",
            llm_profile="test-profile",
            runtime_profile=RuntimeProfile(),
            mode_spec=PaperSpec(pdf_path="/tmp/paper.pdf"),
        )
    )
    paths = ensure_job_dirs(resolve_job_paths(job.id))
    conversation = [
        {
            "ts": 1.0,
            "event": "model_response",
            "text": "Read the paper and create a plan.",
            "tool_calls": [{"name": "read_paper"}],
        },
        {
            "ts": 2.0,
            "event": "tool_result",
            "tool": "prioritize_tasks",
            "result_preview": "wrote prioritized_tasks.md",
        },
    ]
    with (paths.logs_dir / "conversation.jsonl").open("w", encoding="utf-8") as handle:
        for row in conversation:
            handle.write(json.dumps(row) + "\n")

    runner._ingest_conversation_events(job.id, paths.logs_dir / "conversation.jsonl")

    events = store.list_events(job.id)
    assert any(event.event_type == "model_response" for event in events)
    assert any(event.event_type == "tool_result" for event in events)
    assert any(event.phase == RunPhase.ANALYZE for event in events)
    assert any(event.phase == RunPhase.PRIORITIZE for event in events)

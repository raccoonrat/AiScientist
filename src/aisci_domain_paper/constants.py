from __future__ import annotations

from pathlib import Path

from aisci_core.models import JobPaths

MAIN_AGENT_WORKSPACE_REFERENCE = """## Workspace Reference

| Path | Content |
|------|---------|
| `/home/paper/paper.md` | Research paper in markdown form |
| `/home/paper/paper.pdf` | Original paper PDF |
| `/home/paper/rubric.json` | Optional rubric or scope hints |
| `/home/paper/addendum.md` | Scope clarifications and constraints |
| `/home/paper/blacklist.txt` | Blocked resources that must not be used |
| `/home/submission/` | Output git repository |
| `/home/submission/reproduce.sh` | Reproduction entry point |
| `/home/agent/paper_analysis/` | Structured paper analysis: `summary.md`, `structure.md`, `algorithm.md`, `experiments.md`, `baseline.md` |
| `/home/agent/prioritized_tasks.md` | Ranked implementation plan |
| `/home/agent/plan.md` | Implementation plan and auxiliary planning output |
| `/home/agent/impl_log.md` | Implementation changelog |
| `/home/agent/exp_log.md` | Experiment log |
| `/home/agent/experiments/` | Experiment command logs |
| `/home/logs/subagent_logs/` | JSONL traces for the main agent and subagents |
"""


IMPLEMENTATION_WORKSPACE_REFERENCE = """## Workspace Reference

| Path | Content | Availability |
|------|---------|--------------|
| `/home/paper/paper.md` | Research paper in markdown form | Always |
| `/home/paper/paper.pdf` | Original paper PDF | Always |
| `/home/paper/rubric.json` | Optional rubric or scope hints | Always |
| `/home/paper/addendum.md` | Scope clarifications and constraints | Always |
| `/home/paper/blacklist.txt` | Blocked resources that must not be used | Always |
| `/home/submission/` | Code repository and working tree | Always |
| `/home/submission/reproduce.sh` | Reproduction entry point | May not exist yet |
| `/home/agent/paper_analysis/` | Detailed paper analysis | After `read_paper` |
| `/home/agent/prioritized_tasks.md` | Priority-ranked task list | After `prioritize_tasks` |
| `/home/agent/plan.md` | Implementation plan | After planning work runs |
| `/home/agent/impl_log.md` | Implementation changelog written by you | Always |
| `/home/agent/exp_log.md` | Experiment log written by experiment runs | Always |
| `/home/agent/experiments/` | Experiment command logs | After `exec_command` runs |
"""


EXPERIMENT_WORKSPACE_REFERENCE = """## Workspace Reference

| Path | Content | Availability |
|------|---------|--------------|
| `/home/paper/paper.md` | Research paper in markdown form | Always |
| `/home/paper/paper.pdf` | Original paper PDF | Always |
| `/home/paper/rubric.json` | Optional rubric or scope hints | Always |
| `/home/paper/addendum.md` | Scope clarifications and constraints | Always |
| `/home/paper/blacklist.txt` | Blocked resources that must not be used | Always |
| `/home/submission/` | Code repository and working tree | Always |
| `/home/submission/reproduce.sh` | Reproduction entry point | May not exist yet |
| `/home/agent/paper_analysis/` | Detailed paper analysis | After `read_paper` |
| `/home/agent/prioritized_tasks.md` | Priority-ranked task list | After `prioritize_tasks` |
| `/home/agent/plan.md` | Implementation plan | After planning work runs |
| `/home/agent/impl_log.md` | Implementation changelog | After implementation work |
| `/home/agent/exp_log.md` | Experiment log written by you | Always |
| `/home/agent/experiments/` | Experiment command logs | Created by `exec_command` |
"""


SUBAGENT_WORKSPACE_REFERENCE = """## Workspace Reference

| Path | Content | Availability |
|------|---------|--------------|
| `/home/paper/paper.md` | Research paper in markdown form | Always |
| `/home/paper/paper.pdf` | Original paper PDF | Always |
| `/home/paper/rubric.json` | Optional rubric or scope hints | Always |
| `/home/paper/addendum.md` | Scope clarifications and constraints | Always |
| `/home/paper/blacklist.txt` | Blocked resources that must not be used | Always |
| `/home/submission/` | Code repository and working tree | Always |
| `/home/submission/reproduce.sh` | Reproduction entry point | May not exist yet |
| `/home/agent/paper_analysis/` | Detailed paper analysis | After `read_paper` |
| `/home/agent/prioritized_tasks.md` | Priority-ranked task list | After `prioritize_tasks` |
| `/home/agent/plan.md` | Implementation plan | After planning work runs |
| `/home/agent/impl_log.md` | Implementation changelog | After implementation work |
| `/home/agent/exp_log.md` | Experiment log | After experiment work |
"""


def workspace_paths(job_paths: JobPaths) -> dict[str, Path]:
    return {
        "paper": job_paths.workspace_dir / "paper",
        "submission": job_paths.workspace_dir / "submission",
        "agent": job_paths.workspace_dir / "agent",
        "logs": job_paths.logs_dir,
        "analysis": job_paths.workspace_dir / "agent" / "paper_analysis",
        "subagent_logs": job_paths.logs_dir / "subagent_logs",
    }

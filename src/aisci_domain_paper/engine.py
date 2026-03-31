from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openai import BadRequestError

from aisci_agent_runtime.llm_client import ContextLengthError, ContentPolicyError, LLMClient
from aisci_agent_runtime.log_utils import (
    log_messages_to_file,
    log_model_response_event,
    log_tool_result_event,
)
from aisci_agent_runtime.subagents.base import (
    SubagentConfig,
    SubagentOutput,
    SubagentStatus,
    fix_message_consistency,
    prune_messages,
    prune_messages_individual,
)
from aisci_agent_runtime.tools.base import SubagentCompleteSignal
from aisci_domain_paper.constants import MAIN_AGENT_WORKSPACE_REFERENCE
from aisci_domain_paper.prompts import MAIN_AGENT_SYSTEM_PROMPT
from aisci_domain_paper.runtime import (
    build_bootstrap_reproduce_script,
    ensure_submission_repo,
    extract_pdf_excerpt,
    list_files,
)
from aisci_domain_paper.subagents import subagent_class_for_kind
from aisci_domain_paper.tools import build_main_tools


@dataclass(frozen=True)
class PaperRuntimeConfig:
    job_id: str
    objective: str
    llm_profile_name: str
    time_limit_seconds: int = 24 * 3600
    max_steps: int = 80
    reminder_freq: int = 5
    enable_online_research: bool = True
    enable_github_research: bool = True
    bootstrap_only: bool = False


class EmbeddedPaperEngine:
    def __init__(
        self,
        *,
        config: PaperRuntimeConfig,
        shell,
        llm: LLMClient | None,
        paper_dir: Path,
        submission_dir: Path,
        agent_dir: Path,
        logs_dir: Path,
        trace,
    ) -> None:
        self.config = config
        self.shell = shell
        self.llm = llm
        self.paper_dir = paper_dir
        self.submission_dir = submission_dir
        self.agent_dir = agent_dir
        self.logs_dir = logs_dir
        self.trace = trace
        self.analysis_dir = agent_dir / "paper_analysis"
        self.subagent_logs_dir = logs_dir / "subagent_logs"
        self.agent_log_path = logs_dir / "agent.log"
        self.conversation_path = logs_dir / "conversation.jsonl"
        self.capability_path = agent_dir / "capabilities.json"
        self.self_check_path = agent_dir / "final_self_check.md"
        self.self_check_json_path = agent_dir / "final_self_check.json"
        self.prompt_path = agent_dir / "paper_main_prompt.md"
        self.state_path = logs_dir / "paper_session_state.json"
        self.impl_log_path = agent_dir / "impl_log.md"
        self.exp_log_path = agent_dir / "exp_log.md"
        self.plan_path = agent_dir / "plan.md"
        self.prioritized_path = agent_dir / "prioritized_tasks.md"
        self.reproduce_path = submission_dir / "reproduce.sh"
        self.session_id = time.strftime("%Y%m%d_%H%M%S")
        self._subagent_counts: dict[str, int] = {}
        self._impl_runs = 0
        self._exp_runs = 0
        self._validate_runs = 0

    def run(self) -> str:
        self._ensure_workspace()
        self._write_capability_report()
        if self.config.bootstrap_only or self.llm is None:
            return self.run_bootstrap_workflow()
        return self.run_main_loop()

    def run_bootstrap_workflow(self) -> str:
        self.trace.event(
            "agent_step",
            "Paper engine started in bootstrap mode.",
            phase="analyze",
            payload={"llm_enabled": False},
        )
        read_summary = self.read_paper(refresh=True)
        priority_summary = self.prioritize_tasks(refresh=True)
        impl_summary = self._fallback_subagent(
            "implementation",
            self._default_implementation_task(),
            "Bootstrap mode creates the product workspace and reproduce entrypoint.",
            self._allocate_subagent_session("implementation"),
        )
        exp_summary = self._fallback_subagent(
            "experiment",
            "Run a quick bootstrap check for reproduce.sh and essential files.",
            "",
            self._allocate_subagent_session("experiment"),
        )
        validation_summary = self.run_clean_validation(refresh=True)
        summary = "\n".join(
            [
                "Bootstrap paper workflow completed.",
                read_summary,
                priority_summary,
                impl_summary,
                exp_summary,
                validation_summary,
            ]
        )
        self._write_state(summary=summary, mode="bootstrap")
        self.trace.event(
            "agent_step",
            "Paper engine completed bootstrap mode.",
            phase="finalize",
            payload={"self_check": str(self.self_check_path)},
        )
        return summary

    def run_main_loop(self) -> str:
        self.trace.event(
            "agent_step",
            "Paper engine main loop started.",
            phase="analyze",
            payload={"llm_enabled": True, "llm_profile": self.config.llm_profile_name},
        )
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": MAIN_AGENT_SYSTEM_PROMPT},
            {"role": "user", "content": self._initial_user_prompt()},
        ]
        log_messages_to_file(messages, str(self.agent_log_path))

        tools = build_main_tools(self)
        tool_map = {tool.name(): tool for tool in tools}
        tool_schemas = [tool.get_tool_schema() for tool in tools]
        start_time = time.time()

        for step in range(1, self.config.max_steps + 1):
            elapsed = max(
                time.time() - start_time - (self.llm.total_retry_time if self.llm else 0.0),
                0.0,
            )
            if elapsed >= self.config.time_limit_seconds:
                break

            if step > 1 and step % self.config.reminder_freq == 0:
                messages.append({"role": "user", "content": self._build_reminder(step, elapsed)})

            try:
                assert self.llm is not None
                response = self.llm.chat(messages, tools=tool_schemas)
            except ContentPolicyError as exc:
                summary = f"Paper run stopped by content policy: {exc}"
                self._write_state(summary=summary, mode="content_policy_stop")
                return summary
            except ContextLengthError as exc:
                messages = self._reduce_messages(messages, exc)
                continue
            except BadRequestError:
                messages = fix_message_consistency(messages)
                continue

            log_model_response_event(
                str(self.conversation_path),
                self.session_id,
                step,
                len(messages),
                response.text_content,
                [{"id": call.call_id, "name": call.name, "arguments": call.arguments} for call in response.tool_calls],
                response.usage,
                response.reasoning_content,
            )

            assistant_message: dict[str, Any] = {"role": "assistant", "content": response.text_content}
            if response.tool_calls:
                assistant_message["tool_calls"] = [
                    {
                        "id": call.call_id,
                        "type": "function",
                        "function": {
                            "name": call.name,
                            "arguments": json.dumps(call.arguments),
                        },
                    }
                    for call in response.tool_calls
                ]
            messages.append(assistant_message)
            log_messages_to_file(messages, str(self.agent_log_path))

            if not response.tool_calls:
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Continue the paper workflow. Use tools to make progress, and call submit only after "
                            "paper reading, prioritization, implementation, experiments, and final self-check are done."
                        ),
                    }
                )
                continue

            for call in response.tool_calls:
                tool = tool_map.get(call.name)
                if tool is None:
                    tool_result = f"Unknown tool: {call.name}"
                else:
                    try:
                        tool_result = str(tool.execute(self.shell, **call.arguments))
                    except SubagentCompleteSignal as signal:
                        log_tool_result_event(
                            str(self.conversation_path),
                            self.session_id,
                            step,
                            call.name,
                            call.call_id,
                            signal.content,
                        )
                        self._write_state(summary=signal.content, mode="llm_finish")
                        self.trace.event(
                            "agent_step",
                            "Paper engine finished via submit.",
                            phase="finalize",
                            payload={"summary": signal.content},
                        )
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": call.call_id,
                                "content": signal.content,
                            }
                        )
                        log_messages_to_file(messages, str(self.agent_log_path))
                        return signal.content
                    except Exception as exc:  # noqa: BLE001
                        tool_result = f"Tool {call.name} failed: {exc}"

                log_tool_result_event(
                    str(self.conversation_path),
                    self.session_id,
                    step,
                    call.name,
                    call.call_id,
                    tool_result,
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.call_id,
                        "content": tool_result,
                    }
                )
                log_messages_to_file(messages, str(self.agent_log_path))

        final_summary = self._auto_finalize_summary()
        self._write_state(summary=final_summary, mode="auto_finalize")
        self.trace.event(
            "agent_step",
            "Paper engine auto-finalized after the main loop ended.",
            phase="finalize",
            payload={"summary": final_summary},
        )
        return final_summary

    def read_paper(self, refresh: bool = False) -> str:
        self._ensure_workspace()
        summary_path = self.analysis_dir / "summary.md"
        structure_path = self.analysis_dir / "structure.md"
        algorithm_path = self.analysis_dir / "algorithm.md"
        experiments_path = self.analysis_dir / "experiments.md"
        baseline_path = self.analysis_dir / "baseline.md"
        if summary_path.exists() and not refresh:
            return "Paper analysis already exists at /home/agent/paper_analysis/summary.md."

        paper_md = self._read_text(self.paper_dir / "paper.md", limit=18_000)
        pdf_excerpt = extract_pdf_excerpt(self.paper_dir / "paper.pdf", max_pages=4, max_chars=2000)
        headings = self._extract_headings(paper_md)
        files = list_files(self.paper_dir)
        summary_lines = [
            "# Paper Analysis Summary",
            "",
            f"- Objective: {self.config.objective}",
            f"- Paper Files: {len(files)}",
            f"- Online Research: {self._capability_value('online_research')}",
            f"- GitHub Research: {self._capability_value('github_research')}",
            "",
            "## Workspace Reference",
            "",
            MAIN_AGENT_WORKSPACE_REFERENCE.strip(),
            "",
            "## Available Inputs",
            "",
            *(files or ["- No staged paper inputs detected."]),
            "",
            "## Executive Summary",
            "",
            self._summarize_paper_text(paper_md, pdf_excerpt),
            "",
            "## Navigation",
            "",
            "| Section | File | Purpose |",
            "|---|---|---|",
            "| Summary | `/home/agent/paper_analysis/summary.md` | fast overview |",
            "| Structure | `/home/agent/paper_analysis/structure.md` | section map and source inventory |",
            "| Algorithm | `/home/agent/paper_analysis/algorithm.md` | implementation-facing method notes |",
            "| Experiments | `/home/agent/paper_analysis/experiments.md` | experiment and validation notes |",
            "| Baseline | `/home/agent/paper_analysis/baseline.md` | baseline and comparison targets |",
            "",
        ]
        structure_lines = [
            "# Paper Structure",
            "",
            f"- Staged file count: {len(files)}",
            "",
            "## Discovered Files",
            "",
            *(files or ["- No staged files."]),
            "",
            "## Heading Index",
            "",
            *(headings or ["- No markdown headings found."]),
        ]
        algorithm_lines = [
            "# Algorithm Notes",
            "",
            "- Keep `reproduce.sh` runnable early.",
            "- Translate the core method into a narrow implementation surface before deep optimization.",
            "- Prefer one commit per meaningful milestone in `/home/submission`.",
            "",
            "## Method Excerpt",
            "",
            self._pick_excerpt(paper_md or pdf_excerpt, pattern=r"(?i)(method|approach|algorithm|model)", default_label="paper"),
        ]
        experiments_lines = [
            "# Experiment Notes",
            "",
            "- Reproduce main-text experiments first, then appendix-only items if time allows.",
            "- Track exact commands, outputs, and failure diagnoses in `/home/agent/exp_log.md`.",
            "- Final self-check should re-run `bash reproduce.sh` from `/home/submission`.",
            "",
            "## Experiment Excerpt",
            "",
            self._pick_excerpt(paper_md or pdf_excerpt, pattern=r"(?i)(experiment|result|table|dataset|evaluation)", default_label="paper"),
        ]
        baseline_lines = [
            "# Baseline Notes",
            "",
            "- Treat methods that appear in main tables as first-class reproduction targets.",
            "- Create one implementation task per baseline or model variant that materially changes the pipeline.",
            "",
            "## Baseline Excerpt",
            "",
            self._pick_excerpt(paper_md or pdf_excerpt, pattern=r"(?i)(baseline|comparison|related work)", default_label="paper"),
        ]

        summary_path.write_text("\n".join(summary_lines).rstrip() + "\n", encoding="utf-8")
        structure_path.write_text("\n".join(structure_lines).rstrip() + "\n", encoding="utf-8")
        algorithm_path.write_text("\n".join(algorithm_lines).rstrip() + "\n", encoding="utf-8")
        experiments_path.write_text("\n".join(experiments_lines).rstrip() + "\n", encoding="utf-8")
        baseline_path.write_text("\n".join(baseline_lines).rstrip() + "\n", encoding="utf-8")
        self.trace.event(
            "subagent_finish",
            "read_paper wrote /home/agent/paper_analysis/*",
            phase="analyze",
            payload={
                "files": [str(summary_path), str(structure_path), str(algorithm_path), str(experiments_path), str(baseline_path)]
            },
        )
        return "Paper analysis complete. See /home/agent/paper_analysis/summary.md and related files."

    def prioritize_tasks(self, refresh: bool = False) -> str:
        self._ensure_workspace()
        if self.prioritized_path.exists() and self.plan_path.exists() and not refresh:
            return "Prioritized plan already exists at /home/agent/prioritized_tasks.md."
        self.read_paper(refresh=False)
        rubric_lines = self._read_text(self.paper_dir / "rubric.json", limit=10_000)
        expected_outputs = self._read_text(self.paper_dir / "addendum.md", limit=4_000)
        prioritized_lines = [
            "# Prioritized Implementation Plan",
            "",
            "## Executive Summary",
            "",
            f"- Objective: {self.config.objective}",
            "- Strategy: breadth-first across core method, main experiments, and reproducibility.",
            "- Constraint: keep `/home/submission/reproduce.sh` working from the earliest meaningful commit.",
            "",
            "## P0-Critical",
            "",
            "- Create or stabilize `/home/submission/reproduce.sh` and the minimal runnable project layout.",
            "- Implement the paper's primary method and the minimum code path needed to execute it.",
            "- Cover the main reported experiments or outputs that are central to the paper's claim.",
            "- Ensure `/home/agent/impl_log.md` and `/home/agent/exp_log.md` are updated after each major change.",
            "",
            "## P1-Important",
            "",
            "- Add comparison baselines, ablations, or secondary settings that materially support the paper's main result.",
            "- Harden configuration, scripts, and dependency setup so the final self-check passes reliably.",
            "",
            "## P2-Valuable",
            "",
            "- Add appendix-only experiments and convenience analysis once the core flow is stable.",
            "- Improve result inspection, tables, or export outputs for user-facing clarity.",
            "",
            "## Execution Order",
            "",
            "1. Read the structured paper analysis files.",
            "2. Stabilize `reproduce.sh` and repository skeleton.",
            "3. Implement the core method and one end-to-end happy path.",
            "4. Run targeted experiments and record outcomes.",
            "5. Run `clean_reproduce_validation` before finishing.",
            "",
            "## Notes from Staged Constraints",
            "",
            "### Rubric",
            "",
            rubric_lines or "No rubric.json staged.",
            "",
            "### Addendum",
            "",
            expected_outputs or "No addendum.md staged.",
        ]
        plan_lines = [
            "# Plan",
            "",
            "## Phase 1",
            "- Complete paper reading and create the prioritized plan.",
            "- Make the submission repository runnable with a bootstrap `reproduce.sh`.",
            "",
            "## Phase 2",
            "- Implement the paper's core behavior in `/home/submission`.",
            "- Keep implementation notes in `/home/agent/impl_log.md`.",
            "",
            "## Phase 3",
            "- Run experiments with focused diagnostics.",
            "- Capture commands, outputs, and failures in `/home/agent/exp_log.md`.",
            "",
            "## Phase 4",
            "- Run the final self-check and fix any reproducibility gaps.",
            "- Finish only after the project is coherent for a product user.",
        ]
        self.prioritized_path.write_text("\n".join(prioritized_lines).rstrip() + "\n", encoding="utf-8")
        self.plan_path.write_text("\n".join(plan_lines).rstrip() + "\n", encoding="utf-8")
        self.trace.event(
            "subagent_finish",
            "prioritize_tasks wrote /home/agent/prioritized_tasks.md",
            phase="prioritize",
            payload={"plan": str(self.plan_path), "priorities": str(self.prioritized_path)},
        )
        return "Prioritized plan complete. See /home/agent/prioritized_tasks.md and /home/agent/plan.md."

    def run_implementation(
        self,
        *,
        task: str,
        mode: str = "full",
        context: str = "",
        max_steps: int | None = None,
        time_budget: int | None = None,
    ) -> str:
        combined_context = f"mode={mode}\n{context}".strip()
        return self.run_named_subagent(
            subagent_type="implementation",
            objective=task,
            context=combined_context,
            max_steps=max_steps,
            time_limit=time_budget,
        )

    def run_experiment(
        self,
        *,
        task: str,
        mode: str = "full",
        context: str = "",
        max_steps: int | None = None,
        time_budget: int | None = None,
    ) -> str:
        combined_context = f"mode={mode}\n{context}".strip()
        return self.run_named_subagent(
            subagent_type="experiment",
            objective=task,
            context=combined_context,
            max_steps=max_steps,
            time_limit=time_budget,
        )

    def run_named_subagent(
        self,
        *,
        subagent_type: str,
        objective: str,
        context: str = "",
        max_steps: int | None = None,
        time_limit: int | None = None,
    ) -> str:
        session_dir = self._allocate_subagent_session(subagent_type)
        self.trace.event(
            "subagent_start",
            f"{subagent_type} subagent started.",
            phase=self._phase_for_subagent(subagent_type),
            payload={"objective": objective, "session_dir": str(session_dir)},
        )
        if self.llm is None:
            result_text = self._fallback_subagent(subagent_type, objective, context, session_dir)
            self.trace.event(
                "subagent_finish",
                f"{subagent_type} subagent completed in fallback mode.",
                phase=self._phase_for_subagent(subagent_type),
                payload={"session_dir": str(session_dir), "fallback": True},
            )
            return result_text

        subagent_cls = subagent_class_for_kind(subagent_type)
        config = SubagentConfig(
            max_steps=max_steps or self._default_subagent_steps(subagent_type),
            time_limit=time_limit or self._default_subagent_time_limit(subagent_type),
            reminder_freq=max(5, self.config.reminder_freq),
            log_dir=str(session_dir),
        )
        subagent = subagent_cls(
            self,
            self.shell,
            self.llm,
            config,
            objective=objective,
            context=context,
        )
        result = subagent.run(context=subagent.build_context())
        self._record_subagent_counters(subagent_type, result)
        self.trace.event(
            "subagent_finish",
            f"{subagent_type} subagent finished with status={result.status.value}.",
            phase=self._phase_for_subagent(subagent_type),
            payload={
                "objective": objective,
                "status": result.status.value,
                "session_dir": str(session_dir),
                "log_path": result.log_path,
            },
        )
        return self._format_subagent_result(subagent_type, result)

    def run_clean_validation(self, refresh: bool = False) -> str:
        self._ensure_workspace()
        if self.self_check_path.exists() and not refresh:
            return "Final self-check report already exists at /home/agent/final_self_check.md."

        self._ensure_reproduce_script()
        self._git_snapshot()
        hardcoded = self.shell.send_command(
            "grep -R -n '/home/submission' /home/submission 2>/dev/null || true",
            timeout=20,
        ).output.strip()
        reproduce_result = self.shell.send_command(
            "cd /home/submission && bash reproduce.sh",
            timeout=min(1200, max(300, self.config.time_limit_seconds // 4)),
        )
        status = "passed" if reproduce_result.exit_code == 0 else "failed"
        report = {
            "status": status,
            "hardcoded_path_hits": hardcoded.splitlines() if hardcoded else [],
            "reproduce_exit_code": reproduce_result.exit_code,
            "reproduce_output_preview": self._truncate_text(reproduce_result.output, 8_000),
        }
        report_lines = [
            "# Final Self-Check",
            "",
            f"- Status: **{status}**",
            f"- Reproduce Exit Code: `{reproduce_result.exit_code}`",
            f"- Hardcoded `/home/submission` Hits: {len(report['hardcoded_path_hits'])}",
            "",
            "## Checklist",
            "",
            "- `reproduce.sh` exists and is executable.",
            "- The submission repository has a git snapshot.",
            "- The self-check re-runs `bash reproduce.sh` from `/home/submission`.",
            "",
            "## Hardcoded Path Scan",
            "",
            hardcoded or "No hardcoded `/home/submission` matches found.",
            "",
            "## Reproduce Output Preview",
            "",
            "```text",
            report["reproduce_output_preview"],
            "```",
        ]
        self.self_check_path.write_text("\n".join(report_lines).rstrip() + "\n", encoding="utf-8")
        self.self_check_json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        self._append_log(
            self.exp_log_path,
            "Self-check",
            f"Status: {status}\n\nExit code: {reproduce_result.exit_code}\n\n{report['reproduce_output_preview']}",
        )
        self._validate_runs += 1
        self.trace.event(
            "validation_finished",
            f"Final self-check finished with status={status}.",
            phase="validate",
            payload=report,
        )
        return f"Final self-check {status}. Report: /home/agent/final_self_check.md"

    def collect_artifacts(self) -> list[Path]:
        candidates = [
            self.analysis_dir / "summary.md",
            self.analysis_dir / "structure.md",
            self.analysis_dir / "algorithm.md",
            self.analysis_dir / "experiments.md",
            self.analysis_dir / "baseline.md",
            self.prioritized_path,
            self.plan_path,
            self.impl_log_path,
            self.exp_log_path,
            self.reproduce_path,
            self.capability_path,
            self.prompt_path,
            self.self_check_path,
            self.self_check_json_path,
            self.agent_log_path,
            self.conversation_path,
            self.state_path,
        ]
        return [path for path in candidates if path.exists()]

    def _ensure_workspace(self) -> None:
        for path in (
            self.paper_dir,
            self.submission_dir,
            self.agent_dir,
            self.logs_dir,
            self.analysis_dir,
            self.subagent_logs_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)
        ensure_submission_repo(self.submission_dir)
        self.prompt_path.write_text(MAIN_AGENT_SYSTEM_PROMPT, encoding="utf-8")
        if not self.impl_log_path.exists():
            self.impl_log_path.write_text("# Implementation Log\n\n", encoding="utf-8")
        if not self.exp_log_path.exists():
            self.exp_log_path.write_text("# Experiment Log\n\n", encoding="utf-8")
        self._ensure_reproduce_script()
        if not (self.submission_dir / "README.md").exists():
            (self.submission_dir / "README.md").write_text(
                "# Paper Reproduction Workspace\n\nManaged by AiScientist paper mode.\n",
                encoding="utf-8",
            )

    def _ensure_reproduce_script(self) -> None:
        if not self.reproduce_path.exists():
            self.reproduce_path.write_text(
                build_bootstrap_reproduce_script(
                    self.config.objective,
                    extra_notes="Replace this scaffold with the paper-specific reproduction workflow.",
                ),
                encoding="utf-8",
            )
        try:
            self.reproduce_path.chmod(0o755)
        except OSError:
            pass

    def _write_capability_report(self) -> None:
        self.capability_path.write_text(json.dumps(self._capabilities(), indent=2), encoding="utf-8")

    def _capabilities(self) -> dict[str, Any]:
        llm_enabled = self.llm is not None and not self.config.bootstrap_only
        online_available = llm_enabled and self.config.enable_online_research
        github_available = False
        return {
            "llm_enabled": llm_enabled,
            "online_research": {
                "requested": self.config.enable_online_research,
                "available": online_available,
                "mode": "model_native_web_search" if online_available else "disabled_or_unavailable",
            },
            "github_research": {
                "requested": self.config.enable_github_research,
                "available": github_available,
                "mode": "not_implemented",
            },
        }

    def _capability_value(self, key: str) -> str:
        data = self._capabilities().get(key, {})
        if isinstance(data, dict):
            return "available" if data.get("available") else "disabled"
        return str(data)

    def _initial_user_prompt(self) -> str:
        capabilities = self._capabilities()
        return (
            f"You are running a paper reproduction job for objective: {self.config.objective}\n\n"
            "Recommended workflow:\n"
            "1. Call `read_paper` first.\n"
            "2. Call `prioritize_tasks` second.\n"
            "3. Use `implement` for the main code path.\n"
            "4. Use `run_experiment` to validate progress.\n"
            "5. Call `clean_reproduce_validation` before you finish.\n"
            "6. Call `submit` only after the self-check is done.\n\n"
            "Keep `/home/submission/reproduce.sh` runnable early.\n"
            "Treat `/home/submission` as the implementation repo and `/home/agent` as durable notes.\n\n"
            f"Capability status:\n{json.dumps(capabilities, indent=2)}\n\n"
            f"{MAIN_AGENT_WORKSPACE_REFERENCE.strip()}"
        )

    def _build_reminder(self, step: int, elapsed: float) -> str:
        remaining = max(0, self.config.time_limit_seconds - elapsed)
        parts = [
            f"Reminder: step {step}/{self.config.max_steps}. Elapsed {int(elapsed)}s, remaining {int(remaining)}s.",
            f"Artifacts ready: read_paper={'yes' if (self.analysis_dir / 'summary.md').exists() else 'no'}, "
            f"priorities={'yes' if self.prioritized_path.exists() else 'no'}, "
            f"impl_runs={self._impl_runs}, exp_runs={self._exp_runs}, self_checks={self._validate_runs}.",
            "Do not finish before the final self-check passes or you have a concrete failure diagnosis.",
        ]
        return "\n".join(parts)

    def _default_implementation_task(self) -> str:
        return (
            "Implement the core paper workflow in /home/submission, keep reproduce.sh runnable, "
            "and update /home/agent/impl_log.md with concrete progress."
        )

    def _allocate_subagent_session(self, subagent_type: str) -> Path:
        count = self._subagent_counts.get(subagent_type, 0) + 1
        self._subagent_counts[subagent_type] = count
        session_dir = self.subagent_logs_dir / f"{subagent_type}_{count:03d}_{time.strftime('%Y%m%d_%H%M%S')}"
        session_dir.mkdir(parents=True, exist_ok=True)
        return session_dir

    def _fallback_subagent(self, subagent_type: str, objective: str, context: str, session_dir: Path) -> str:
        session_note = session_dir / "fallback_summary.md"
        if subagent_type in {"reader", "paper_reader"}:
            result = self.read_paper(refresh=True)
        elif subagent_type == "prioritization":
            result = self.prioritize_tasks(refresh=True)
        elif subagent_type == "implementation":
            self._append_log(self.impl_log_path, "Bootstrap implementation", f"{objective}\n\n{context}".strip())
            if not (self.submission_dir / "src").exists():
                (self.submission_dir / "src").mkdir(parents=True, exist_ok=True)
            main_py = self.submission_dir / "src" / "paper_repro.py"
            if not main_py.exists():
                main_py.write_text(
                    "from pathlib import Path\n\n"
                    "def main() -> None:\n"
                    "    root = Path(__file__).resolve().parents[1]\n"
                    "    print('paper reproduction workspace ready:', root)\n\n"
                    "if __name__ == '__main__':\n"
                    "    main()\n",
                    encoding="utf-8",
                )
            result = "Bootstrap implementation completed. Created src/paper_repro.py and updated impl_log.md."
            self._impl_runs += 1
        elif subagent_type == "experiment":
            output = self.shell.send_command("cd /home/submission && bash reproduce.sh", timeout=300)
            self._append_log(
                self.exp_log_path,
                "Bootstrap experiment",
                f"{objective}\n\nExit code: {output.exit_code}\n\n{self._truncate_text(output.output, 4_000)}",
            )
            result = f"Bootstrap experiment finished with exit_code={output.exit_code}."
            self._exp_runs += 1
        elif subagent_type == "validation":
            result = self.run_clean_validation(refresh=True)
        elif subagent_type == "plan":
            self.plan_path.write_text(f"# Plan\n\n- {objective}\n\n{context}\n", encoding="utf-8")
            result = "Bootstrap planning note written to /home/agent/plan.md."
        else:
            result = f"Fallback subagent note: {objective}\n\n{context}".strip()
        session_note.write_text(result + "\n", encoding="utf-8")
        return result

    def _record_subagent_counters(self, subagent_type: str, result: SubagentOutput) -> None:
        if result.status != SubagentStatus.COMPLETED:
            return
        if subagent_type == "implementation":
            self._impl_runs += 1
        elif subagent_type == "experiment":
            self._exp_runs += 1
        elif subagent_type == "validation":
            self._validate_runs += 1

    def _format_subagent_result(self, subagent_type: str, result: SubagentOutput) -> str:
        return (
            f"[{subagent_type}:{result.status.value}] "
            f"steps={result.num_steps} runtime={result.runtime_seconds:.1f}s\n\n"
            f"{result.content}"
        )

    def _phase_for_subagent(self, subagent_type: str) -> str:
        if subagent_type in {"reader", "paper_reader"}:
            return "analyze"
        if subagent_type == "prioritization":
            return "prioritize"
        if subagent_type == "experiment":
            return "validate"
        if subagent_type == "validation":
            return "validate"
        return "implement"

    def _default_subagent_steps(self, subagent_type: str) -> int:
        return {
            "reader": 20,
            "paper_reader": 20,
            "prioritization": 18,
            "implementation": 40,
            "experiment": 24,
            "validation": 16,
            "explore": 20,
            "plan": 20,
            "general": 20,
            "generic": 20,
            "env_setup": 20,
            "resource_download": 20,
        }.get(subagent_type, 20)

    def _default_subagent_time_limit(self, subagent_type: str) -> int:
        return {
            "reader": min(1800, self.config.time_limit_seconds),
            "paper_reader": min(1800, self.config.time_limit_seconds),
            "prioritization": min(1800, self.config.time_limit_seconds),
            "implementation": min(4 * 3600, self.config.time_limit_seconds),
            "experiment": min(2 * 3600, self.config.time_limit_seconds),
            "validation": min(1800, self.config.time_limit_seconds),
            "explore": min(1800, self.config.time_limit_seconds),
            "plan": min(1800, self.config.time_limit_seconds),
            "general": min(1800, self.config.time_limit_seconds),
            "generic": min(1800, self.config.time_limit_seconds),
            "env_setup": min(1800, self.config.time_limit_seconds),
            "resource_download": min(1800, self.config.time_limit_seconds),
        }.get(subagent_type, min(1800, self.config.time_limit_seconds))

    def _reduce_messages(self, messages: list[dict[str, Any]], exc: ContextLengthError) -> list[dict[str, Any]]:
        if exc.prune_individual and self.llm is not None:
            messages = prune_messages_individual(messages, max_tokens_per_message=self.llm.config.context_window)
        return prune_messages(messages)

    def _auto_finalize_summary(self) -> str:
        if not self.self_check_path.exists():
            self.run_clean_validation(refresh=True)
        return (
            "Paper main loop ended without submit. "
            "A final self-check report was generated automatically at /home/agent/final_self_check.md."
        )

    def _read_text(self, path: Path, *, limit: int = 8_000) -> str:
        if not path.exists():
            return ""
        return path.read_text(encoding="utf-8", errors="replace")[:limit]

    def _summarize_paper_text(self, paper_md: str, pdf_excerpt: str) -> str:
        text = paper_md or pdf_excerpt or "No paper content available."
        cleaned = re.sub(r"\s+", " ", text).strip()
        if not cleaned:
            return "No paper content available."
        sentences = re.split(r"(?<=[.!?])\s+", cleaned)
        return " ".join(sentences[:8])[:2_500]

    def _extract_headings(self, paper_md: str) -> list[str]:
        headings = []
        for line in paper_md.splitlines():
            if line.lstrip().startswith("#"):
                headings.append(f"- {line.strip()}")
            if len(headings) >= 80:
                break
        return headings

    def _pick_excerpt(self, text: str, *, pattern: str, default_label: str) -> str:
        if not text:
            return f"No {default_label} excerpt available."
        match = re.search(pattern, text)
        if match:
            start = max(0, match.start() - 600)
            end = min(len(text), match.end() + 1200)
            return text[start:end].strip()
        return text[:1800].strip()

    def _append_log(self, path: Path, title: str, body: str) -> None:
        existing = path.read_text(encoding="utf-8") if path.exists() else ""
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        entry = f"## {title} ({stamp})\n\n{body.strip()}\n\n"
        path.write_text(existing + entry, encoding="utf-8")

    def _git_snapshot(self) -> None:
        self.shell.send_command(
            "cd /home/submission && git add -A && git status --short > /tmp/paper_git_snapshot.txt || true",
            timeout=30,
        )

    def _write_state(self, *, summary: str, mode: str) -> None:
        state = {
            "job_id": self.config.job_id,
            "mode": mode,
            "summary": summary,
            "impl_runs": self._impl_runs,
            "exp_runs": self._exp_runs,
            "validate_runs": self._validate_runs,
            "ts": time.time(),
        }
        self.state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")

    def _truncate_text(self, text: str, limit: int) -> str:
        if len(text) <= limit:
            return text
        half = limit // 2
        return text[:half] + "\n...[truncated]...\n" + text[-half:]

from __future__ import annotations

import os
import re
from pathlib import Path

from aisci_agent_runtime.llm_client import LLMConfig, create_llm_client
from aisci_agent_runtime.llm_profiles import resolve_llm_profile
from aisci_agent_runtime.shell_interface import ShellInterface
from aisci_agent_runtime.trace import AgentTraceWriter
from aisci_domain_paper.engine import EmbeddedPaperEngine, PaperRuntimeConfig


def _bool_env(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def parse_duration_to_seconds(text: str) -> int:
    value = (text or "24h").strip().lower()
    total = 0
    for amount, unit in re.findall(r"(\d+)([smhd])", value):
        n = int(amount)
        total += {
            "s": n,
            "m": n * 60,
            "h": n * 3600,
            "d": n * 86400,
        }[unit]
    return total or 24 * 3600


def _has_llm_credentials() -> bool:
    return any(
        os.environ.get(key)
        for key in (
            "OPENAI_API_KEY",
            "AZURE_OPENAI_API_KEY",
        )
    )


def _build_llm(profile_name: str, *, enable_online_research: bool):
    if not _has_llm_credentials():
        raise RuntimeError(
            "Paper mode requires OPENAI_API_KEY or AZURE_OPENAI_API_KEY. "
            "No local fallback loop is available."
        )
    profile = resolve_llm_profile(profile_name)
    return create_llm_client(
        LLMConfig(
            model=profile.model,
            api_mode=profile.api_mode,
            reasoning_effort=profile.reasoning_effort,
            web_search=enable_online_research and profile.api_mode == "responses",
            max_tokens=profile.max_tokens,
            context_window=profile.context_window,
            use_phase=profile.use_phase,
        )
    )


def main() -> None:
    logs_dir = Path("/home/logs")
    paper_dir = Path("/home/paper")
    submission_dir = Path("/home/submission")
    agent_dir = Path("/home/agent")
    trace = AgentTraceWriter(logs_dir)
    if not _has_llm_credentials():
        raise RuntimeError(
            "Paper mode requires OPENAI_API_KEY or AZURE_OPENAI_API_KEY. "
            "No local fallback loop is available."
        )
    engine = EmbeddedPaperEngine(
        config=PaperRuntimeConfig(
            job_id=os.environ.get("AISCI_JOB_ID", "paper-job"),
            objective=os.environ.get("AISCI_OBJECTIVE", "paper reproduction job"),
            llm_profile_name=os.environ.get("AISCI_LLM_PROFILE", "gpt-5.4-responses"),
            time_limit_seconds=int(os.environ.get("TIME_LIMIT_SECS", str(24 * 3600))),
            max_steps=int(os.environ.get("AISCI_MAX_STEPS", "80")),
            reminder_freq=int(os.environ.get("AISCI_REMINDER_FREQ", "5")),
            enable_online_research=_bool_env("AISCI_ENABLE_ONLINE_RESEARCH", True),
        ),
        shell=ShellInterface("/home"),
        llm=_build_llm(
            os.environ.get("AISCI_LLM_PROFILE", "gpt-5.4-responses"),
            enable_online_research=_bool_env("AISCI_ENABLE_ONLINE_RESEARCH", True),
        ),
        paper_dir=paper_dir,
        submission_dir=submission_dir,
        agent_dir=agent_dir,
        logs_dir=logs_dir,
        trace=trace,
    )
    engine.run()


if __name__ == "__main__":
    main()

from __future__ import annotations

from aisci_domain_paper.prompts import (
    ENV_SETUP_SYSTEM_PROMPT,
    EXPERIMENT_SYSTEM_PROMPT,
    IMPLEMENTATION_SYSTEM_PROMPT,
    MAIN_AGENT_SYSTEM_PROMPT,
    PAPER_READER_SYSTEM_PROMPT,
    PRIORITIZATION_SYSTEM_PROMPT,
    RESOURCE_DOWNLOAD_SYSTEM_PROMPT,
)
from aisci_domain_paper.subagents import GeneralSubagent, subagent_class_for_kind


def test_main_prompt_contains_upstream_paper_workflow() -> None:
    assert "reproduce.sh" in MAIN_AGENT_SYSTEM_PROMPT
    assert "read_paper" in MAIN_AGENT_SYSTEM_PROMPT
    assert "prioritize_tasks" in MAIN_AGENT_SYSTEM_PROMPT
    assert "implement" in MAIN_AGENT_SYSTEM_PROMPT
    assert "run_experiment" in MAIN_AGENT_SYSTEM_PROMPT
    assert "clean_reproduce_validation" in MAIN_AGENT_SYSTEM_PROMPT
    assert "submit" in MAIN_AGENT_SYSTEM_PROMPT


def test_subagent_prompts_are_no_longer_placeholder_role_blurbs() -> None:
    assert "Implementation Specialist" in IMPLEMENTATION_SYSTEM_PROMPT
    assert "Experiment Agent" in EXPERIMENT_SYSTEM_PROMPT
    assert "Prioritization Strategist" in PRIORITIZATION_SYSTEM_PROMPT
    assert "Paper Reader Specialist" in PAPER_READER_SYSTEM_PROMPT
    assert "Environment Setup Specialist" in ENV_SETUP_SYSTEM_PROMPT
    assert "Resource Download Specialist" in RESOURCE_DOWNLOAD_SYSTEM_PROMPT


def test_generic_alias_maps_to_general_subagent() -> None:
    assert subagent_class_for_kind("generic") is GeneralSubagent

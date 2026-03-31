from __future__ import annotations

from aisci_domain_paper.prompts import (
    ENV_SETUP_SYSTEM_PROMPT,
    EXPERIMENT_SYSTEM_PROMPT,
    EXPLORE_SYSTEM_PROMPT,
    GENERAL_SYSTEM_PROMPT,
    IMPLEMENTATION_SYSTEM_PROMPT,
    MAIN_AGENT_SYSTEM_PROMPT,
    PLAN_SYSTEM_PROMPT,
    PRIORITIZATION_SYSTEM_PROMPT,
    RESOURCE_DOWNLOAD_SYSTEM_PROMPT,
    render_implementation_system_prompt,
    render_main_agent_system_prompt,
    render_plan_system_prompt,
    render_prioritization_system_prompt,
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
    assert "finish_run" not in MAIN_AGENT_SYSTEM_PROMPT


def test_subagent_prompts_are_no_longer_placeholder_role_blurbs() -> None:
    assert "Implementation Specialist" in IMPLEMENTATION_SYSTEM_PROMPT
    assert "Experiment Agent" in EXPERIMENT_SYSTEM_PROMPT
    assert "Prioritization Strategist" in PRIORITIZATION_SYSTEM_PROMPT
    assert "Environment Setup Specialist" in ENV_SETUP_SYSTEM_PROMPT
    assert "Resource Download Specialist" in RESOURCE_DOWNLOAD_SYSTEM_PROMPT


def test_live_prompt_renderers_hide_unavailable_optional_tools() -> None:
    capabilities = {
        "online_research": {"available": False},
        "linter": {"available": True},
    }
    main_prompt = render_main_agent_system_prompt(capabilities)
    implementation_prompt = render_implementation_system_prompt(capabilities)
    prioritization_prompt = render_prioritization_system_prompt(capabilities)
    plan_prompt = render_plan_system_prompt(capabilities)

    assert "web_search" not in main_prompt
    assert "link_summary" not in implementation_prompt
    assert "parse_rubric" in prioritization_prompt
    assert "write_priorities" in prioritization_prompt
    assert "bash" not in prioritization_prompt
    assert "python" not in prioritization_prompt
    assert "write_plan" in plan_prompt
    assert "edit_file" not in plan_prompt


def test_full_prompts_match_current_runtime_tool_contract() -> None:
    assert "web_search" in MAIN_AGENT_SYSTEM_PROMPT
    assert "link_summary" in MAIN_AGENT_SYSTEM_PROMPT
    assert "goal=" in MAIN_AGENT_SYSTEM_PROMPT
    assert "git_commit" not in GENERAL_SYSTEM_PROMPT
    assert "edit_file" not in GENERAL_SYSTEM_PROMPT
    assert "write_plan" in PLAN_SYSTEM_PROMPT
    assert "write_priorities" in PRIORITIZATION_SYSTEM_PROMPT
    assert "parse_rubric" in PRIORITIZATION_SYSTEM_PROMPT
    assert "bash" not in PRIORITIZATION_SYSTEM_PROMPT
    assert "python" not in PRIORITIZATION_SYSTEM_PROMPT
    assert "/home/agent/plan.md" not in PRIORITIZATION_SYSTEM_PROMPT
    assert "web_search" in EXPLORE_SYSTEM_PROMPT


def test_main_prompt_mentions_clean_validation_cadence_and_optional_impl_task() -> None:
    prompt = render_main_agent_system_prompt(
        {
            "online_research": {"available": True},
            "linter": {"available": True},
        }
    )
    impl_prompt = render_implementation_system_prompt(
        {
            "online_research": {"available": True},
            "linter": {"available": True},
        }
    )

    assert "Call `clean_reproduce_validation()` after the first major implementation round" in prompt
    assert "`task` is optional" in prompt
    assert "clean validation runs `git clean -fd`" in impl_prompt
    assert 'link_summary(url=' in impl_prompt


def test_generic_alias_maps_to_general_subagent() -> None:
    assert subagent_class_for_kind("generic") is GeneralSubagent

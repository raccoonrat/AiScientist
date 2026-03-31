from __future__ import annotations

from aisci_domain_paper.tools.basic_tool import build_implementation_tools, callback_tool


def build_implement_tool(engine):
    return callback_tool(
        "implement",
        "Delegate substantial coding work to the implementation subagent.",
        {
            "type": "object",
            "properties": {
                "task": {"type": "string", "description": "What to build or fix."},
                "mode": {"type": "string", "enum": ["full", "fix"], "description": "Implementation mode."},
                "context": {"type": "string", "description": "Feedback or constraints from previous work."},
                "time_budget": {"type": "integer", "description": "Time budget in seconds."},
                "max_steps": {"type": "integer", "description": "Optional step cap override."},
            },
            "required": ["task"],
            "additionalProperties": False,
        },
        lambda shell, task, mode="full", context="", time_budget=None, max_steps=None: engine.run_implementation(
            task=task,
            mode=mode,
            context=context,
            max_steps=max_steps,
            time_budget=time_budget,
        ),
    )


def build_spawn_env_setup_tool(engine):
    return callback_tool(
        "spawn_env_setup",
        "Delegate environment setup work to the environment setup subagent.",
        {
            "type": "object",
            "properties": {
                "task": {"type": "string"},
                "context": {"type": "string"},
                "time_budget": {"type": "integer"},
                "max_steps": {"type": "integer"},
            },
            "required": ["task"],
            "additionalProperties": False,
        },
        lambda shell, task, context="", time_budget=None, max_steps=None: engine.run_named_subagent(
            subagent_type="env_setup",
            objective=task,
            context=context,
            max_steps=max_steps,
            time_limit=time_budget,
        ),
    )


def build_spawn_resource_download_tool(engine):
    return callback_tool(
        "spawn_resource_download",
        "Delegate dataset or model download work to the resource download subagent.",
        {
            "type": "object",
            "properties": {
                "task": {"type": "string"},
                "context": {"type": "string"},
                "time_budget": {"type": "integer"},
                "max_steps": {"type": "integer"},
            },
            "required": ["task"],
            "additionalProperties": False,
        },
        lambda shell, task, context="", time_budget=None, max_steps=None: engine.run_named_subagent(
            subagent_type="resource_download",
            objective=task,
            context=context,
            max_steps=max_steps,
            time_limit=time_budget,
        ),
    )


__all__ = [
    "build_implement_tool",
    "build_implementation_tools",
    "build_spawn_env_setup_tool",
    "build_spawn_resource_download_tool",
]

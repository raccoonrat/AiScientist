from __future__ import annotations

from aisci_domain_paper.tools.basic_tool import build_experiment_tools, callback_tool


def build_run_experiment_tool(engine):
    return callback_tool(
        "run_experiment",
        "Delegate experiment execution and validation to the experiment subagent.",
        {
            "type": "object",
            "properties": {
                "task": {"type": "string", "description": "What to validate or run."},
                "mode": {"type": "string", "enum": ["full", "validate"], "description": "Experiment mode."},
                "context": {"type": "string", "description": "Diagnostics or expectations from previous work."},
                "time_budget": {"type": "integer", "description": "Time budget in seconds."},
                "max_steps": {"type": "integer", "description": "Optional step cap override."},
            },
            "required": ["task"],
            "additionalProperties": False,
        },
        lambda shell, task, mode="full", context="", time_budget=None, max_steps=None: engine.run_experiment(
            task=task,
            mode=mode,
            context=context,
            max_steps=max_steps,
            time_budget=time_budget,
        ),
    )


__all__ = ["build_run_experiment_tool", "build_experiment_tools"]

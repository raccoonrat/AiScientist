from __future__ import annotations

from aisci_domain_paper.tools.basic_tool import build_prioritization_tools, callback_tool


def build_prioritize_tasks_tool(engine):
    return callback_tool(
        "prioritize_tasks",
        "Prioritize the paper implementation work and write a ranked plan to /home/agent/prioritized_tasks.md.",
        {
            "type": "object",
            "properties": {
                "refresh": {"type": "boolean"},
            },
            "additionalProperties": False,
        },
        lambda shell, refresh=False: engine.prioritize_tasks(refresh=refresh),
    )


__all__ = ["build_prioritize_tasks_tool", "build_prioritization_tools"]

from __future__ import annotations

from aisci_domain_paper.tools.basic_tool import build_reader_tools, callback_tool


def build_read_paper_tool(engine):
    return callback_tool(
        "read_paper",
        "Read the staged paper bundle and write structured analysis artifacts to /home/agent/paper_analysis/.",
        {
            "type": "object",
            "properties": {
                "refresh": {
                    "type": "boolean",
                    "description": "Regenerate analysis even if files already exist.",
                }
            },
            "additionalProperties": False,
        },
        lambda shell, refresh=False: engine.read_paper(refresh=refresh),
    )


__all__ = ["build_read_paper_tool", "build_reader_tools"]

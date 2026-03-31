from __future__ import annotations

from aisci_domain_paper.tools.basic_tool import callback_tool


def build_clean_validation_tool(engine):
    return callback_tool(
        "clean_reproduce_validation",
        "Run the final self-check workflow and write a validation report.",
        {
            "type": "object",
            "properties": {
                "refresh": {"type": "boolean"},
            },
            "additionalProperties": False,
        },
        lambda shell, refresh=False: engine.run_clean_validation(refresh=refresh),
    )


__all__ = ["build_clean_validation_tool"]

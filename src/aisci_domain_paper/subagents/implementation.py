from __future__ import annotations

from aisci_domain_paper.prompts.templates import IMPLEMENTATION_SYSTEM_PROMPT
from aisci_domain_paper.subagents.base import PaperSubagent
from aisci_domain_paper.tools import (
    build_implementation_tools,
    build_spawn_env_setup_tool,
    build_spawn_resource_download_tool,
)


class PaperImplementationSubagent(PaperSubagent):
    @property
    def name(self) -> str:
        return "implementation"

    def system_prompt(self) -> str:
        return IMPLEMENTATION_SYSTEM_PROMPT

    def get_tools(self):
        return [
            *build_implementation_tools(),
            build_spawn_env_setup_tool(self.engine),
            build_spawn_resource_download_tool(self.engine),
        ]

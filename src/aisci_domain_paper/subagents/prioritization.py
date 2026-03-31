from __future__ import annotations

from aisci_domain_paper.prompts.templates import PRIORITIZATION_SYSTEM_PROMPT
from aisci_domain_paper.subagents.base import PaperSubagent
from aisci_domain_paper.tools import build_prioritization_tools


class PaperPrioritizationSubagent(PaperSubagent):
    @property
    def name(self) -> str:
        return "prioritization"

    def system_prompt(self) -> str:
        return PRIORITIZATION_SYSTEM_PROMPT

    def get_tools(self):
        return build_prioritization_tools()

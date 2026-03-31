from __future__ import annotations

from aisci_domain_paper.prompts.templates import EXPERIMENT_SYSTEM_PROMPT
from aisci_domain_paper.subagents.base import PaperSubagent
from aisci_domain_paper.tools import build_experiment_tools


class PaperExperimentSubagent(PaperSubagent):
    @property
    def name(self) -> str:
        return "experiment"

    def system_prompt(self) -> str:
        return EXPERIMENT_SYSTEM_PROMPT

    def get_tools(self):
        return build_experiment_tools()

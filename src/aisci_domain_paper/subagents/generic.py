from __future__ import annotations

from aisci_domain_paper.prompts.templates import EXPLORE_SYSTEM_PROMPT, GENERAL_SYSTEM_PROMPT, PLAN_SYSTEM_PROMPT
from aisci_domain_paper.subagents.base import PaperSubagent
from aisci_domain_paper.tools import build_generic_tools


class ExploreSubagent(PaperSubagent):
    @property
    def name(self) -> str:
        return "explore"

    def system_prompt(self) -> str:
        return EXPLORE_SYSTEM_PROMPT

    def get_tools(self):
        return build_generic_tools()


class PlanSubagent(PaperSubagent):
    @property
    def name(self) -> str:
        return "plan"

    def system_prompt(self) -> str:
        return PLAN_SYSTEM_PROMPT

    def get_tools(self):
        return build_generic_tools()


class GeneralSubagent(PaperSubagent):
    @property
    def name(self) -> str:
        return "general"

    def system_prompt(self) -> str:
        return GENERAL_SYSTEM_PROMPT

    def get_tools(self):
        return build_generic_tools()

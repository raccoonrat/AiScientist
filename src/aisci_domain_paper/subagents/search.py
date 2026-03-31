from __future__ import annotations

from aisci_domain_paper.prompts.templates import SEARCH_EXECUTOR_PROMPT, SEARCH_STRATEGIST_PROMPT
from aisci_domain_paper.subagents.base import PaperSubagent
from aisci_domain_paper.tools import build_reader_tools


class SearchStrategistSubagent(PaperSubagent):
    @property
    def name(self) -> str:
        return "search_strategist"

    def system_prompt(self) -> str:
        return SEARCH_STRATEGIST_PROMPT

    def get_tools(self):
        return build_reader_tools()


class SearchExecutorSubagent(PaperSubagent):
    @property
    def name(self) -> str:
        return "search_executor"

    def system_prompt(self) -> str:
        return SEARCH_EXECUTOR_PROMPT

    def get_tools(self):
        return build_reader_tools()

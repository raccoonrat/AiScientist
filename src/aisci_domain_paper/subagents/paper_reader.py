from __future__ import annotations

from aisci_domain_paper.prompts.templates import PAPER_READER_SYSTEM_PROMPT
from aisci_domain_paper.subagents.base import PaperSubagent
from aisci_domain_paper.tools import build_reader_tools


class PaperReaderSubagent(PaperSubagent):
    @property
    def name(self) -> str:
        return "paper_reader"

    def system_prompt(self) -> str:
        return PAPER_READER_SYSTEM_PROMPT

    def get_tools(self):
        return build_reader_tools()

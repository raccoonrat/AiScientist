from __future__ import annotations

from aisci_agent_runtime.subagents.base import SubagentConfig
from aisci_agent_runtime.summary_utils import SummaryConfig


DEFAULT_IMPLEMENTATION_CONFIG = SubagentConfig(
    max_steps=500,
    time_limit=28_800,
    reminder_freq=20,
    summary_config=SummaryConfig(),
)
DEFAULT_EXPERIMENT_CONFIG = SubagentConfig(
    max_steps=500,
    time_limit=36_000,
    reminder_freq=30,
    summary_config=SummaryConfig(),
)
DEFAULT_ENV_SETUP_CONFIG = SubagentConfig(max_steps=300, time_limit=7_200, reminder_freq=15)
DEFAULT_DOWNLOAD_CONFIG = SubagentConfig(max_steps=300, time_limit=7_200, reminder_freq=15)
DEFAULT_PAPER_STRUCTURE_CONFIG = SubagentConfig(
    max_steps=500,
    time_limit=36_000,
    reminder_freq=15,
)
DEFAULT_PAPER_READER_CONFIG = SubagentConfig(
    max_steps=500,
    time_limit=36_000,
    reminder_freq=15,
)
DEFAULT_PAPER_SYNTHESIS_CONFIG = SubagentConfig(
    max_steps=500,
    time_limit=36_000,
    reminder_freq=15,
)
DEFAULT_PRIORITIZATION_CONFIG = SubagentConfig(
    max_steps=500,
    time_limit=36_000,
    reminder_freq=15,
)
DEFAULT_EXPLORE_SUBAGENT_CONFIG = SubagentConfig(
    max_steps=300,
    time_limit=14_400,
    reminder_freq=15,
)
DEFAULT_PLAN_SUBAGENT_CONFIG = SubagentConfig(
    max_steps=200,
    time_limit=7_200,
    reminder_freq=15,
)
DEFAULT_GENERAL_SUBAGENT_CONFIG = SubagentConfig(
    max_steps=300,
    time_limit=14_400,
    reminder_freq=20,
)

MAIN_AGENT_BASH_DEFAULT_TIMEOUT = 36_000
MAIN_AGENT_BASH_MAX_TIMEOUT = 86_400
IMPLEMENTATION_BASH_DEFAULT_TIMEOUT = 36_000
EXPERIMENT_BASH_DEFAULT_TIMEOUT = 36_000
EXPERIMENT_COMMAND_TIMEOUT = 36_000
EXPERIMENT_VALIDATE_TIME_LIMIT = 18_000
EXPLORE_BASH_DEFAULT_TIMEOUT = 36_000
PLAN_BASH_DEFAULT_TIMEOUT = 36_000
GENERAL_BASH_DEFAULT_TIMEOUT = 36_000

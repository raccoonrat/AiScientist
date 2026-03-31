from aisci_domain_paper.subagents.env_setup import EnvSetupSubagent
from aisci_domain_paper.subagents.experiment import PaperExperimentSubagent
from aisci_domain_paper.subagents.generic import ExploreSubagent, GeneralSubagent, PlanSubagent
from aisci_domain_paper.subagents.implementation import PaperImplementationSubagent
from aisci_domain_paper.subagents.paper_reader import PaperReaderSubagent
from aisci_domain_paper.subagents.prioritization import PaperPrioritizationSubagent
from aisci_domain_paper.subagents.resource_download import ResourceDownloadSubagent
from aisci_domain_paper.subagents.search import SearchExecutorSubagent, SearchStrategistSubagent
from aisci_domain_paper.subagents.validation import PaperValidationSubagent


def subagent_class_for_kind(kind: str):
    normalized = kind.strip().lower()
    mapping = {
        "reader": PaperReaderSubagent,
        "paper_reader": PaperReaderSubagent,
        "prioritization": PaperPrioritizationSubagent,
        "implementation": PaperImplementationSubagent,
        "experiment": PaperExperimentSubagent,
        "validation": PaperValidationSubagent,
        "explore": ExploreSubagent,
        "plan": PlanSubagent,
        "general": GeneralSubagent,
        "generic": GeneralSubagent,
        "env_setup": EnvSetupSubagent,
        "resource_download": ResourceDownloadSubagent,
    }
    return mapping[normalized]


__all__ = [
    "EnvSetupSubagent",
    "ExploreSubagent",
    "GeneralSubagent",
    "PaperExperimentSubagent",
    "PaperImplementationSubagent",
    "PaperPrioritizationSubagent",
    "PaperReaderSubagent",
    "PaperValidationSubagent",
    "PlanSubagent",
    "ResourceDownloadSubagent",
    "SearchExecutorSubagent",
    "SearchStrategistSubagent",
    "subagent_class_for_kind",
]

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from aisci_agent_runtime.tools.base import SubagentCompleteSignal, Tool
from aisci_agent_runtime.tools.shell_tools import (
    AddExpLogTool,
    AddImplLogTool,
    BashToolWithTimeout,
    ExecCommandTool,
    PythonTool,
    ReadFileChunkTool,
    SearchFileTool,
)


@dataclass(frozen=True)
class _ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]


class CallbackTool(Tool):
    def __init__(self, spec: _ToolSpec, callback: Callable[..., Any]):
        self._spec = spec
        self._callback = callback

    def name(self) -> str:
        return self._spec.name

    def execute(self, shell, **kwargs) -> str:  # noqa: ANN001
        return str(self._callback(shell=shell, **kwargs))

    def get_tool_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self._spec.name,
                "description": self._spec.description,
                "parameters": self._spec.parameters,
            },
        }


def callback_tool(name: str, description: str, parameters: dict[str, Any], callback) -> CallbackTool:
    return CallbackTool(_ToolSpec(name=name, description=description, parameters=parameters), callback)


class MappedFileEditTool(Tool):
    def name(self) -> str:
        return "edit_file"

    def execute(
        self,
        shell,
        command: str,
        path: str,
        file_text: str = "",
        old_str: str = "",
        new_str: str = "",
        insert_line: int = 0,
        **kwargs: Any,
    ) -> str:
        if command == "create":
            shell.write_file(path, file_text)
            lines = file_text.count("\n") + 1 if file_text else 0
            return f"Created {path} ({lines} lines)"

        if command == "str_replace":
            if not old_str:
                return "Error: old_str is required for str_replace"
            if not shell.file_exists(path):
                return f"Error: {path} does not exist"
            content = shell.read_file(path)
            count = content.count(old_str)
            if count == 0:
                return f"Error: old_str not found in {path}. Use read_file_chunk first."
            if count > 1:
                return f"Error: old_str appears {count} times in {path}. Provide more context."
            shell.write_file(path, content.replace(old_str, new_str, 1))
            return f"Replaced in {path}"

        if command == "insert":
            if not shell.file_exists(path):
                return f"Error: {path} does not exist"
            content = shell.read_file(path)
            lines = content.split("\n")
            idx = max(0, min(insert_line, len(lines)))
            lines.insert(idx, new_str)
            shell.write_file(path, "\n".join(lines))
            return f"Inserted at line {idx} in {path}"

        return f"Error: unknown command '{command}'. Use create / str_replace / insert."

    def get_tool_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "edit_file",
                "description": "Create or edit files with create, str_replace, or insert modes.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "enum": ["create", "str_replace", "insert"]},
                        "path": {"type": "string"},
                        "file_text": {"type": "string"},
                        "old_str": {"type": "string"},
                        "new_str": {"type": "string"},
                        "insert_line": {"type": "integer"},
                    },
                    "required": ["command", "path"],
                    "additionalProperties": False,
                },
            },
        }


class PaperGitCommitTool(Tool):
    def name(self) -> str:
        return "git_commit"

    def execute(self, shell, message: str, **kwargs: Any) -> str:
        shell.send_command("cd /home/submission && git init 2>/dev/null", timeout=10)
        gitignore_path = "/home/submission/.gitignore"
        if not shell.file_exists(gitignore_path):
            shell.write_file(
                gitignore_path,
                "\n".join(
                    [
                        "# Auto-managed by AiScientist paper mode",
                        "venv/",
                        ".venv/",
                        "__pycache__/",
                        "*.pyc",
                        ".cache/",
                        "data/",
                        "models/",
                        "checkpoints/",
                        "",
                    ]
                ),
            )
        shell.write_file("/tmp/_paper_commit_msg.txt", message)
        result = shell.send_command(
            "cd /home/submission && "
            "git config user.email 'aiscientist@local' && "
            "git config user.name 'AiScientist' && "
            "git add -A && git commit -F /tmp/_paper_commit_msg.txt 2>&1",
            timeout=90,
        )
        return result.output.strip()

    def get_tool_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "git_commit",
                "description": "Stage and commit the /home/submission repository with the provided message.",
                "parameters": {
                    "type": "object",
                    "properties": {"message": {"type": "string"}},
                    "required": ["message"],
                    "additionalProperties": False,
                },
            },
        }


class SubmitTool(Tool):
    def name(self) -> str:
        return "submit"

    def execute(self, shell, summary: str, **kwargs: Any) -> str:  # noqa: ARG002
        raise SubagentCompleteSignal(summary, kwargs)

    def get_tool_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "submit",
                "description": "Finish the paper run after reading, prioritization, implementation, experiments, and self-check are complete.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "summary": {
                            "type": "string",
                            "description": "Concise final summary of what was implemented and validated.",
                        }
                    },
                    "required": ["summary"],
                    "additionalProperties": False,
                },
            },
        }


class FinishRunTool(SubmitTool):
    def name(self) -> str:
        return "finish_run"

    def get_tool_schema(self) -> dict[str, Any]:
        schema = super().get_tool_schema()
        schema["function"]["name"] = "finish_run"
        schema["function"]["description"] = "Compatibility alias for submit(). Prefer submit()."
        return schema


def build_shared_file_tools() -> list[Tool]:
    return [
        ReadFileChunkTool(),
        SearchFileTool(),
        BashToolWithTimeout(),
        PythonTool(),
        MappedFileEditTool(),
    ]


def build_reader_tools() -> list[Tool]:
    from aisci_agent_runtime.tools.base import SubagentCompleteTool

    return [*build_shared_file_tools(), SubagentCompleteTool()]


def build_prioritization_tools() -> list[Tool]:
    from aisci_agent_runtime.tools.base import SubagentCompleteTool

    return [*build_shared_file_tools(), SubagentCompleteTool()]


def build_generic_tools() -> list[Tool]:
    from aisci_agent_runtime.tools.base import SubagentCompleteTool

    return [*build_shared_file_tools(), PaperGitCommitTool(), SubagentCompleteTool()]


def build_implementation_tools() -> list[Tool]:
    from aisci_agent_runtime.tools.base import SubagentCompleteTool

    return [
        *build_shared_file_tools(),
        PaperGitCommitTool(),
        AddImplLogTool(),
        SubagentCompleteTool(),
    ]


def build_experiment_tools() -> list[Tool]:
    from aisci_agent_runtime.tools.base import SubagentCompleteTool

    return [
        ReadFileChunkTool(),
        SearchFileTool(),
        BashToolWithTimeout(default_timeout=900, max_timeout=14_400),
        PythonTool(default_timeout=900, max_timeout=14_400),
        MappedFileEditTool(),
        ExecCommandTool(default_timeout=3600, max_timeout=18_000),
        PaperGitCommitTool(),
        AddExpLogTool(),
        SubagentCompleteTool(),
    ]


def build_env_setup_tools() -> list[Tool]:
    from aisci_agent_runtime.tools.base import SubagentCompleteTool

    return [ReadFileChunkTool(), BashToolWithTimeout(default_timeout=600, max_timeout=3600), MappedFileEditTool(), SubagentCompleteTool()]


def build_resource_download_tools() -> list[Tool]:
    from aisci_agent_runtime.tools.base import SubagentCompleteTool

    return [
        ReadFileChunkTool(),
        BashToolWithTimeout(default_timeout=900, max_timeout=7200),
        PythonTool(default_timeout=900, max_timeout=7200),
        MappedFileEditTool(),
        SubagentCompleteTool(),
    ]

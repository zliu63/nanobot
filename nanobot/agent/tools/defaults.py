"""Default tool registration factory."""

from pathlib import Path

from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.tools.filesystem import ReadFileTool, WriteFileTool, EditFileTool, ListDirTool
from nanobot.agent.tools.shell import ExecTool
from nanobot.agent.tools.web import WebSearchTool, WebFetchTool


def build_default_tools(
    *,
    workspace: Path,
    exec_config: "ExecToolConfig",
    restrict_to_workspace: bool = False,
    brave_api_key: str | None = None,
    include_agent_tools: bool = True,
    bus: "MessageBus | None" = None,
    subagents: "SubagentManager | None" = None,
    cron_service: "CronService | None" = None,
) -> ToolRegistry:
    """Build the standard tool registry.

    Args:
        include_agent_tools: If True, include MessageTool, SpawnTool, and CronTool
            (should be False for subagents to prevent recursion).
    """
    tools = ToolRegistry()
    allowed_dir = workspace if restrict_to_workspace else None

    # File tools
    tools.register(ReadFileTool(allowed_dir=allowed_dir))
    tools.register(WriteFileTool(allowed_dir=allowed_dir))
    tools.register(EditFileTool(allowed_dir=allowed_dir))
    tools.register(ListDirTool(allowed_dir=allowed_dir))

    # Shell tool
    tools.register(ExecTool(
        working_dir=str(workspace),
        timeout=exec_config.timeout,
        restrict_to_workspace=restrict_to_workspace,
    ))

    # Web tools
    tools.register(WebSearchTool(api_key=brave_api_key))
    tools.register(WebFetchTool())

    # Agent-only tools (message, spawn, cron)
    if include_agent_tools:
        from nanobot.agent.tools.message import MessageTool
        from nanobot.agent.tools.spawn import SpawnTool
        from nanobot.agent.tools.cron import CronTool

        if bus is not None:
            tools.register(MessageTool(send_callback=bus.publish_outbound))
        if subagents is not None:
            tools.register(SpawnTool(manager=subagents))
        if cron_service is not None:
            tools.register(CronTool(cron_service))

    return tools

"""Tool registry for dynamic tool management."""

from typing import Any
import time

from nanobot.agent.tools.base import Tool
from nanobot.agent.tools.monitoring import monitor


class ToolRegistry:
    """
    Registry for agent tools.
    
    Allows dynamic registration and execution of tools.
    """
    
    def __init__(self):
        self._tools: dict[str, Tool] = {}
    
    def register(self, tool: Tool) -> None:
        """Register a tool."""
        self._tools[tool.name] = tool
    
    def unregister(self, name: str) -> None:
        """Unregister a tool by name."""
        self._tools.pop(name, None)
    
    def get(self, name: str) -> Tool | None:
        """Get a tool by name."""
        return self._tools.get(name)
    
    def has(self, name: str) -> bool:
        """Check if a tool is registered."""
        return name in self._tools
    
    def get_definitions(self) -> list[dict[str, Any]]:
        """Get all tool definitions in OpenAI format."""
        return [tool.to_schema() for tool in self._tools.values()]
    
    async def execute(self, name: str, params: dict[str, Any]) -> str:
        """
        Execute a tool by name with given parameters.
        
        Args:
            name: Tool name.
            params: Tool parameters.
        
        Returns:
            Tool execution result as string.
        
        Raises:
            KeyError: If tool not found.
        """
        tool = self._tools.get(name)
        if not tool:
            monitor.record_execution(name, 0.0, success=False, error=f"Tool '{name}' not found")
            return f"Error: Tool '{name}' not found"

        start_time = time.time()
        try:
            errors = tool.validate_params(params)
            if errors:
                error_msg = f"Invalid parameters for tool '{name}': " + "; ".join(errors)
                monitor.record_execution(name, time.time() - start_time, success=False, error=error_msg)
                return f"Error: {error_msg}"
            
            result = await tool.execute(**params)
            monitor.record_execution(name, time.time() - start_time, success=True)
            return result
        except Exception as e:
            monitor.record_execution(name, time.time() - start_time, success=False, error=str(e))
            return f"Error executing {name}: {str(e)}"
    
    @property
    def tool_names(self) -> list[str]:
        """Get list of registered tool names."""
        return list(self._tools.keys())
    
    def __len__(self) -> int:
        return len(self._tools)
    
    def __contains__(self, name: str) -> bool:
        return name in self._tools

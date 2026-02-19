"""
Monitoring module for nanobot tool execution tracking.

This module provides:
1. Tool execution statistics collection
2. Performance monitoring
3. Health status reporting
4. Data persistence

Usage:
    from nanobot.monitoring import monitor, get_stats
    
    # Monitor a tool function
    @monitor.tool('tool_name')
    async def my_tool_function(args):
        # tool implementation
        pass
    
    # Get current statistics
    stats = get_stats()
"""

import time
import json
import asyncio
from datetime import datetime
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, asdict, field
from functools import wraps
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

@dataclass
class ToolExecutionRecord:
    """Record of a single tool execution."""
    tool_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_ms: Optional[float] = None
    success: Optional[bool] = None
    error_message: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None
    result_summary: Optional[str] = None
    
    def complete(self, success: bool, error_message: Optional[str] = None, 
                 result_summary: Optional[str] = None):
        """Mark the execution as complete."""
        self.end_time = datetime.now()
        self.duration_ms = (self.end_time - self.start_time).total_seconds() * 1000
        self.success = success
        self.error_message = error_message
        self.result_summary = result_summary

@dataclass
class ToolStatistics:
    """Aggregate statistics for a tool."""
    name: str
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    total_duration_ms: float = 0.0
    avg_duration_ms: float = 0.0
    min_duration_ms: Optional[float] = None
    max_duration_ms: Optional[float] = None
    last_called: Optional[datetime] = None
    last_error: Optional[str] = None
    recent_errors: List[str] = field(default_factory=list)
    
    def record_call(self, duration_ms: float, success: bool, error_message: Optional[str] = None):
        """Record a tool call."""
        self.total_calls += 1
        self.total_duration_ms += duration_ms
        self.avg_duration_ms = self.total_duration_ms / self.total_calls
        
        if self.min_duration_ms is None or duration_ms < self.min_duration_ms:
            self.min_duration_ms = duration_ms
        if self.max_duration_ms is None or duration_ms > self.max_duration_ms:
            self.max_duration_ms = duration_ms
        
        self.last_called = datetime.now()
        
        if success:
            self.successful_calls += 1
        else:
            self.failed_calls += 1
            self.last_error = error_message
            if error_message:
                self.recent_errors.append(f"{self.last_called.isoformat()}: {error_message}")
                # Keep only last 10 errors
                if len(self.recent_errors) > 10:
                    self.recent_errors = self.recent_errors[-10:]
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        if self.total_calls == 0:
            return 0.0
        return (self.successful_calls / self.total_calls) * 100
    
    @property
    def avg_success_rate(self) -> float:
        """Alias for success_rate for backward compatibility."""
        return self.success_rate

class ToolMonitor:
    """Main monitoring class for tracking tool executions."""
    
    def __init__(self, max_history: int = 1000):
        self.stats: Dict[str, ToolStatistics] = {}
        self.history: List[ToolExecutionRecord] = []
        self.max_history = max_history
        self.session_start = datetime.now()
        self.total_tool_calls = 0
        self.total_errors = 0
        
    def tool(self, name: str):
        """Decorator to monitor a tool function."""
        def decorator(func: Callable):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                return await self._monitor_execution(name, func, *args, **kwargs)
            
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                # For synchronous functions, run in event loop
                return asyncio.run(self._monitor_execution(name, func, *args, **kwargs))
            
            # Return appropriate wrapper based on function type
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper
        
        return decorator
    
    async def _monitor_execution(self, tool_name: str, func: Callable, *args, **kwargs):
        """Monitor a single tool execution."""
        # Create execution record
        record = ToolExecutionRecord(
            tool_name=tool_name,
            start_time=datetime.now(),
            parameters=self._summarize_parameters(kwargs)
        )
        
        # Initialize statistics if needed
        if tool_name not in self.stats:
            self.stats[tool_name] = ToolStatistics(name=tool_name)
        
        # Execute and monitor
        try:
            start_time = time.time()
            result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
            duration_ms = (time.time() - start_time) * 1000
            
            # Record success
            self.stats[tool_name].record_call(duration_ms, True)
            record.complete(True, result_summary=self._summarize_result(result))
            
            self.total_tool_calls += 1
            logger.debug(f"Tool {tool_name} executed successfully in {duration_ms:.2f}ms")
            
            return result
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            error_msg = str(e)
            
            # Record failure
            self.stats[tool_name].record_call(duration_ms, False, error_msg)
            record.complete(False, error_message=error_msg)
            
            self.total_tool_calls += 1
            self.total_errors += 1
            logger.warning(f"Tool {tool_name} failed after {duration_ms:.2f}ms: {error_msg}")
            
            # Re-raise the exception
            raise
    
    def _summarize_parameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Create a safe summary of tool parameters."""
        summary = {}
        for key, value in params.items():
            if isinstance(value, (str, int, float, bool, type(None))):
                summary[key] = value
            elif isinstance(value, (list, tuple)):
                summary[key] = f"{type(value).__name__}[{len(value)}]"
            elif isinstance(value, dict):
                summary[key] = f"dict[{len(value)}]"
            else:
                summary[key] = type(value).__name__
        return summary
    
    def _summarize_result(self, result: Any) -> str:
        """Create a safe summary of tool result."""
        if result is None:
            return "None"
        elif isinstance(result, str):
            # Truncate long strings
            if len(result) > 100:
                return f"str[{len(result)}]: {result[:97]}..."
            return f"str[{len(result)}]"
        elif isinstance(result, (int, float, bool)):
            return str(result)
        elif isinstance(result, (list, tuple, dict)):
            return f"{type(result).__name__}[{len(result)}]"
        else:
            return type(result).__name__
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        return {
            "session": {
                "start_time": self.session_start.isoformat(),
                "duration_seconds": (datetime.now() - self.session_start).total_seconds(),
                "total_tool_calls": self.total_tool_calls,
                "total_errors": self.total_errors,
                "error_rate": (self.total_errors / self.total_tool_calls * 100) if self.total_tool_calls > 0 else 0,
            },
            "tools": {
                name: {
                    "total_calls": stats.total_calls,
                    "successful_calls": stats.successful_calls,
                    "failed_calls": stats.failed_calls,
                    "success_rate": stats.success_rate,
                    "avg_duration_ms": stats.avg_duration_ms,
                    "min_duration_ms": stats.min_duration_ms,
                    "max_duration_ms": stats.max_duration_ms,
                    "last_called": stats.last_called.isoformat() if stats.last_called else None,
                    "last_error": stats.last_error,
                    "recent_errors": stats.recent_errors[-5:] if stats.recent_errors else [],
                }
                for name, stats in self.stats.items()
            },
            "performance": {
                "total_duration_ms": sum(s.total_duration_ms for s in self.stats.values()),
                "avg_duration_ms_all": (
                    sum(s.total_duration_ms for s in self.stats.values()) / 
                    sum(s.total_calls for s in self.stats.values())
                    if sum(s.total_calls for s in self.stats.values()) > 0 else 0
                ),
                "most_used_tool": max(
                    self.stats.items(), 
                    key=lambda x: x[1].total_calls, 
                    default=(None, None)
                )[0],
                "least_reliable_tool": min(
                    self.stats.items(), 
                    key=lambda x: x[1].success_rate if x[1].total_calls > 0 else 100,
                    default=(None, None)
                )[0],
            }
        }
    
    def save_to_file(self, filepath: str):
        """Save statistics to JSON file."""
        data = self.get_statistics()
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    def clear(self):
        """Clear all statistics (for testing)."""
        self.stats.clear()
        self.history.clear()
        self.session_start = datetime.now()
        self.total_tool_calls = 0
        self.total_errors = 0

# Global monitor instance
_monitor_instance: Optional[ToolMonitor] = None

def get_monitor() -> ToolMonitor:
    """Get or create the global monitor instance."""
    global _monitor_instance
    if _monitor_instance is None:
        _monitor_instance = ToolMonitor()
    return _monitor_instance

def monitor_tool(name: str):
    """Convenience decorator for monitoring tools."""
    return get_monitor().tool(name)

def get_stats() -> Dict[str, Any]:
    """Get current statistics."""
    return get_monitor().get_statistics()

def save_stats(filepath: str):
    """Save statistics to file."""
    get_monitor().save_to_file(filepath)

def reset_stats():
    """Reset all statistics (for testing)."""
    get_monitor().clear()

# Example usage
if __name__ == "__main__":
    # Test the monitoring system
    import asyncio
    
    @monitor_tool("test_tool")
    async def test_async_tool(delay: float = 0.1):
        """Test tool that simulates work."""
        await asyncio.sleep(delay)
        return f"Slept for {delay}s"
    
    @monitor_tool("test_failing_tool")
    async def test_failing_tool():
        """Test tool that always fails."""
        raise ValueError("Test error")
    
    async def main():
        # Test successful execution
        result = await test_async_tool(0.05)
        print(f"Result: {result}")
        
        # Test failing execution
        try:
            await test_failing_tool()
        except ValueError:
            print("Tool failed as expected")
        
        # Get statistics
        stats = get_stats()
        print(json.dumps(stats, indent=2))
    
    asyncio.run(main())
"""
Tool execution monitoring system for nanobot.
Records tool usage statistics for performance analysis and debugging.
"""

import time
import json
from datetime import datetime
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict, field
from functools import wraps
import threading

@dataclass
class ToolStats:
    """Statistics for a single tool."""
    name: str
    call_count: int = 0
    success_count: int = 0
    error_count: int = 0
    total_execution_time: float = 0.0
    last_execution: Optional[datetime] = None
    last_error: Optional[str] = None
    errors: list = field(default_factory=list)
    
    @property
    def avg_execution_time(self) -> float:
        """Average execution time in seconds."""
        if self.call_count == 0:
            return 0.0
        return self.total_execution_time / self.call_count
    
    @property
    def success_rate(self) -> float:
        """Success rate as percentage."""
        if self.call_count == 0:
            return 0.0
        return (self.success_count / self.call_count) * 100
    
    def record_success(self, execution_time: float):
        """Record a successful tool execution."""
        self.call_count += 1
        self.success_count += 1
        self.total_execution_time += execution_time
        self.last_execution = datetime.now()
    
    def record_error(self, execution_time: float, error: str):
        """Record a failed tool execution."""
        self.call_count += 1
        self.error_count += 1
        self.total_execution_time += execution_time
        self.last_execution = datetime.now()
        self.last_error = error
        self.errors.append({
            "timestamp": datetime.now().isoformat(),
            "error": error,
            "execution_time": execution_time
        })

class ToolMonitor:
    """Monitor for tracking tool execution statistics."""
    
    def __init__(self):
        self.stats: Dict[str, ToolStats] = {}
        self.session_start = datetime.now()
        self._lock = threading.Lock()
    
    def get_or_create_stats(self, tool_name: str) -> ToolStats:
        """Get existing stats or create new ones."""
        with self._lock:
            if tool_name not in self.stats:
                self.stats[tool_name] = ToolStats(name=tool_name)
            return self.stats[tool_name]
    
    def record_execution(self, tool_name: str, execution_time: float, success: bool, error: Optional[str] = None):
        """Record a tool execution."""
        stats = self.get_or_create_stats(tool_name)
        if success:
            stats.record_success(execution_time)
        else:
            stats.record_error(execution_time, error or "Unknown error")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        with self._lock:
            total_calls = sum(s.call_count for s in self.stats.values())
            total_success = sum(s.success_count for s in self.stats.values())
            total_time = sum(s.total_execution_time for s in self.stats.values())
            
            return {
                "session_start": self.session_start.isoformat(),
                "session_duration_seconds": (datetime.now() - self.session_start).total_seconds(),
                "total_tools_monitored": len(self.stats),
                "total_tool_calls": total_calls,
                "successful_calls": total_success,
                "failed_calls": total_calls - total_success,
                "success_rate_percent": (total_success / total_calls * 100) if total_calls > 0 else 0,
                "total_execution_time_seconds": total_time,
                "avg_execution_time_seconds": total_time / total_calls if total_calls > 0 else 0,
                "most_used_tools": sorted(
                    [(name, stats.call_count) for name, stats in self.stats.items()],
                    key=lambda x: x[1],
                    reverse=True
                )[:5]
            }
    
    def get_detailed_stats(self) -> Dict[str, Any]:
        """Get detailed statistics for all tools."""
        with self._lock:
            return {
                "summary": self.get_summary(),
                "tools": {
                    name: {
                        "call_count": stats.call_count,
                        "success_count": stats.success_count,
                        "error_count": stats.error_count,
                        "success_rate_percent": stats.success_rate,
                        "total_execution_time": stats.total_execution_time,
                        "avg_execution_time": stats.avg_execution_time,
                        "last_execution": stats.last_execution.isoformat() if stats.last_execution else None,
                        "last_error": stats.last_error
                    }
                    for name, stats in self.stats.items()
                }
            }
    
    def save_to_file(self, filepath: str):
        """Save statistics to a JSON file."""
        with self._lock:
            data = {
                "exported_at": datetime.now().isoformat(),
                "data": self.get_detailed_stats()
            }
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2, default=str)
    
    def print_summary(self):
        """Print a human-readable summary to console."""
        summary = self.get_summary()
        tools = self.get_detailed_stats()["tools"]
        
        print("\n" + "="*60)
        print("TOOL EXECUTION MONITORING REPORT")
        print("="*60)
        print(f"Session started: {summary['session_start']}")
        print(f"Session duration: {summary['session_duration_seconds']:.1f} seconds")
        print(f"Tools monitored: {summary['total_tools_monitored']}")
        print(f"Total tool calls: {summary['total_tool_calls']}")
        print(f"Success rate: {summary['success_rate_percent']:.1f}%")
        print(f"Total execution time: {summary['total_execution_time_seconds']:.2f}s")
        print(f"Average execution time: {summary['avg_execution_time_seconds']:.3f}s")
        
        print("\nMost used tools:")
        for i, (tool_name, count) in enumerate(summary['most_used_tools'], 1):
            stats = tools[tool_name]
            print(f"  {i}. {tool_name}: {count} calls, "
                  f"{stats['success_rate_percent']:.1f}% success, "
                  f"avg {stats['avg_execution_time']:.3f}s")
        
        print("\nTool details:")
        for tool_name, tool_stats in tools.items():
            if tool_stats['call_count'] > 0:
                print(f"  • {tool_name}:")
                print(f"    - Calls: {tool_stats['call_count']}")
                print(f"    - Success: {tool_stats['success_count']}")
                print(f"    - Errors: {tool_stats['error_count']}")
                print(f"    - Success rate: {tool_stats['success_rate_percent']:.1f}%")
                print(f"    - Avg time: {tool_stats['avg_execution_time']:.3f}s")
                if tool_stats['last_error']:
                    print(f"    - Last error: {tool_stats['last_error'][:100]}...")
        
        print("="*60)

# Global monitor instance
monitor = ToolMonitor()

def monitor_tool_execution(func):
    """Decorator to monitor tool execution."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        tool_name = func.__name__
        start_time = time.time()
        
        try:
            result = await func(*args, **kwargs)
            execution_time = time.time() - start_time
            monitor.record_execution(tool_name, execution_time, success=True)
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            monitor.record_execution(tool_name, execution_time, success=False, error=str(e))
            raise
    
    return wrapper
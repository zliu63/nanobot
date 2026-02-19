"""
Monitoring tool for nanobot - provides access to tool execution statistics.
This tool allows users to view monitoring data without needing direct access to the monitoring module.
"""

import json
from datetime import datetime
from typing import Dict, Any, Optional

from nanobot.agent.tools.base import Tool
from nanobot.agent.tools.monitoring import monitor


class MonitoringTool(Tool):
    """Tool for accessing monitoring statistics."""
    
    def __init__(self):
        super().__init__(
            name="monitor",
            description="View tool execution statistics and system health",
            parameters={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["summary", "detailed", "save", "health", "reset"],
                        "description": "Action to perform: 'summary' for brief stats, 'detailed' for full details, 'save' to export to file, 'health' for system health check, 'reset' to clear statistics",
                        "default": "summary"
                    },
                    "filename": {
                        "type": "string",
                        "description": "Filename to save statistics to (required for 'save' action)",
                        "default": "monitoring_stats.json"
                    },
                    "tool_name": {
                        "type": "string",
                        "description": "Specific tool name to get details for (optional)",
                        "default": ""
                    }
                },
                "required": ["action"]
            }
        )
    
    async def execute(self, action: str = "summary", filename: str = "monitoring_stats.json", 
                     tool_name: str = "") -> str:
        """
        Execute monitoring command.
        
        Args:
            action: Action to perform
            filename: Filename for saving statistics
            tool_name: Specific tool name to query
        
        Returns:
            Monitoring results as formatted string
        """
        if action == "summary":
            return self._get_summary()
        elif action == "detailed":
            return self._get_detailed(tool_name)
        elif action == "save":
            return self._save_to_file(filename)
        elif action == "health":
            return self._health_check()
        elif action == "reset":
            return self._reset_stats()
        else:
            return f"Error: Unknown action '{action}'"
    
    def _get_summary(self) -> str:
        """Get summary statistics."""
        summary = monitor.get_summary()
        
        result = [
            "📊 **TOOL EXECUTION MONITORING SUMMARY**",
            "=" * 50,
            f"Session started: {summary['session_start']}",
            f"Duration: {summary['session_duration_seconds']:.1f} seconds",
            f"Tools monitored: {summary['total_tools_monitored']}",
            f"Total tool calls: {summary['total_tool_calls']}",
            f"Successful calls: {summary['successful_calls']}",
            f"Failed calls: {summary['failed_calls']}",
            f"Success rate: {summary['success_rate_percent']:.1f}%",
            f"Total execution time: {summary['total_execution_time_seconds']:.2f}s",
            f"Average execution time: {summary['avg_execution_time_seconds']:.3f}s",
            "",
            "**Most Used Tools:**"
        ]
        
        for i, (tool_name, count) in enumerate(summary['most_used_tools'], 1):
            result.append(f"  {i}. {tool_name}: {count} calls")
        
        result.extend([
            "",
            "**Health Status:**",
            self._get_health_status(summary),
            "=" * 50,
            "Use `monitor` with action='detailed' for more information.",
            "Use `monitor` with action='health' for detailed health check."
        ])
        
        return "\n".join(result)
    
    def _get_detailed(self, tool_name: str = "") -> str:
        """Get detailed statistics."""
        detailed = monitor.get_detailed_stats()
        summary = detailed["summary"]
        tools = detailed["tools"]
        
        if tool_name:
            # Get specific tool details
            if tool_name not in tools:
                return f"Error: Tool '{tool_name}' not found in monitoring data."
            
            tool_stats = tools[tool_name]
            result = [
                f"🔍 **DETAILED STATISTICS FOR: {tool_name}**",
                "=" * 50,
                f"Total calls: {tool_stats['call_count']}",
                f"Successful: {tool_stats['success_count']}",
                f"Failed: {tool_stats['error_count']}",
                f"Success rate: {tool_stats['success_rate_percent']:.1f}%",
                f"Total execution time: {tool_stats['total_execution_time']:.2f}s",
                f"Average execution time: {tool_stats['avg_execution_time']:.3f}s",
                f"Last execution: {tool_stats['last_execution'] or 'Never'}",
            ]
            
            if tool_stats['last_error']:
                result.append(f"Last error: {tool_stats['last_error']}")
            
            return "\n".join(result)
        
        # Get all tools details
        result = [
            "📋 **DETAILED TOOL STATISTICS**",
            "=" * 50,
            f"Session: {summary['session_start']} ({summary['session_duration_seconds']:.1f}s)",
            f"Total tools: {summary['total_tools_monitored']}",
            f"Total calls: {summary['total_tool_calls']}",
            f"Success rate: {summary['success_rate_percent']:.1f}%",
            "",
            "**Tool Details:**"
        ]
        
        for name, stats in sorted(tools.items(), key=lambda x: x[1]['call_count'], reverse=True):
            if stats['call_count'] > 0:
                result.append(
                    f"• {name}: {stats['call_count']} calls, "
                    f"{stats['success_rate_percent']:.1f}% success, "
                    f"avg {stats['avg_execution_time']:.3f}s"
                )
        
        result.extend([
            "",
            "**Performance Summary:**",
            f"Fastest tool: {self._find_fastest_tool(tools)}",
            f"Most reliable tool: {self._find_most_reliable_tool(tools)}",
            f"Most used tool: {self._find_most_used_tool(tools)}",
            "=" * 50
        ])
        
        return "\n".join(result)
    
    def _save_to_file(self, filename: str) -> str:
        """Save statistics to file."""
        try:
            monitor.save_to_file(filename)
            return f"✅ Statistics saved to '{filename}'"
        except Exception as e:
            return f"❌ Error saving statistics: {str(e)}"
    
    def _health_check(self) -> str:
        """Perform system health check."""
        detailed = monitor.get_detailed_stats()
        summary = detailed["summary"]
        tools = detailed["tools"]
        
        issues = []
        warnings = []
        
        # Check overall health
        if summary['total_tool_calls'] == 0:
            warnings.append("No tool calls recorded yet - system may be idle")
        
        if summary['success_rate_percent'] < 90:
            issues.append(f"Low overall success rate: {summary['success_rate_percent']:.1f}%")
        
        # Check individual tools
        for tool_name, stats in tools.items():
            if stats['call_count'] > 10 and stats['success_rate_percent'] < 80:
                issues.append(f"Tool '{tool_name}' has low success rate: {stats['success_rate_percent']:.1f}%")
            
            if stats['avg_execution_time'] > 5.0:  # More than 5 seconds average
                warnings.append(f"Tool '{tool_name}' is slow: avg {stats['avg_execution_time']:.3f}s")
        
        # Check for error streaks
        error_prone_tools = [
            name for name, stats in tools.items() 
            if stats['call_count'] > 5 and stats['error_count'] >= 3
        ]
        if error_prone_tools:
            issues.append(f"Tools with multiple errors: {', '.join(error_prone_tools)}")
        
        # Generate report
        result = [
            "🏥 **SYSTEM HEALTH CHECK**",
            "=" * 50,
            f"Check time: {datetime.now().isoformat()}",
            f"Session duration: {summary['session_duration_seconds']:.1f}s",
            f"Tools monitored: {summary['total_tools_monitored']}",
            f"Total calls: {summary['total_tool_calls']}",
            f"Success rate: {summary['success_rate_percent']:.1f}%",
            ""
        ]
        
        if issues:
            result.append("❌ **ISSUES FOUND:**")
            for issue in issues:
                result.append(f"  • {issue}")
            result.append("")
        
        if warnings:
            result.append("⚠️ **WARNINGS:**")
            for warning in warnings:
                result.append(f"  • {warning}")
            result.append("")
        
        if not issues and not warnings:
            result.append("✅ **SYSTEM HEALTHY**")
            result.append("All systems operating within normal parameters.")
        
        result.extend([
            "",
            "**Recommendations:**"
        ])
        
        if issues:
            result.append("1. Investigate tools with low success rates")
            result.append("2. Check error logs for failing tools")
            result.append("3. Consider adding retry logic for error-prone tools")
        
        if warnings:
            result.append("4. Optimize slow tools for better performance")
        
        if not issues and not warnings:
            result.append("1. Continue normal operations")
            result.append("2. Consider adding more monitoring features")
        
        result.append("=" * 50)
        
        return "\n".join(result)
    
    def _reset_stats(self) -> str:
        """Reset monitoring statistics."""
        # Note: The existing monitor doesn't have a reset method
        # For now, we'll create a new monitor instance
        from nanobot.agent.tools import monitoring
        monitoring.monitor = monitoring.ToolMonitor()
        return "✅ Monitoring statistics have been reset. New session started."
    
    def _get_health_status(self, summary: Dict[str, Any]) -> str:
        """Get simple health status string."""
        if summary['total_tool_calls'] == 0:
            return "🟡 Idle - No tool calls yet"
        
        success_rate = summary['success_rate_percent']
        if success_rate >= 95:
            return "✅ Healthy - Excellent success rate"
        elif success_rate >= 80:
            return "🟡 Warning - Acceptable success rate"
        else:
            return "🔴 Critical - Low success rate"
    
    def _find_fastest_tool(self, tools: Dict[str, Any]) -> str:
        """Find the tool with lowest average execution time."""
        fastest = None
        fastest_time = float('inf')
        
        for name, stats in tools.items():
            if stats['call_count'] > 0 and stats['avg_execution_time'] < fastest_time:
                fastest = name
                fastest_time = stats['avg_execution_time']
        
        return f"{fastest or 'N/A'} ({fastest_time:.3f}s)" if fastest else "N/A"
    
    def _find_most_reliable_tool(self, tools: Dict[str, Any]) -> str:
        """Find the tool with highest success rate."""
        most_reliable = None
        best_rate = -1
        
        for name, stats in tools.items():
            if stats['call_count'] >= 3 and stats['success_rate_percent'] > best_rate:
                most_reliable = name
                best_rate = stats['success_rate_percent']
        
        return f"{most_reliable or 'N/A'} ({best_rate:.1f}%)" if most_reliable else "N/A"
    
    def _find_most_used_tool(self, tools: Dict[str, Any]) -> str:
        """Find the most frequently used tool."""
        most_used = None
        most_calls = -1
        
        for name, stats in tools.items():
            if stats['call_count'] > most_calls:
                most_used = name
                most_calls = stats['call_count']
        
        return f"{most_used or 'N/A'} ({most_calls} calls)" if most_used else "N/A"


# Create instance for registration
monitoring_tool = MonitoringTool()
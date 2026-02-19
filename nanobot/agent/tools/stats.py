"""
Tool for viewing monitoring statistics and system health.
Provides insights into tool usage, performance, and automated health checks.
"""

from typing import Any, Optional
import json
from datetime import datetime

from nanobot.agent.tools.base import Tool
from nanobot.agent.tools.monitoring import monitor


class StatsTool(Tool):
    """Tool for viewing monitoring statistics and system health."""
    
    name = "stats"
    description = "View tool execution monitoring statistics and system health checks"
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["view", "health", "tool", "reset"],
                "description": "Action to perform: 'view' for statistics (default), 'health' for system health check, 'tool' for specific tool details, 'reset' to clear statistics",
                "default": "view"
            },
            "format": {
                "type": "string",
                "enum": ["summary", "detailed", "json", "save"],
                "description": "Output format for 'view' action: summary (human-readable), detailed (full details), json (raw JSON), save (save to file)",
                "default": "summary"
            },
            "filename": {
                "type": "string",
                "description": "Filename to save statistics (required when format='save')",
                "default": ""
            },
            "tool_name": {
                "type": "string",
                "description": "Specific tool name to get details for (for action='tool')",
                "default": ""
            },
            "confirm": {
                "type": "boolean",
                "description": "Confirmation required for reset action (must be true)",
                "default": False
            }
        },
        "required": []
    }
    
    async def execute(self, action: str = "view", format: str = "summary", 
                     filename: str = "", tool_name: str = "", confirm: bool = False) -> str:
        """
        Execute the stats tool to view monitoring statistics and system health.
        
        Args:
            action: Action to perform (view, health, tool, reset)
            format: Output format for view action (summary, detailed, json, save)
            filename: Filename to save to (for format='save')
            tool_name: Specific tool name to get details for (for action='tool')
            confirm: Confirmation required for reset action
        
        Returns:
            Statistics or health check results in requested format.
        """
        if action == "view":
            # Backward compatibility: handle format-based view
            if format == "summary":
                return self._get_summary()
            elif format == "detailed":
                return self._get_detailed()
            elif format == "json":
                return self._get_json()
            elif format == "save":
                return self._save_to_file(filename)
            else:
                return f"Error: Unknown format '{format}'. Use 'summary', 'detailed', 'json', or 'save'."
        
        elif action == "health":
            return self._health_check()
        
        elif action == "tool":
            if not tool_name:
                return "Error: tool_name is required for action='tool'"
            return self._get_tool_details(tool_name)
        
        elif action == "reset":
            if not confirm:
                return "Error: Reset requires confirmation. Use confirm=true to clear statistics."
            return self._reset_stats()
        
        else:
            return f"Error: Unknown action '{action}'. Use 'view', 'health', 'tool', or 'reset'."
    
    def _get_summary(self) -> str:
        """Get human-readable summary."""
        summary = monitor.get_summary()
        tools = monitor.get_detailed_stats()["tools"]
        
        output = []
        output.append("=" * 70)
        output.append("TOOL EXECUTION MONITORING REPORT")
        output.append("=" * 70)
        output.append(f"Session started: {summary['session_start']}")
        output.append(f"Session duration: {summary['session_duration_seconds']:.1f} seconds")
        output.append(f"Tools monitored: {summary['total_tools_monitored']}")
        output.append(f"Total tool calls: {summary['total_tool_calls']}")
        output.append(f"Successful calls: {summary['successful_calls']}")
        output.append(f"Failed calls: {summary['failed_calls']}")
        output.append(f"Success rate: {summary['success_rate_percent']:.1f}%")
        output.append(f"Total execution time: {summary['total_execution_time_seconds']:.2f}s")
        output.append(f"Average execution time: {summary['avg_execution_time_seconds']:.3f}s")
        
        output.append("\n📊 Most used tools:")
        for i, (tool_name, count) in enumerate(summary['most_used_tools'], 1):
            stats = tools[tool_name]
            output.append(f"  {i}. {tool_name}:")
            output.append(f"     • Calls: {count}")
            output.append(f"     • Success rate: {stats['success_rate_percent']:.1f}%")
            output.append(f"     • Avg time: {stats['avg_execution_time']:.3f}s")
            if stats['last_error']:
                output.append(f"     • Last error: {stats['last_error'][:80]}...")
        
        # Show recent errors if any
        error_tools = [name for name, s in tools.items() if s['error_count'] > 0]
        if error_tools:
            output.append("\n⚠️  Tools with errors:")
            for tool_name in error_tools[:5]:  # Show top 5 error tools
                stats = tools[tool_name]
                output.append(f"  • {tool_name}: {stats['error_count']} errors")
                if stats['last_error']:
                    output.append(f"    Last error: {stats['last_error'][:60]}...")
        
        output.append("\n💡 Usage tips:")
        output.append("  • Use format='detailed' for full tool-by-tool statistics")
        output.append("  • Use format='json' for machine-readable data")
        output.append("  • Use format='save' with filename to export data")
        output.append("=" * 70)
        
        return "\n".join(output)
    
    def _get_detailed(self) -> str:
        """Get detailed statistics."""
        tools = monitor.get_detailed_stats()["tools"]
        
        output = []
        output.append("=" * 70)
        output.append("DETAILED TOOL STATISTICS")
        output.append("=" * 70)
        
        for tool_name, stats in sorted(tools.items()):
            if stats['call_count'] > 0:
                output.append(f"\n🔧 {tool_name}:")
                output.append(f"  • Total calls: {stats['call_count']}")
                output.append(f"  • Successful: {stats['success_count']}")
                output.append(f"  • Errors: {stats['error_count']}")
                output.append(f"  • Success rate: {stats['success_rate_percent']:.1f}%")
                output.append(f"  • Total execution time: {stats['total_execution_time']:.2f}s")
                output.append(f"  • Average execution time: {stats['avg_execution_time']:.3f}s")
                if stats['last_execution']:
                    output.append(f"  • Last execution: {stats['last_execution']}")
                if stats['last_error']:
                    output.append(f"  • Last error: {stats['last_error']}")
        
        # Show tools with zero calls
        zero_call_tools = [name for name, stats in tools.items() if stats['call_count'] == 0]
        if zero_call_tools:
            output.append(f"\n📭 Tools not used yet: {', '.join(zero_call_tools)}")
        
        output.append("=" * 70)
        return "\n".join(output)
    
    def _get_json(self) -> str:
        """Get statistics as JSON."""
        data = {
            "exported_at": datetime.now().isoformat(),
            "data": monitor.get_detailed_stats()
        }
        return json.dumps(data, indent=2, default=str)
    
    def _save_to_file(self, filename: str) -> str:
        """Save statistics to file."""
        if not filename:
            return "Error: filename is required when format='save'"
        
        try:
            monitor.save_to_file(filename)
            return f"✅ Statistics saved to {filename}"
        except Exception as e:
            return f"Error saving statistics: {str(e)}"
    
    def _health_check(self) -> str:
        """Perform automated system health check."""
        detailed = monitor.get_detailed_stats()
        summary = detailed["summary"]
        tools = detailed["tools"]
        
        # Health analysis
        issues = []
        warnings = []
        recommendations = []
        
        # Overall system health
        if summary['total_tool_calls'] == 0:
            warnings.append("System idle - no tool calls recorded")
        elif summary['success_rate_percent'] < 90:
            issues.append(f"Low overall success rate: {summary['success_rate_percent']:.1f}%")
            recommendations.append("Investigate failing tools")
        
        # Tool-specific analysis
        error_prone_tools = []
        slow_tools = []
        reliable_tools = []
        
        for tool_name, stats in tools.items():
            if stats['call_count'] >= 5:  # Only analyze tools with sufficient data
                # Check for error-prone tools
                if stats['success_rate_percent'] < 80:
                    error_prone_tools.append((tool_name, stats['success_rate_percent']))
                
                # Check for slow tools
                if stats['avg_execution_time'] > 2.0:  # More than 2 seconds average
                    slow_tools.append((tool_name, stats['avg_execution_time']))
                
                # Identify reliable tools
                if stats['success_rate_percent'] >= 95 and stats['call_count'] >= 10:
                    reliable_tools.append((tool_name, stats['success_rate_percent']))
        
        if error_prone_tools:
            tool_list = ", ".join([f"{name} ({rate:.1f}%)" for name, rate in error_prone_tools[:3]])
            issues.append(f"Error-prone tools: {tool_list}")
            recommendations.append(f"Review and fix {error_prone_tools[0][0]} implementation")
        
        if slow_tools:
            tool_list = ", ".join([f"{name} ({time:.2f}s)" for name, time in slow_tools[:3]])
            warnings.append(f"Slow tools: {tool_list}")
            recommendations.append("Optimize tool execution for better performance")
        
        # Generate health report
        output = []
        output.append("=" * 70)
        output.append("🏥 SYSTEM HEALTH CHECK REPORT")
        output.append("=" * 70)
        output.append(f"Check time: {datetime.now().isoformat()}")
        output.append(f"Session duration: {summary['session_duration_seconds']:.1f}s")
        output.append(f"Tools monitored: {summary['total_tools_monitored']}")
        output.append(f"Total tool calls: {summary['total_tool_calls']}")
        output.append(f"Success rate: {summary['success_rate_percent']:.1f}%")
        
        # Overall health status
        if not issues and not warnings:
            health_status = "✅ HEALTHY"
            output.append(f"\n{health_status}: All systems operating normally")
        elif issues:
            health_status = "🔴 CRITICAL"
            output.append(f"\n{health_status}: Issues detected requiring attention")
        else:
            health_status = "🟡 WARNING"
            output.append(f"\n{health_status}: Warnings detected")
        
        # Issues section
        if issues:
            output.append("\n❌ ISSUES:")
            for issue in issues:
                output.append(f"  • {issue}")
        
        # Warnings section
        if warnings:
            output.append("\n⚠️  WARNINGS:")
            for warning in warnings:
                output.append(f"  • {warning}")
        
        # Positive findings
        if reliable_tools:
            output.append("\n✅ RELIABLE TOOLS:")
            for tool_name, rate in reliable_tools[:5]:
                output.append(f"  • {tool_name}: {rate:.1f}% success rate")
        
        # Recommendations
        if recommendations:
            output.append("\n💡 RECOMMENDATIONS:")
            for i, rec in enumerate(recommendations, 1):
                output.append(f"  {i}. {rec}")
        
        # Additional recommendations based on system state
        if summary['total_tool_calls'] < 10:
            output.append("\n📊 DATA NOTE: Limited data available")
            output.append("  • Run more tool calls for accurate health assessment")
        
        output.append("\n🔍 NEXT STEPS:")
        output.append("  1. Use 'stats' with action='tool' to investigate specific tools")
        output.append("  2. Review error logs for failing tools")
        output.append("  3. Consider performance optimization for slow tools")
        
        output.append("=" * 70)
        return "\n".join(output)
    
    def _get_tool_details(self, tool_name: str) -> str:
        """Get detailed statistics for a specific tool."""
        tools = monitor.get_detailed_stats()["tools"]
        
        if tool_name not in tools:
            return f"Error: Tool '{tool_name}' not found in monitoring data."
        
        stats = tools[tool_name]
        
        output = []
        output.append("=" * 70)
        output.append(f"🔧 TOOL ANALYSIS: {tool_name}")
        output.append("=" * 70)
        
        if stats['call_count'] == 0:
            output.append(f"\n📭 Tool '{tool_name}' has not been used yet.")
            output.append("=" * 70)
            return "\n".join(output)
        
        # Basic statistics
        output.append(f"\n📊 BASIC STATISTICS:")
        output.append(f"  • Total calls: {stats['call_count']}")
        output.append(f"  • Successful: {stats['success_count']}")
        output.append(f"  • Errors: {stats['error_count']}")
        output.append(f"  • Success rate: {stats['success_rate_percent']:.1f}%")
        output.append(f"  • Total execution time: {stats['total_execution_time']:.2f}s")
        output.append(f"  • Average execution time: {stats['avg_execution_time']:.3f}s")
        
        if stats['min_duration_ms'] and stats['max_duration_ms']:
            output.append(f"  • Fastest execution: {stats['min_duration_ms']:.0f}ms")
            output.append(f"  • Slowest execution: {stats['max_duration_ms']:.0f}ms")
        
        if stats['last_execution']:
            output.append(f"  • Last execution: {stats['last_execution']}")
        
        # Performance assessment
        output.append(f"\n📈 PERFORMANCE ASSESSMENT:")
        
        if stats['success_rate_percent'] >= 95:
            output.append(f"  • Reliability: ✅ EXCELLENT ({stats['success_rate_percent']:.1f}%)")
        elif stats['success_rate_percent'] >= 80:
            output.append(f"  • Reliability: 🟡 ACCEPTABLE ({stats['success_rate_percent']:.1f}%)")
        else:
            output.append(f"  • Reliability: 🔴 POOR ({stats['success_rate_percent']:.1f}%)")
        
        if stats['avg_execution_time'] < 0.5:
            output.append(f"  • Speed: ✅ FAST ({stats['avg_execution_time']:.3f}s)")
        elif stats['avg_execution_time'] < 2.0:
            output.append(f"  • Speed: 🟡 MODERATE ({stats['avg_execution_time']:.3f}s)")
        else:
            output.append(f"  • Speed: 🔴 SLOW ({stats['avg_execution_time']:.3f}s)")
        
        # Error analysis
        if stats['error_count'] > 0:
            output.append(f"\n⚠️  ERROR ANALYSIS:")
            output.append(f"  • Error rate: {(stats['error_count'] / stats['call_count'] * 100):.1f}%")
            if stats['last_error']:
                output.append(f"  • Last error: {stats['last_error']}")
            
            if hasattr(stats, 'recent_errors') and stats['recent_errors']:
                output.append(f"  • Recent errors: {len(stats['recent_errors'])} in history")
        
        # Usage patterns
        output.append(f"\n📅 USAGE PATTERNS:")
        if stats['call_count'] >= 10:
            calls_per_minute = stats['call_count'] / (summary['session_duration_seconds'] / 60)
            output.append(f"  • Frequency: ~{calls_per_minute:.1f} calls per minute")
        
        # Recommendations
        output.append(f"\n💡 RECOMMENDATIONS:")
        
        if stats['success_rate_percent'] < 80:
            output.append("  1. Investigate and fix error causes")
            output.append("  2. Add better error handling")
            output.append("  3. Consider adding retry logic")
        
        if stats['avg_execution_time'] > 2.0:
            output.append("  1. Optimize tool implementation")
            output.append("  2. Consider caching frequent operations")
            output.append("  3. Review for performance bottlenecks")
        
        if stats['success_rate_percent'] >= 95 and stats['avg_execution_time'] < 0.5:
            output.append("  1. Tool is performing well - no action needed")
            output.append("  2. Consider using this tool as a reference for others")
        
        output.append("=" * 70)
        return "\n".join(output)
    
    def _reset_stats(self) -> str:
        """Reset monitoring statistics and start new session."""
        # Note: The existing monitor doesn't have a reset method
        # We need to create a new monitor instance
        # This is a workaround until we add proper reset to monitoring.py
        
        try:
            # Import the module to replace the global monitor
            import nanobot.agent.tools.monitoring as monitoring_module
            monitoring_module.monitor = monitoring_module.ToolMonitor()
            
            return "✅ Monitoring statistics have been reset. New session started.\n" \
                   f"Reset time: {datetime.now().isoformat()}"
        except Exception as e:
            return f"❌ Error resetting statistics: {str(e)}"
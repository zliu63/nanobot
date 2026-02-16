---
name: evolve
description: Self-evolution - improve the standby nanobot copy
---

# Self-Evolution

You are running as part of a blue/green deployment. You can modify the **standby** copy's code to evolve yourself.

## Context
- You run in one of two slots: blue (port 8001) or green (port 8002)
- The other slot is **standby** — you may modify its code
- The orchestrator watches for a `.swap-ready` sentinel file to trigger traffic switching
- State file: `/Users/ziyangliu/ProjectF/nanobot-orchestrator/state.json`

## Steps

1. **Read state**: `read_file /Users/ziyangliu/ProjectF/nanobot-orchestrator/state.json` — determine which slot is standby
2. **Identify standby path**:
   - If standby is "blue": `/Users/ziyangliu/ProjectF/nanobot-blue/`
   - If standby is "green": `/Users/ziyangliu/ProjectF/nanobot-green/`
3. **Analyze**: Read standby source code, identify improvements
4. **Modify**: Use `edit_file` / `write_file` to modify standby code
5. **Test**: `exec: cd /path/to/nanobot-{standby} && python -m pytest tests/ -x`
6. **Log**: Append to `/Users/ziyangliu/ProjectF/nanobot-orchestrator/evolution-log.jsonl`:
   ```json
   {"event": "evolution", "slot": "<standby>", "changes": "<summary>", "test_result": "passed"}
   ```
7. **Record**: Write what you changed to `memory/MEMORY.md`
8. **Signal**: Create sentinel file at `/Users/ziyangliu/ProjectF/nanobot-{standby}/.swap-ready`
   The orchestrator will detect this, restart the standby, health-check it, and switch traffic.

## Evolution Priorities (descending)
1. Add new skills (SKILL.md files)
2. Add new tools (Tool subclasses)
3. Optimize existing code (performance, clarity)
4. Fix bugs

## Safety Rules
- **NEVER** modify `/Users/ziyangliu/ProjectF/nanobot-orchestrator/` — the orchestrator is off-limits
- **NEVER** modify the **active** copy (yourself) — only the standby
- **NEVER** create `.swap-ready` unless all tests pass
- Keep changes small and incremental — one improvement per evolution cycle
- **NEVER** run `git` commands — the orchestrator handles all git commits and pushes automatically

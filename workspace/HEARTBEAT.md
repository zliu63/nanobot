# Heartbeat Tasks

This file is checked every 30 minutes. Add tasks you want the agent to handle periodically.
If the file has no actionable content (only headers/comments), the heartbeat is skipped.

---

## Standing Instructions

### Self-Evolution: Check for Pending Evolution Tasks

**IMPORTANT — Read this every heartbeat:**

1. Read the orchestrator state: `read_file /Users/ziyangliu/ProjectF/nanobot-orchestrator/state.json`
2. Determine if you are the **standby** instance (not the active one serving users).
3. Check if a pending evolution task exists: `read_file ~/.nanobot/workspace/experiments/pending-evolution.json`
4. If the file exists AND `"status": "pending"` AND you are **standby**:
   - Read the `change_description` and `target_file` fields carefully.
   - Load and use the **evolve** skill to implement the change on your own code.
   - After implementation, update `pending-evolution.json` to set `"status": "in_progress"`.
   - Run tests. If tests pass, set status to `"implemented"` and create `.swap-ready`.
   - If tests fail or you are unsure, set status to `"failed"` with a `"failure_reason"` field.
5. If you are the **active** instance, skip the evolution check — you serve users, not experiments.
6. If no pending evolution or already processed, reply `HEARTBEAT_OK`.

---

## Active Tasks

<!-- Add your periodic tasks below this line -->


## Completed

<!-- Move completed tasks here or delete them -->

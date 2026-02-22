"""Self-evolution engine: periodic self-audit and evolution planning.

Phase 1 responsibilities:
  1. Count accumulated signals and decide when to audit.
  2. Aggregate signal metrics into a structured report.
  3. Call LLM for diagnosis + hypotheses.
  4. Execute "immediate actions" (memory ops, focus notes).
  5. Write code-change hypotheses to workspace/experiments/pending-evolution.json
     so the standby instance can pick them up and implement them (Phase 2).
  6. Persist audit reports to workspace/self-audit/ and evolution_log.md.
"""

import json
from collections import Counter, defaultdict
from datetime import datetime, timezone

from nanobot.agent.json_utils import parse_json_response
from pathlib import Path
from typing import Any

from loguru import logger

from nanobot.utils.helpers import ensure_dir

# Trigger an audit after this many NEW signals since the last audit.
_AUDIT_EVERY_N = 20

# Keep at most this many signals in memory for analysis.
_MAX_SIGNALS_FOR_AUDIT = 100


class SelfAuditEngine:
    """Reads interaction signals, diagnoses patterns, produces evolution hypotheses.

    Lifecycle:
      - Instantiated once inside AgentLoop.__init__()
      - should_audit() is checked after every signal is collected
      - run_audit(provider, model) is called as a fire-and-forget background task
    """

    def __init__(self, workspace: Path):
        self.workspace = workspace
        self.signals_file = workspace / "signals" / "signals.jsonl"
        self.audit_dir = ensure_dir(workspace / "self-audit")
        self.experiments_dir = ensure_dir(workspace / "experiments")
        self.evolution_log = workspace / "evolution_log.md"
        self._state_file = self.experiments_dir / "_audit_state.json"

    # ──────────────────────────────────────────────────────────────────────────
    # Trigger logic
    # ──────────────────────────────────────────────────────────────────────────

    def should_audit(self) -> bool:
        """Return True if enough new signals have accumulated since last audit."""
        current = self._count_signals()
        last = self._load_state().get("last_audit_signal_count", 0)
        return (current - last) >= _AUDIT_EVERY_N

    def _count_signals(self) -> int:
        if not self.signals_file.exists():
            return 0
        return sum(1 for line in self.signals_file.read_text(encoding="utf-8").splitlines() if line.strip())

    # ──────────────────────────────────────────────────────────────────────────
    # Main audit flow
    # ──────────────────────────────────────────────────────────────────────────

    async def run_audit(self, provider: Any, model: str) -> dict | None:
        """Full audit cycle. Returns the parsed audit result dict, or None on error."""
        from nanobot.agent.memory import MemoryStore

        signals = self._load_recent_signals()
        if not signals:
            logger.debug("Self-audit: no signals yet, skipping.")
            return None

        aggregated = self._aggregate(signals)
        memory_summary = MemoryStore(self.workspace).build_memory_summary()
        prev_audit = self._load_last_audit()
        audit_number = self._load_state().get("audit_count", 0) + 1

        prompt = self._build_prompt(aggregated, memory_summary, prev_audit, audit_number)
        logger.info(f"Self-audit #{audit_number} starting ({len(signals)} signals analysed)")

        try:
            response = await provider.chat(
                messages=[
                    {"role": "system", "content": "You are a self-improvement agent. Respond only with valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                model=model,
                max_tokens=4096,
                temperature=0.3,
            )
            text = (response.content or "").strip()
            logger.debug(f"Self-audit raw LLM response:\n{text}")
            result: dict = parse_json_response(text)
        except Exception as e:
            logger.error(f"Self-audit LLM call failed: {e}")
            return None

        result["_meta"] = {
            "audit_number": audit_number,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "signals_analyzed": len(signals),
        }

        # Execute immediate memory/focus actions
        self._execute_immediate_actions(result.get("immediate_actions", []))

        # Write code-change hypotheses for standby to pick up
        code_hypotheses = [
            h for h in result.get("hypotheses", [])
            if h.get("requires_code_change", False)
        ]
        if code_hypotheses:
            self._write_pending_evolution(code_hypotheses, aggregated, audit_number)

        # Persist audit report
        self._save_audit_report(result, audit_number)
        self._append_evolution_log(result, audit_number)
        self._save_state(audit_number)

        logger.info(
            f"Self-audit #{audit_number} complete | "
            f"hypotheses={len(result.get('hypotheses', []))} "
            f"(code_changes={len(code_hypotheses)}) | "
            f"immediate_actions={len(result.get('immediate_actions', []))}"
        )
        return result

    # ──────────────────────────────────────────────────────────────────────────
    # Signal aggregation
    # ──────────────────────────────────────────────────────────────────────────

    def _load_recent_signals(self) -> list[dict]:
        if not self.signals_file.exists():
            return []
        lines = [l.strip() for l in self.signals_file.read_text(encoding="utf-8").splitlines() if l.strip()]
        rows = []
        for line in lines:
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
        # Return only the most recent N (no need to look at ancient history)
        return rows[-_MAX_SIGNALS_FOR_AUDIT:]

    def _aggregate(self, signals: list[dict]) -> dict:
        """Reduce raw signal rows into a human-readable metrics dict."""
        n = len(signals)
        corrections = 0
        tool_uses = 0
        tool_failures = 0
        elapsed_total = 0
        turns_total = 0
        channels: Counter = Counter()
        tools_counter: Counter = Counter()
        llm_calls_total = 0

        # Per-session tracking for turn counts
        sessions_seen: set = set()

        for row in signals:
            s = row.get("signals", {})
            if s.get("user_correction_detected"):
                corrections += 1
            tool_uses += s.get("tool_use_count", 0)
            tool_failures += s.get("tool_failures", 0)
            elapsed_total += s.get("elapsed_ms", 0)
            llm_calls_total += s.get("total_llm_calls", 1)
            channels[s.get("channel", "unknown")] += 1
            for t in s.get("tools_used", []):
                tools_counter[t] += 1
            sessions_seen.add(row.get("session_id", "?"))

        # Reconstruction of avg turns from session data
        session_turns: dict = defaultdict(int)
        for row in signals:
            sid = row.get("session_id", "?")
            t = row.get("signals", {}).get("conversation_turns", 1)
            session_turns[sid] = max(session_turns[sid], t)
        avg_turns = (sum(session_turns.values()) / len(session_turns)) if session_turns else 0

        first_ts = signals[0].get("timestamp", "?")[:10] if signals else "?"
        last_ts = signals[-1].get("timestamp", "?")[:10] if signals else "?"

        return {
            "period": f"{first_ts} → {last_ts}",
            "total_interactions": n,
            "unique_sessions": len(sessions_seen),
            "correction_rate": round(corrections / n, 3) if n else 0,
            "correction_count": corrections,
            "tool_failure_rate": round(tool_failures / max(tool_uses, 1), 3),
            "tool_use_count": tool_uses,
            "tool_failure_count": tool_failures,
            "avg_elapsed_ms": round(elapsed_total / n) if n else 0,
            "avg_llm_calls_per_interaction": round(llm_calls_total / n, 2) if n else 0,
            "avg_turns_per_session": round(avg_turns, 2),
            "channels": dict(channels.most_common(5)),
            "top_tools": dict(tools_counter.most_common(5)),
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Prompt building
    # ──────────────────────────────────────────────────────────────────────────

    def _build_prompt(
        self,
        agg: dict,
        memory_summary: str,
        prev_audit: dict | None,
        audit_number: int,
    ) -> str:
        metrics_block = "\n".join(f"  {k}: {v}" for k, v in agg.items())

        prev_block = "(this is the first audit)" if not prev_audit else (
            f"Previous diagnosis: {prev_audit.get('diagnosis', '?')}\n"
            f"Previous hypotheses: {json.dumps(prev_audit.get('hypotheses', []), ensure_ascii=False)}\n"
            f"Previous immediate_actions: {json.dumps(prev_audit.get('immediate_actions', []), ensure_ascii=False)}"
        )

        return f"""You are a self-improvement engine analysing your own interaction data. Produce a structured audit.

## Audit #{audit_number}
### Interaction Metrics (last {agg['total_interactions']} interactions, {agg['period']})
{metrics_block}

### Memory State
{memory_summary or "(no structured memories yet)"}

### Previous Audit
{prev_block}

---

Analyse the metrics above and respond with ONLY valid JSON with these keys:

"diagnosis": string — What patterns do you see? What's working? What isn't?

"hypotheses": list of objects, each with:
  - "id": string (e.g. "hyp_001")
  - "priority_score": float (0-10, higher = more important)
  - "statement": string — if we change X, metric Y should improve
  - "requires_code_change": boolean — true if this needs editing Python source files
  - "target_file": string (if requires_code_change) — e.g. "nanobot/agent/loop.py"
  - "change_description": string — what specifically to change
  - "expected_metric": string — what improvement to measure
  - "evaluation_after_n_interactions": integer

"immediate_actions": list of objects (things you can do RIGHT NOW without code changes), each:
  - "type": "memory_note" | "focus_note"
  - "content": string
  (memory_note: adds a note to MEMORY.md; focus_note: writes to workspace/FOCUS.md)

"observations": string — anything else worth noting for the next audit

Rules:
- Be CONCISE. Keep diagnosis under 200 words. Keep each hypothesis statement under 50 words.
- Max 3 hypotheses. Max 2 immediate_actions.
- Be specific and honest. If the data is too sparse to diagnose, say so.
- Keep hypotheses small and testable (one thing at a time).
- Only mark requires_code_change=true if the improvement genuinely needs code edits.
- Return {{}} for any key if nothing applicable.
- Your TOTAL response must be valid JSON under 2000 tokens. Brevity is critical."""

    # ──────────────────────────────────────────────────────────────────────────
    # Action execution
    # ──────────────────────────────────────────────────────────────────────────

    def _execute_immediate_actions(self, actions: list[dict]) -> None:
        """Execute memory/focus notes that don't require code changes."""
        from nanobot.agent.memory import MemoryStore

        for action in actions:
            try:
                kind = action.get("type")
                content = action.get("content", "").strip()
                if not content:
                    continue

                if kind == "memory_note":
                    # Append to the legacy MEMORY.md section
                    memory = MemoryStore(self.workspace)
                    existing = memory.read_long_term()
                    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                    note = f"\n\n<!-- self-audit note {timestamp} -->\n{content}"
                    memory.write_long_term(existing + note)
                    logger.debug(f"Self-audit: wrote memory_note")

                elif kind == "focus_note":
                    focus_file = self.workspace / "FOCUS.md"
                    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
                    with open(focus_file, "a", encoding="utf-8") as f:
                        f.write(f"\n## {timestamp}\n{content}\n")
                    logger.debug(f"Self-audit: wrote focus_note to FOCUS.md")

            except Exception as e:
                logger.debug(f"Self-audit immediate_action failed: {e}")

    def _write_pending_evolution(
        self,
        code_hypotheses: list[dict],
        baseline_metrics: dict,
        audit_number: int,
    ) -> None:
        """Write the top code-change hypothesis to experiments/pending-evolution.json (sorted by priority_score).

        The standby instance's heartbeat will check this file and implement the change
        using the evolve skill.  Only one pending evolution at a time.
        """
        pending_file = self.experiments_dir / "pending-evolution.json"

        # Don't overwrite an existing pending evolution that hasn't been picked up yet
        if pending_file.exists():
            try:
                existing = json.loads(pending_file.read_text(encoding="utf-8"))
                if existing.get("status") == "pending":
                    logger.info("Self-audit: pending evolution already queued, skipping new write.")
                    return
            except Exception:
                pass

        if not code_hypotheses:
            return

        # Sort by priority_score desc, take top
        code_hypotheses.sort(key=lambda h: h.get('priority_score', 0), reverse=True)
        hyp = code_hypotheses[0]

        pending = {
            "status": "pending",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "audit_number": audit_number,
            "hypothesis_id": hyp.get("id", f"hyp_{audit_number}_001"),
            "priority_score": hyp.get("priority_score", 0),
            "statement": hyp.get("statement", ""),
            "target_file": hyp.get("target_file", ""),
            "change_description": hyp.get("change_description", ""),
            "expected_metric": hyp.get("expected_metric", ""),
            "evaluation_after_n_interactions": hyp.get("evaluation_after_n_interactions", 20),
            "baseline_metrics": {
                k: baseline_metrics[k]
                for k in ("correction_rate", "tool_failure_rate", "avg_turns_per_session",
                          "avg_elapsed_ms", "total_interactions")
                if k in baseline_metrics
            },
        }
        pending_file.write_text(json.dumps(pending, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info(f"Self-audit: wrote pending evolution → {hyp.get('id')} (score={hyp.get('priority_score')}) ({hyp.get('target_file')})")

    # ──────────────────────────────────────────────────────────────────────────
    # Persistence helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _load_last_audit(self) -> dict | None:
        state = self._load_state()
        last_file = state.get("last_audit_file")
        if not last_file:
            return None
        p = Path(last_file)
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                return None
        return None

    def _save_audit_report(self, result: dict, audit_number: int) -> None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        report_file = self.audit_dir / f"audit_{ts}_#{audit_number}.json"
        report_file.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        # Update state so we know where the last audit is
        state = self._load_state()
        state["last_audit_file"] = str(report_file)
        self._write_state(state)

    def _append_evolution_log(self, result: dict, audit_number: int) -> None:
        meta = result.get("_meta", {})
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        hypotheses = result.get("hypotheses", [])
        code_hyps = [h for h in hypotheses if h.get("requires_code_change")]
        actions = result.get("immediate_actions", [])

        entry = f"""
## Audit #{audit_number} — {timestamp}

**Diagnosis**: {result.get("diagnosis", "(none)")}

**Hypotheses** ({len(hypotheses)} total, {len(code_hyps)} requiring code changes):
"""
        for h in hypotheses:
            marker = "🔧" if h.get("requires_code_change") else "💡"
            entry += f"- {marker} [{h.get('id', '?')}] {h.get('statement', '')}\n"

        if actions:
            entry += f"\n**Immediate actions** ({len(actions)}):\n"
            for a in actions:
                entry += f"- [{a.get('type', '?')}] {a.get('content', '')[:80]}\n"

        if result.get("observations"):
            entry += f"\n**Observations**: {result['observations']}\n"

        with open(self.evolution_log, "a", encoding="utf-8") as f:
            f.write(entry + "\n---\n")

    def _load_state(self) -> dict:
        if self._state_file.exists():
            try:
                return json.loads(self._state_file.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {}

    def _save_state(self, audit_count: int) -> None:
        state = self._load_state()
        state["audit_count"] = audit_count
        state["last_audit_signal_count"] = self._count_signals()
        state["last_audit_at"] = datetime.now(timezone.utc).isoformat()
        self._write_state(state)

    def _write_state(self, state: dict) -> None:
        self._state_file.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")

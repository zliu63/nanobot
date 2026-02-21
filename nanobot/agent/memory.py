"""Memory system for persistent agent memory.

Three-tier architecture:
- memories.jsonl  : structured entries with decay metadata (ground truth)
- MEMORY.md       : human-readable mirror of active memories
- HISTORY.md      : append-only grep-searchable event log (unchanged)
"""

import json
import math
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from nanobot.utils.helpers import ensure_dir


class MemoryEntry:
    """A single structured memory entry with Ebbinghaus-style decay."""

    def __init__(
        self,
        content: str,
        category: str = "general",
        tags: list[str] | None = None,
        decay_rate: float = 0.02,
        strength: float = 1.0,
        mem_id: str | None = None,
        created_at: str | None = None,
        last_accessed: str | None = None,
        access_count: int = 0,
        status: str = "active",
    ):
        now_iso = datetime.now(timezone.utc).isoformat()
        self.id = mem_id or f"mem_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        self.content = content
        self.category = category
        self.tags = tags or []
        self.created_at = created_at or now_iso
        self.last_accessed = last_accessed or now_iso
        self.access_count = access_count
        self.strength = strength
        self.decay_rate = decay_rate
        self.status = status

    def compute_strength(self) -> float:
        """Compute current strength using exponential decay + access bonus."""
        now = datetime.now(timezone.utc)
        try:
            last = datetime.fromisoformat(self.last_accessed)
        except (ValueError, TypeError):
            last = now
        days_elapsed = max(0.0, (now - last).total_seconds() / 86400)
        decayed = self.strength * math.exp(-self.decay_rate * days_elapsed)
        access_bonus = math.log1p(self.access_count) * 0.05
        return min(1.0, max(0.0, decayed + access_bonus))

    def touch(self) -> None:
        """Reinforce memory on access (resets the decay clock)."""
        self.last_accessed = datetime.now(timezone.utc).isoformat()
        self.access_count += 1
        # Re-anchor strength to current computed value then boost
        self.strength = min(1.0, self.compute_strength() + 0.2)

    @property
    def tier(self) -> str:
        """active / archived / compressed based on current strength."""
        s = self.compute_strength()
        if s > 0.5:
            return "active"
        elif s > 0.2:
            return "archived"
        return "compressed"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "category": self.category,
            "tags": self.tags,
            "created_at": self.created_at,
            "last_accessed": self.last_accessed,
            "access_count": self.access_count,
            "strength": self.strength,
            "decay_rate": self.decay_rate,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "MemoryEntry":
        return cls(
            content=d["content"],
            category=d.get("category", "general"),
            tags=d.get("tags", []),
            decay_rate=d.get("decay_rate", 0.02),
            strength=d.get("strength", 1.0),
            mem_id=d.get("id"),
            created_at=d.get("created_at"),
            last_accessed=d.get("last_accessed"),
            access_count=d.get("access_count", 0),
            status=d.get("status", "active"),
        )


class MemoryStore:
    MAX_ACTIVE = 500
    MAX_FILE_MB = 10
    """
    Manages all persistent memory:

    Structured path (new):
      workspace/memory/memories.jsonl  — one entry per line, with metadata
      workspace/memory/MEMORY.md       — auto-rebuilt human-readable view

    Legacy path (kept for consolidation compatibility):
      workspace/memory/MEMORY.md       — also written directly by _consolidate_memory
      workspace/memory/HISTORY.md      — append-only grep log (unchanged)
    """

    def __init__(self, workspace: Path):
        self.memory_dir = ensure_dir(workspace / "memory")
        self.memory_file = self.memory_dir / "MEMORY.md"
        self.history_file = self.memory_dir / "HISTORY.md"
        self.memories_jsonl = self.memory_dir / "memories.jsonl"

    # ──────────────────────────────────────────────────────────
    # Structured memory operations
    # ──────────────────────────────────────────────────────────

    def load_memories(self) -> list[MemoryEntry]:
        """Load all entries from memories.jsonl."""
        if not self.memories_jsonl.exists():
            return []
        entries: list[MemoryEntry] = []
        for line in self.memories_jsonl.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(MemoryEntry.from_dict(json.loads(line)))
            except Exception:
                pass
        return entries

    def save_memories(self, entries: list[MemoryEntry]) -> None:
        """Persist all entries to memories.jsonl (full rewrite)."""
        lines = [json.dumps(e.to_dict(), ensure_ascii=False) for e in entries]
        content = "\n".join(lines) + "\n" if lines else ""
        self.memories_jsonl.write_text(content, encoding="utf-8")

    def add_memory(
        self,
        content: str,
        category: str = "general",
        tags: list[str] | None = None,
    ) -> MemoryEntry:
        """Add a new structured memory and rebuild MEMORY.md."""
        entries = self.load_memories()
        entry = MemoryEntry(content=content, category=category, tags=tags or [])
        entries.append(entry)
        self.save_memories(entries)
        self._rebuild_memory_md(entries)
        return entry

    def update_memory(self, mem_id: str, **kwargs: Any) -> bool:
        """Update fields on an existing memory entry."""
        entries = self.load_memories()
        for e in entries:
            if e.id == mem_id:
                for k, v in kwargs.items():
                    if hasattr(e, k):
                        setattr(e, k, v)
                self.save_memories(entries)
                self._rebuild_memory_md(entries)
                return True
        return False

    def delete_memory(self, mem_id: str) -> bool:
        """Soft-delete a memory entry (sets status=deleted)."""
        return self.update_memory(mem_id, status="deleted")

    def get_active_memories(self) -> list[MemoryEntry]:
        """Return active-tier memories (computed strength > 0.5, not deleted)."""
        return [
            e for e in self.load_memories()
            if e.status != "deleted" and e.compute_strength() > 0.5
        ]

    def get_archived_memories(self) -> list[MemoryEntry]:
        """Return archived-tier memories (0.2 < strength ≤ 0.5, not deleted)."""
        return [
            e for e in self.load_memories()
            if e.status != "deleted" and 0.2 < e.compute_strength() <= 0.5
        ]

    def build_memory_summary(self) -> str:
        """One-line-per-entry summary for use in audit prompts."""
        entries = [e for e in self.load_memories() if e.status != "deleted"]
        if not entries:
            return ""
        active = [e for e in entries if e.compute_strength() > 0.5]
        archived = [e for e in entries if 0.2 < e.compute_strength() <= 0.5]
        lines = [f"Total: {len(entries)} memories ({len(active)} active, {len(archived)} archived)"]
        by_cat: dict[str, list[MemoryEntry]] = defaultdict(list)
        for e in active:
            by_cat[e.category].append(e)
        for cat, mems in sorted(by_cat.items()):
            lines.append(f"\n[{cat}] {len(mems)} entries:")
            for m in mems[:4]:
                lines.append(f"  - [{m.id}] {m.content[:100]}")
        if archived:
            lines.append(f"\n[archived] {len(archived)} entries (strength 0.2-0.5, available on request)")
        return "\n".join(lines)

    def _rebuild_memory_md(self, entries: list[MemoryEntry]) -> None:
        """Regenerate MEMORY.md from the active entries in memories.jsonl.

        Note: this is called after structured memory operations. When the legacy
        _consolidate_memory path also writes to MEMORY.md, both coexist: the
        structured section is prepended, legacy content follows.
        """
        active = [e for e in entries if e.status != "deleted" and e.compute_strength() > 0.5]
        if not active:
            return  # Don't wipe legacy content if no structured memories exist yet

        by_cat: dict[str, list[MemoryEntry]] = defaultdict(list)
        for e in active:
            by_cat[e.category].append(e)

        lines: list[str] = ["# Structured Memories\n"]
        for cat, mems in sorted(by_cat.items()):
            lines.append(f"## {cat}")
            for m in mems:
                tags_str = f" `{' '.join(m.tags)}`" if m.tags else ""
                lines.append(f"- [{m.id}] {m.content}{tags_str}")
            lines.append("")

        structured_block = "\n".join(lines)

        # Preserve any legacy MEMORY.md content that isn't our structured block
        legacy = ""
        if self.memory_file.exists():
            existing = self.memory_file.read_text(encoding="utf-8")
            if "# Structured Memories" in existing:
                # Strip the old structured block, keep everything after it
                parts = existing.split("# Long-term Memory", 1)
                legacy = "# Long-term Memory" + parts[1] if len(parts) > 1 else ""
            elif existing.strip():
                legacy = existing

        combined = structured_block + ("\n\n" + legacy if legacy.strip() else "")
        self.memory_file.write_text(combined, encoding="utf-8")

    # ──────────────────────────────────────────────────────────
    # Legacy-compatible interface (used by _consolidate_memory in loop.py)
    # ──────────────────────────────────────────────────────────

    def read_long_term(self) -> str:
        """Read full MEMORY.md content."""
        if self.memory_file.exists():
            return self.memory_file.read_text(encoding="utf-8")
        return ""

    def write_long_term(self, content: str) -> None:
        """Overwrite MEMORY.md directly (legacy consolidation path)."""
        self.memory_file.write_text(content, encoding="utf-8")

    def append_history(self, entry: str) -> None:
        """Append an entry to HISTORY.md."""
        with open(self.history_file, "a", encoding="utf-8") as f:
            f.write(entry.rstrip() + "\n\n")

    def prune_old_memories(self) -> None:
        """hyp_010: Auto-archive old technical_fact (>48h create+access) and project_context (>72h create).
        P0: Hard cap active memories + file size."""
        entries = self.load_memories()
        now = datetime.now(timezone.utc)
        pruned = 0
        for e in entries:
            if e.status == "deleted":
                continue
            try:
                create_time = datetime.fromisoformat(e.created_at)
                access_time = datetime.fromisoformat(e.last_accessed)
                age_hours = (now - min(create_time, access_time)).total_seconds() / 3600
                if e.category == "technical_fact" and age_hours > 48:
                    e.status = "archived"
                    pruned += 1
                elif e.category == "project_context" and (now - create_time).total_seconds() / 3600 > 72:
                    e.status = "archived"
                    pruned += 1
            except ValueError:
                pass  # Invalid timestamps skip
        
        # Hard cap: LRU evict active (strength >0.5)
        active = [e for e in entries if e.status != "deleted" and e.compute_strength() > 0.5]
        if len(active) > self.MAX_ACTIVE:
            active.sort(key=lambda e: e.last_accessed or e.created_at)
            excess = active[:-self.MAX_ACTIVE]
            for e in excess:
                e.status = "archived"
                pruned += 1
            self.save_memories(entries)
            self._rebuild_memory_md(entries)
        
        # Enforce file size
        self._enforce_file_size()
        
        print(f"Pruned {pruned} old memories (active now: {len(self.get_active_memories())})")  # No logger dep

    def _enforce_file_size(self) -> None:
        """Enforce max file size by truncating oldest 20% if exceeded."""
        if not self.memories_jsonl.exists():
            return
        try:
            size_mb = self.memories_jsonl.stat().st_size / (1024 ** 2)
            if size_mb > self.MAX_FILE_MB:
                # Backup first
                from datetime import datetime
                archive_name = f"memories_archive_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
                archive_path = self.memory_dir / archive_name
                self.memories_jsonl.copy(archive_path)
                print(f"Memory file capped at {self.MAX_FILE_MB}MB, archived to {archive_name}")
                
                # Truncate to last 80%
                with open(self.memories_jsonl, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                keep = lines[-int(len(lines) * 0.8):]
                with open(self.memories_jsonl, 'w', encoding='utf-8') as f:
                    f.writelines(keep)
        except Exception as e:
            print(f"File size enforcement failed: {e}")
    
    def get_memory_context(self) -> str:
        """Build the memory section for the system prompt.

        Includes both the MEMORY.md content (legacy + structured mirror)
        and a compact listing of active structured entries.
        """
        parts: list[str] = []

        # Primary: MEMORY.md (contains legacy facts + structured mirror)
        long_term = self.read_long_term()
        if long_term:
            parts.append(f"## Long-term Memory\n{long_term}")

        # Secondary: structured memories not already reflected in MEMORY.md
        # (only include if memories.jsonl exists and has content beyond what's in MEMORY.md)
        active = self.get_active_memories()
        archived = self.get_archived_memories()
        if active and "# Structured Memories" not in long_term:
            # Structured block not yet in MEMORY.md — show it directly
            by_cat: dict[str, list[MemoryEntry]] = defaultdict(list)
            for e in active:
                by_cat[e.category].append(e)
            lines: list[str] = ["## Structured Memories (active)"]
            for cat, mems in sorted(by_cat.items()):
                lines.append(f"### {cat}")
                for m in mems:
                    lines.append(f"- [{m.id}] {m.content}")
            parts.append("\n".join(lines))

        if archived:
            parts.append(
                f"## Archived Memories ({len(archived)} entries)\n"
                "These exist but are less frequently accessed. "
                "IDs available in memories.jsonl if needed."
            )

        return "\n\n".join(parts)

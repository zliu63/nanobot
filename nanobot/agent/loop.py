"""Agent loop: the core processing engine."""

import asyncio
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger

from nanobot.bus.events import InboundMessage, OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.providers.base import LLMProvider
from nanobot.agent.context import ContextBuilder
from nanobot.agent.tools.defaults import build_default_tools
from nanobot.agent.tools.message import MessageTool
from nanobot.agent.tools.spawn import SpawnTool
from nanobot.agent.tools.cron import CronTool
from nanobot.agent.memory import MemoryStore
from nanobot.agent.self_evolution import SelfAuditEngine
from nanobot.agent.subagent import SubagentManager
from nanobot.utils.helpers import ensure_dir
from nanobot.session.manager import Session, SessionManager


class AgentLoop:
    """
    The agent loop is the core processing engine.

    It:
    1. Receives messages from the bus
    2. Builds context with history, memory, skills
    3. Calls the LLM
    4. Executes tool calls
    5. Sends responses back
    """

    def __init__(
        self,
        bus: MessageBus,
        provider: LLMProvider,
        workspace: Path,
        model: str | None = None,
        max_iterations: int = 20,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        memory_window: int = 50,
        brave_api_key: str | None = None,
        exec_config: "ExecToolConfig | None" = None,
        cron_service: "CronService | None" = None,
        restrict_to_workspace: bool = False,
        session_manager: SessionManager | None = None,
    ):
        from nanobot.config.schema import ExecToolConfig
        from nanobot.cron.service import CronService
        self.bus = bus
        self.provider = provider
        self.workspace = workspace
        self.model = model or provider.get_default_model()
        self.max_iterations = max_iterations
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.memory_window = memory_window
        self.brave_api_key = brave_api_key
        self.exec_config = exec_config or ExecToolConfig()
        self.cron_service = cron_service
        self.restrict_to_workspace = restrict_to_workspace

        self.context = ContextBuilder(workspace)
        self.sessions = session_manager or SessionManager(workspace)
        self.subagents = SubagentManager(
            provider=provider,
            workspace=workspace,
            bus=bus,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            brave_api_key=brave_api_key,
            exec_config=self.exec_config,
            restrict_to_workspace=restrict_to_workspace,
        )
        
        self.tick_counter = 0
        self._running = False
        self.tools = build_default_tools(
            workspace=self.workspace,
            exec_config=self.exec_config,
            restrict_to_workspace=self.restrict_to_workspace,
            brave_api_key=self.brave_api_key,
            include_agent_tools=True,
            bus=self.bus,
            subagents=self.subagents,
            cron_service=self.cron_service,
        )
        self._audit_engine = SelfAuditEngine(workspace)
    
    def _set_tool_context(self, channel: str, chat_id: str) -> None:
        """Update context for all tools that need routing info."""
        if message_tool := self.tools.get("message"):
            if isinstance(message_tool, MessageTool):
                message_tool.set_context(channel, chat_id)

        if spawn_tool := self.tools.get("spawn"):
            if isinstance(spawn_tool, SpawnTool):
                spawn_tool.set_context(channel, chat_id)

        if cron_tool := self.tools.get("cron"):
            if isinstance(cron_tool, CronTool):
                cron_tool.set_context(channel, chat_id)

    async def _run_agent_loop(self, initial_messages: list[dict]) -> tuple[str | None, list[str], int, int]:
        """
        Run the agent iteration loop.

        Args:
            initial_messages: Starting messages for the LLM conversation.

        Returns:
            Tuple of (final_content, tools_used, tool_failures, llm_calls).
        """
        messages = initial_messages
        iteration = 0
        final_content = None
        tools_used: list[str] = []
        tool_failures = 0

        max_iters = min(5 + len(initial_messages)//2, self.max_iterations)
        while iteration < max_iters:
            iteration += 1

            response = await self.provider.chat(
                messages=messages,
                tools=self.tools.get_definitions(),
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            if response.has_tool_calls:
                tool_call_dicts = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments)
                        }
                    }
                    for tc in response.tool_calls
                ]
                messages = self.context.add_assistant_message(
                    messages, response.content, tool_call_dicts,
                    reasoning_content=response.reasoning_content,
                )

                # Execute all tool calls in parallel for independent tools
                async def _execute_tool(tc):
                    args_str = json.dumps(tc.arguments, ensure_ascii=False)
                    logger.info(f"Tool call: {tc.name}({args_str[:200]})")
                    result = await self.tools.execute(tc.name, tc.arguments)
                    return tc, result

                tool_results = await asyncio.gather(
                    *[_execute_tool(tc) for tc in response.tool_calls]
                )

                for tool_call, result in tool_results:
                    tools_used.append(tool_call.name)
                    if isinstance(result, str) and result.startswith("Error"):
                        tool_failures += 1
                    messages = self.context.add_tool_result(
                        messages, tool_call.id, tool_call.name, result
                    )
                messages.append({"role": "user", "content": "Reflect on the results and decide next steps."})
            else:
                final_content = response.content
                if final_content and len(final_content) > 50:
                    break

        return final_content, tools_used, tool_failures, iteration

    async def run(self) -> None:
        """Run the agent loop, processing messages from the bus."""
        self._running = True
        logger.info("Agent loop started")

        while self._running:
            try:
                msg = await asyncio.wait_for(
                    self.bus.consume_inbound(),
                    timeout=1.0
                )
                try:
                    response = await self._process_message(msg)
                    if response:
                        await self.bus.publish_outbound(response)
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    await self.bus.publish_outbound(OutboundMessage(
                        channel=msg.channel,
                        chat_id=msg.chat_id,
                        content=f"Sorry, I encountered an error: {str(e)}"
                    ))
            except asyncio.TimeoutError:
                self.tick_counter += 1
                if self.tick_counter % 300 == 0 and self.cron_service:  # ~5min
                    await self._proactive_tick()
                continue
    
    def stop(self) -> None:
        """Stop the agent loop."""
        self._running = False
        logger.info("Agent loop stopping")
    
    async def _process_message(self, msg: InboundMessage, session_key: str | None = None) -> OutboundMessage | None:
        """
        Process a single inbound message.
        
        Args:
            msg: The inbound message to process.
            session_key: Override session key (used by process_direct).
        
        Returns:
            The response message, or None if no response needed.
        """
        # System messages route back via chat_id ("channel:chat_id")
        if msg.channel == "system":
            return await self._process_system_message(msg)
        
        preview = msg.content[:80] + "..." if len(msg.content) > 80 else msg.content
        logger.info(f"Processing message from {msg.channel}:{msg.sender_id}: {preview}")
        
        key = session_key or msg.session_key
        session = self.sessions.get_or_create(key)
        
        # Handle slash commands
        cmd = msg.content.strip().lower()
        if cmd == "/new":
            # Capture messages before clearing (avoid race condition with background task)
            messages_to_archive = session.messages.copy()
            session.clear()
            self.sessions.save(session)
            self.sessions.invalidate(session.key)

            async def _consolidate_and_cleanup():
                temp_session = Session(key=session.key)
                temp_session.messages = messages_to_archive
                await self._consolidate_memory(temp_session, archive_all=True)

            asyncio.create_task(_consolidate_and_cleanup())
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id,
                                  content="New session started. Memory consolidation in progress.")
        if cmd == "/help":
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id,
                                  content="🐈 nanobot commands:\n/new — Start a new conversation\n/help — Show available commands")
        
        if len(session.messages) > self.memory_window:
            asyncio.create_task(self._consolidate_memory(session))

        # Prune old memories on each interaction
        memory = MemoryStore(self.workspace)
        memory.prune_old_memories()

        self._set_tool_context(msg.channel, msg.chat_id)
        initial_messages = self.context.build_messages(
            history=session.get_history(max_messages=self.memory_window),
            current_message=msg.content,
            media=msg.media if msg.media else None,
            channel=msg.channel,
            chat_id=msg.chat_id,
        )
        t_start = time.perf_counter()
        final_content, tools_used, tool_failures, llm_calls = await self._run_agent_loop(initial_messages)
        elapsed_ms = int((time.perf_counter() - t_start) * 1000)

        if final_content is None:
            final_content = "I've completed processing but have no response to give."

        preview = final_content[:120] + "..." if len(final_content) > 120 else final_content
        logger.info(f"Response to {msg.channel}:{msg.sender_id}: {preview}")

        session.add_message("user", msg.content)
        session.add_message("assistant", final_content,
                            tools_used=tools_used if tools_used else None)
        self.sessions.save(session)

        # Phase 0: signal collection (fire-and-forget, never blocks response)
        asyncio.create_task(self._collect_signals(
            session=session,
            msg=msg,
            final_content=final_content,
            tools_used=tools_used,
            tool_failures=tool_failures,
            llm_calls=llm_calls,
            elapsed_ms=elapsed_ms,
        ))

        # Phase 0: memory reflection (only when conversation is substantive)
        if self._should_reflect(msg.content, session):
            asyncio.create_task(self._reflect_on_memory(session, msg.content, final_content))

        return OutboundMessage(
            channel=msg.channel,
            chat_id=msg.chat_id,
            content=final_content,
            metadata=msg.metadata or {},  # Pass through for channel-specific needs (e.g. Slack thread_ts)
        )
    
    async def _proactive_tick(self):
        """P3 Proactivity: Inject due cron jobs as system messages."""
        try:
            due_jobs = self.cron_service.get_due_proactive()
            for job in due_jobs:
                content = job.payload.message
                # Use the job's delivery channel/chat_id if specified,
                # so the response reaches the correct destination (e.g. Telegram)
                if job.payload.deliver and job.payload.channel:
                    chat_id = f"{job.payload.channel}:{job.payload.to or ''}"
                else:
                    chat_id = f"proactive:{job.id}"
                sys_msg = InboundMessage(
                    channel="system",
                    sender_id="heartbeat",
                    chat_id=chat_id,
                    content=content
                )
                asyncio.create_task(self._process_message(sys_msg))
            if due_jobs:
                logger.info(f"Proactive tick: injected {len(due_jobs)} due jobs")
        except Exception as e:
            logger.warning(f"Proactive tick failed: {e}")
    
    async def _process_system_message(self, msg: InboundMessage) -> OutboundMessage | None:
        """
        Process a system message (e.g., subagent announce).
        
        The chat_id field contains "original_channel:original_chat_id" to route
        the response back to the correct destination.
        """
        logger.info(f"Processing system message from {msg.sender_id}")
        
        # Parse origin from chat_id (format: "channel:chat_id")
        if ":" in msg.chat_id:
            parts = msg.chat_id.split(":", 1)
            origin_channel = parts[0]
            origin_chat_id = parts[1]
        else:
            # Fallback
            origin_channel = "cli"
            origin_chat_id = msg.chat_id
        
        session_key = f"{origin_channel}:{origin_chat_id}"
        session = self.sessions.get_or_create(session_key)
        self._set_tool_context(origin_channel, origin_chat_id)
        initial_messages = self.context.build_messages(
            history=session.get_history(max_messages=self.memory_window),
            current_message=msg.content,
            channel=origin_channel,
            chat_id=origin_chat_id,
        )
        final_content, _, _tf, _lc = await self._run_agent_loop(initial_messages)

        if final_content is None:
            final_content = "Background task completed."
        
        session.add_message("user", f"[System: {msg.sender_id}] {msg.content}")
        session.add_message("assistant", final_content)
        self.sessions.save(session)
        
        return OutboundMessage(
            channel=origin_channel,
            chat_id=origin_chat_id,
            content=final_content
        )
    
    # ──────────────────────────────────────────────────────────────────────
    # Phase 0: Signal collection & memory reflection
    # ──────────────────────────────────────────────────────────────────────

    def _should_reflect(self, user_message: str, session) -> bool:
        """Decide whether to trigger memory reflection after this turn."""
        # Always reflect if correction detected
        if self._detect_correction(user_message):
            return True
        # Reflect after 3+ complete turns (6 messages = 3 user + 3 assistant)
        if len(session.messages) >= 6:
            return True
        return False

    def _detect_correction(self, text: str) -> bool:
        """Heuristic: did the user correct a previous response?"""
        markers = [
            "不对", "错了", "不是", "不对吧", "我说的是", "我的意思是", "你理解错",
            "that's wrong", "not right", "actually,", "no,", "you misunderstood",
            "i meant", "incorrect",
        ]
        lower = text.lower()
        return any(m in lower for m in markers)

    async def _collect_signals(
        self,
        session,
        msg,
        final_content: str,
        tools_used: list[str],
        tool_failures: int,
        llm_calls: int,
        elapsed_ms: int,
    ) -> None:
        """Record interaction quality signals to workspace/signals/signals.jsonl."""
        try:
            signals_dir = ensure_dir(self.workspace / "signals")
            signals_file = signals_dir / "signals.jsonl"

            # Count user messages in the session to estimate turn count
            user_turns = sum(1 for m in session.messages if m.get("role") == "user")

            signal = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "session_id": session.key,
                "signals": {
                    "conversation_turns": user_turns,
                    "user_correction_detected": self._detect_correction(msg.content),
                    "tools_used": sorted(set(tools_used)),
                    "tool_use_count": len(tools_used),
                    "tool_failures": tool_failures,
                    "total_llm_calls": llm_calls,
                    "response_length_chars": len(final_content),
                    "elapsed_ms": elapsed_ms,
                    "channel": msg.channel,
                },
            }
            with open(signals_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(signal, ensure_ascii=False) + "\n")

            # Phase 1: trigger self-audit if enough signals have accumulated
            if self._audit_engine.should_audit():
                asyncio.create_task(self._run_self_audit())

        except Exception as e:
            logger.debug(f"Signal collection failed (non-critical): {e}")

    async def _run_self_audit(self) -> None:
        """Fire-and-forget wrapper around SelfAuditEngine.run_audit()."""
        try:
            result = await self._audit_engine.run_audit(self.provider, self.model)
            if result:
                diag = result.get("diagnosis", "")[:120]
                logger.info(f"Self-audit complete: {diag}")
        except Exception as e:
            logger.debug(f"Self-audit failed (non-critical): {e}")

    async def _reflect_on_memory(
        self,
        session,
        user_message: str,
        assistant_response: str,
    ) -> None:
        """LLM-driven memory reflection: decide what to add/update/delete.

        Runs as a background task after substantive conversations.
        Does a single LLM call, applies JSON operations to memories.jsonl.
        """
        try:
            memory = MemoryStore(self.workspace)

            # Build a compact conversation excerpt (last 3 turns max)
            recent = session.messages[-6:] if len(session.messages) >= 6 else session.messages
            convo_lines: list[str] = []
            for m in recent:
                content = m.get("content", "")
                if content:
                    role = m["role"].upper()
                    convo_lines.append(f"[{role}] {content[:400]}")

            memory_summary = memory.build_memory_summary()

            prompt = f"""You are a memory management agent. Based on the conversation below, decide what structured memories to create, update, or remove.

## Current Structured Memory State
{memory_summary or "(empty — this is the first reflection)"}

## Recent Conversation
{chr(10).join(convo_lines)}

Respond with ONLY valid JSON (no markdown fences). Use these optional keys:
- "add": list of {{"content": str, "category": str, "tags": list[str]}}
- "update": list of {{"id": str, "content": str}}
- "delete": list of memory IDs (string)

Valid categories: user_preference, project_context, technical_fact, relationship, daily_event

Rules:
- Only add facts that are genuinely worth remembering long-term (not transient info)
- Only update if the new information clearly supersedes the old
- Only delete if the existing memory is factually wrong or entirely irrelevant
- Return {{}} if nothing needs to change

Example: {{"add": [{{"content": "User is migrating a Flask app to FastAPI", "category": "project_context", "tags": ["fastapi", "flask", "python"]}}]}}"""

            response = await self.provider.chat(
                messages=[
                    {"role": "system", "content": "You are a memory management agent. Respond only with valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                model=self.model,
                max_tokens=1024,
                temperature=0.3,
            )

            text = (response.content or "").strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

            ops: dict = json.loads(text)

            added = updated = deleted = 0
            for item in ops.get("add", []):
                memory.add_memory(
                    content=item["content"],
                    category=item.get("category", "general"),
                    tags=item.get("tags", []),
                )
                added += 1

            for item in ops.get("update", []):
                if memory.update_memory(item["id"], content=item["content"]):
                    updated += 1

            for mem_id in ops.get("delete", []):
                if memory.delete_memory(str(mem_id)):
                    deleted += 1

            if added + updated + deleted > 0:
                logger.info(f"Memory reflection: +{added} updated={updated} deleted={deleted}")

        except Exception as e:
            logger.debug(f"Memory reflection failed (non-critical): {e}")

    async def _consolidate_memory(self, session, archive_all: bool = False) -> None:
        """Consolidate old messages into MEMORY.md + HISTORY.md.

        Args:
            archive_all: If True, clear all messages and reset session (for /new command).
                       If False, only write to files without modifying session.
        """
        memory = MemoryStore(self.workspace)

        if archive_all:
            old_messages = session.messages
            keep_count = 0
            logger.info(f"Memory consolidation (archive_all): {len(session.messages)} total messages archived")
        else:
            keep_count = self.memory_window // 2
            if len(session.messages) <= keep_count:
                logger.debug(f"Session {session.key}: No consolidation needed (messages={len(session.messages)}, keep={keep_count})")
                return

            messages_to_process = len(session.messages) - session.last_consolidated
            if messages_to_process <= 0:
                logger.debug(f"Session {session.key}: No new messages to consolidate (last_consolidated={session.last_consolidated}, total={len(session.messages)})")
                return

            old_messages = session.messages[session.last_consolidated:-keep_count]
            if not old_messages:
                return
            logger.info(f"Memory consolidation started: {len(session.messages)} total, {len(old_messages)} new to consolidate, {keep_count} keep")

        lines = []
        for m in old_messages:
            if not m.get("content"):
                continue
            tools = f" [tools: {', '.join(m['tools_used'])}]" if m.get("tools_used") else ""
            lines.append(f"[{m.get('timestamp', '?')[:16]}] {m['role'].upper()}{tools}: {m['content']}")
        conversation = "\n".join(lines)
        current_memory = memory.read_long_term()

        prompt = f"""You are a memory consolidation agent. Process this conversation and return a JSON object with exactly two keys:

1. "history_entry": A paragraph (2-5 sentences) summarizing the key events/decisions/topics. Start with a timestamp like [YYYY-MM-DD HH:MM]. Include enough detail to be useful when found by grep search later.

2. "memory_update": The updated long-term memory content. Add any new facts: user location, preferences, personal info, habits, project context, technical decisions, tools/services used. If nothing new, return the existing content unchanged.

## Current Long-term Memory
{current_memory or "(empty)"}

## Conversation to Process
{conversation}

Respond with ONLY valid JSON, no markdown fences."""

        try:
            response = await self.provider.chat(
                messages=[
                    {"role": "system", "content": "You are a memory consolidation agent. Respond only with valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                model=self.model,
            )
            text = (response.content or "").strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            result = json.loads(text)

            if entry := result.get("history_entry"):
                memory.append_history(entry)
            if update := result.get("memory_update"):
                if update != current_memory:
                    memory.write_long_term(update)

            if archive_all:
                session.last_consolidated = 0
            else:
                session.last_consolidated = len(session.messages) - keep_count
            logger.info(f"Memory consolidation done: {len(session.messages)} messages, last_consolidated={session.last_consolidated}")
        except Exception as e:
            logger.error(f"Memory consolidation failed: {e}")

    async def process_direct(
        self,
        content: str,
        session_key: str = "cli:direct",
        channel: str = "cli",
        chat_id: str = "direct",
    ) -> str:
        """
        Process a message directly (for CLI or cron usage).
        
        Args:
            content: The message content.
            session_key: Session identifier (overrides channel:chat_id for session lookup).
            channel: Source channel (for tool context routing).
            chat_id: Source chat ID (for tool context routing).
        
        Returns:
            The agent's response.
        """
        msg = InboundMessage(
            channel=channel,
            sender_id="user",
            chat_id=chat_id,
            content=content
        )
        
        response = await self._process_message(msg, session_key=session_key)
        return response.content if response else ""

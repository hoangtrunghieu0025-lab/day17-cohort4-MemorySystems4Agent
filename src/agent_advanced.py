from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

from agent_baseline import BASE_SYSTEM_PROMPT, live_credentials_available, message_text
from config import LabConfig, load_config
from memory_store import (
    CompactMemoryManager,
    UserProfileStore,
    answer_from_facts,
    estimate_tokens,
    extract_profile_candidates,
)
from model_provider import build_chat_model

try:  # module-level on purpose: tool annotations are resolved from this module's globals
    from langchain.tools import ToolRuntime
except ImportError:  # offline mode works without LangChain installed
    ToolRuntime = None

ADVANCED_SYSTEM_PROMPT = (
    BASE_SYSTEM_PROMPT
    + " Bạn có hồ sơ người dùng bền vững (User.md) và bản tóm tắt hội thoại cũ; ưu tiên thông tin mới nhất"
    " trong hồ sơ, không dùng giá trị trong mục Superseded."
)


@dataclass
class AgentContext:
    user_id: str
    memory_path: str


class AdvancedAgent:
    """Agent B / Advanced Agent.

    Memory layers:
    1. short-term: recent messages of the current thread (kept verbatim by `CompactMemoryManager`)
    2. persistent: `User.md` per user (survives new threads / sessions)
    3. compact: summary of older messages once the thread exceeds the token threshold
    """

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.profile_store = UserProfileStore(self.config.state_dir / "profiles")
        self.compact_memory = CompactMemoryManager(
            threshold_tokens=self.config.compact_threshold_tokens,
            keep_messages=self.config.compact_keep_messages,
        )
        self.thread_tokens: dict[str, int] = {}
        self.thread_prompt_tokens: dict[str, int] = {}
        self.langchain_agent = self._maybe_build_langchain_agent()

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Route between the deterministic offline path and the live LangChain path."""

        if self.langchain_agent is not None and not self.force_offline:
            return self._reply_live(user_id, thread_id, message)
        return self._reply_offline(user_id, thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        return self.thread_tokens.get(thread_id, 0)

    def prompt_token_usage(self, thread_id: str) -> int:
        return self.thread_prompt_tokens.get(thread_id, 0)

    def memory_file_size(self, user_id: str) -> int:
        return self.profile_store.file_size(user_id)

    def compaction_count(self, thread_id: str) -> int:
        return self.compact_memory.compaction_count(thread_id)

    def _begin_turn(self, user_id: str, thread_id: str, message: str) -> tuple[list[str], int]:
        """Steps 1-4 of a turn: extract facts -> User.md -> short-term/compact memory -> prompt load."""

        changes = self.profile_store.apply_candidates(user_id, extract_profile_candidates(message))
        self.compact_memory.append(thread_id, "user", message)
        return sorted(set(changes)), self._estimate_prompt_context_tokens(user_id, thread_id)

    def _end_turn(
        self, thread_id: str, message: str, response: str, prompt_tokens: int, changes: list[str]
    ) -> dict[str, Any]:
        if changes:  # tool-call trace: persisting memory has a visible (small) token cost
            response = f"{response} [User.md đã cập nhật: {', '.join(changes)}]"
        self.compact_memory.append(thread_id, "assistant", response)
        agent_tokens = estimate_tokens(message) + estimate_tokens(response)
        self.thread_tokens[thread_id] = self.thread_tokens.get(thread_id, 0) + agent_tokens
        self.thread_prompt_tokens[thread_id] = self.thread_prompt_tokens.get(thread_id, 0) + prompt_tokens
        return {
            "response": response,
            "agent_tokens": agent_tokens,
            "prompt_tokens": prompt_tokens,
            "compactions": self.compaction_count(thread_id),
            "memory_updates": changes,
        }

    def _reply_offline(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        changes, prompt_tokens = self._begin_turn(user_id, thread_id, message)
        response = self._offline_response(user_id, thread_id, message)
        return self._end_turn(thread_id, message, response, prompt_tokens, changes)

    def _reply_live(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        changes, prompt_tokens = self._begin_turn(user_id, thread_id, message)
        result = self.langchain_agent.invoke(
            {"messages": [{"role": "user", "content": message}]},
            config={"configurable": {"thread_id": thread_id}},
            context=AgentContext(user_id=user_id, memory_path=str(self.profile_store.path_for(user_id))),
        )
        response = message_text(result["messages"][-1].content)
        return self._end_turn(thread_id, message, response, prompt_tokens, changes)

    def _estimate_prompt_context_tokens(self, user_id: str, thread_id: str) -> int:
        """Context carried into one turn: system prompt + User.md (lean view) + summary + kept messages."""

        context = self.compact_memory.context(thread_id)
        messages: list[dict[str, str]] = context["messages"]  # type: ignore[assignment]
        return (
            estimate_tokens(ADVANCED_SYSTEM_PROMPT)
            + estimate_tokens(self.profile_store.prompt_view(user_id))
            + estimate_tokens(str(context["summary"]))
            + sum(estimate_tokens(m["content"]) for m in messages)
        )

    def _offline_response(self, user_id: str, thread_id: str, message: str) -> str:
        """Answer recall questions from persisted `User.md` facts (works in a brand-new thread)."""

        return answer_from_facts(message, self.profile_store.facts(user_id))

    def _maybe_build_langchain_agent(self):
        """Wire a live agent: User.md tools + dynamic prompt injection + summarization middleware.

        Returns None (=> offline mode) when forced offline, when dependencies are missing, or when
        no credentials are configured.
        """

        if self.force_offline or not live_credentials_available(self.config):
            return None
        try:
            from langchain.agents import create_agent
            from langchain.agents.middleware import SummarizationMiddleware, dynamic_prompt
            from langchain.tools import tool
            from langgraph.checkpoint.memory import InMemorySaver

            store = self.profile_store
            model = build_chat_model(self.config.model)

            @tool
            def read_user_profile(runtime: ToolRuntime[AgentContext]) -> str:
                """Read the persistent User.md profile of the current user."""

                return store.read_text(runtime.context.user_id)

            @tool
            def edit_user_profile(search_text: str, replacement: str, runtime: ToolRuntime[AgentContext]) -> str:
                """Replace one occurrence of `search_text` in the current user's User.md."""

                changed = store.edit_text(runtime.context.user_id, search_text, replacement)
                return "updated" if changed else "search_text not found"

            @dynamic_prompt
            def profile_prompt(request) -> str:
                profile = store.prompt_view(request.runtime.context.user_id)
                return f"{ADVANCED_SYSTEM_PROMPT}\n\n{profile}" if profile else ADVANCED_SYSTEM_PROMPT

            summarizer = SummarizationMiddleware(
                model=model,
                trigger=("tokens", self.config.compact_threshold_tokens),
                keep=("messages", self.config.compact_keep_messages),
            )
            return create_agent(
                model,
                tools=[read_user_profile, edit_user_profile],
                middleware=[profile_prompt, summarizer],
                checkpointer=InMemorySaver(),
                context_schema=AgentContext,
            )
        except Exception as exc:  # missing SDK / bad config -> fall back instead of crashing the lab
            warnings.warn(f"Advanced live agent unavailable, using offline mode: {exc}")
            return None

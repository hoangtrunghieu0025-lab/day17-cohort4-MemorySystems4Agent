from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

from config import LabConfig, load_config
from memory_store import answer_from_facts, estimate_tokens, facts_from_messages
from model_provider import build_chat_model

BASE_SYSTEM_PROMPT = "Bạn là trợ lý AI hữu ích. Trả lời bằng tiếng Việt, rõ ràng."


@dataclass
class SessionState:
    messages: list[dict[str, str]] = field(default_factory=list)
    token_usage: int = 0
    prompt_tokens_processed: int = 0


def live_credentials_available(config: LabConfig) -> bool:
    """A live agent needs an API key (ollama only needs a reachable server, assumed local)."""

    model = config.model
    return model.provider == "ollama" or bool(model.api_key)


def message_text(content: Any) -> str:
    """Normalize LangChain message content (str or list of blocks) to plain text."""

    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block.get("text", "") if isinstance(block, dict) else str(block) for block in content
        )
    return str(content)


class BaselineAgent:
    """Agent A: within-thread memory only.

    - the whole thread is replayed in every prompt (no compaction)
    - no persistent `User.md`: a new `thread_id` starts from zero, whatever the `user_id`
    """

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.sessions: dict[str, SessionState] = {}
        self.langchain_agent = self._maybe_build_langchain_agent()

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Return the response plus token accounting. `user_id` is deliberately ignored."""

        if self.langchain_agent is not None and not self.force_offline:
            return self._reply_live(thread_id, message)
        return self._reply_offline(thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        """Cumulative tokens exchanged with the user (user message + agent reply) in one thread."""

        return self._session(thread_id).token_usage

    def prompt_token_usage(self, thread_id: str) -> int:
        """Cumulative prompt tokens processed: system prompt + full thread history, summed per turn."""

        return self._session(thread_id).prompt_tokens_processed

    def compaction_count(self, thread_id: str) -> int:
        # Baseline has no compact memory.
        return 0

    def _session(self, thread_id: str) -> SessionState:
        return self.sessions.setdefault(thread_id, SessionState())

    def _begin_turn(self, thread_id: str, message: str) -> tuple[SessionState, int]:
        session = self._session(thread_id)
        session.messages.append({"role": "user", "content": message})
        prompt_tokens = estimate_tokens(BASE_SYSTEM_PROMPT) + sum(
            estimate_tokens(m["content"]) for m in session.messages
        )
        return session, prompt_tokens

    def _end_turn(self, session: SessionState, message: str, response: str, prompt_tokens: int) -> dict[str, Any]:
        session.messages.append({"role": "assistant", "content": response})
        agent_tokens = estimate_tokens(message) + estimate_tokens(response)
        session.token_usage += agent_tokens
        session.prompt_tokens_processed += prompt_tokens
        return {
            "response": response,
            "agent_tokens": agent_tokens,
            "prompt_tokens": prompt_tokens,
            "compactions": 0,
        }

    def _reply_offline(self, thread_id: str, message: str) -> dict[str, Any]:
        """Deterministic behaviour: answers only from facts found in *this thread's* user messages."""

        session, prompt_tokens = self._begin_turn(thread_id, message)
        thread_facts = facts_from_messages([m["content"] for m in session.messages if m["role"] == "user"])
        response = answer_from_facts(message, thread_facts)
        return self._end_turn(session, message, response, prompt_tokens)

    def _reply_live(self, thread_id: str, message: str) -> dict[str, Any]:
        session, prompt_tokens = self._begin_turn(thread_id, message)
        result = self.langchain_agent.invoke(
            {"messages": [{"role": "user", "content": message}]},
            config={"configurable": {"thread_id": thread_id}},
        )
        response = message_text(result["messages"][-1].content)
        return self._end_turn(session, message, response, prompt_tokens)

    def _maybe_build_langchain_agent(self):
        """Build a real LangChain agent (short-term memory via InMemorySaver) when possible.

        Returns None (=> offline mode) when forced offline, when dependencies are missing, or when
        no credentials are configured.
        """

        if self.force_offline or not live_credentials_available(self.config):
            return None
        try:
            from langchain.agents import create_agent
            from langgraph.checkpoint.memory import InMemorySaver

            return create_agent(
                build_chat_model(self.config.model),
                tools=[],
                system_prompt=BASE_SYSTEM_PROMPT,
                checkpointer=InMemorySaver(),
            )
        except Exception as exc:  # missing SDK / bad config -> fall back instead of crashing the lab
            warnings.warn(f"Baseline live agent unavailable, using offline mode: {exc}")
            return None

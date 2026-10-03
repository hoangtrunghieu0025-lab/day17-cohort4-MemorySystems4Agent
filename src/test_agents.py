from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

import agent_advanced
import agent_baseline
from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from benchmark import heuristic_quality, recall_points, run_suite
from config import load_config
from memory_store import (
    MAX_INTERESTS,
    Candidate,
    CompactMemoryManager,
    UserProfileStore,
    estimate_tokens,
    extract_profile_updates,
    summarize_messages,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def make_config(tmp_path: Path):
    """Isolated config: state under tmp_path and a small compact threshold so compaction triggers fast."""

    config = load_config(tmp_path)
    return replace(config, compact_threshold_tokens=120, compact_keep_messages=4)


def _stress_conversation() -> dict:
    return json.loads((DATA_DIR / "advanced_long_context.json").read_text(encoding="utf-8"))[0]


# --- User.md -----------------------------------------------------------------------------------


def test_user_markdown_read_write_edit(tmp_path: Path) -> None:
    store = UserProfileStore(tmp_path / "profiles")

    # reading a missing profile returns a default markdown profile, without creating a file
    assert "User Profile: alice" in store.read_text("alice")
    assert store.file_size("alice") == 0

    # write
    path = store.write_text("alice", "# User Profile: alice\n\n- name: Alice\n")
    assert path.name == "User.md" and path.exists()
    assert store.read_text("alice").endswith("- name: Alice\n")
    assert store.file_size("alice") == path.stat().st_size > 0

    # edit: replaces one occurrence, reports whether anything changed
    assert store.edit_text("alice", "Alice", "Alicia") is True
    assert "Alicia" in store.read_text("alice")
    assert store.edit_text("alice", "does-not-exist", "x") is False
    assert store.edit_text("alice", "", "x") is False

    # user ids are sanitized: no path traversal out of the profiles directory
    evil = store.path_for("../../etc/passwd")
    assert (tmp_path / "profiles") in evil.parents


def test_user_markdown_round_trips_structured_facts(tmp_path: Path) -> None:
    store = UserProfileStore(tmp_path / "profiles")
    store.upsert_fact("bob", "name", "Bob")
    store.upsert_fact("bob", "interest", "Python")
    store.upsert_fact("bob", "interest", "Python")  # repeated mention, not a duplicate line
    assert store.facts("bob") == {"name": "Bob", "interest": "Python"}
    assert store.read_text("bob").count("- Python") == 1

    # a hand edit through edit_text is picked up by the structured reader
    assert store.edit_text("bob", "- name: Bob", "- name: Robert")
    assert store.facts("bob")["name"] == "Robert"


# --- extraction: corrections, noise, questions, confidence --------------------------------------


def test_extraction_handles_corrections_and_negation() -> None:
    updates = extract_profile_updates("À, mình đính chính: giờ mình đang ở Huế chứ không còn ở Đà Nẵng mỗi ngày nữa.")
    assert updates["location"] == "Huế"

    updates = extract_profile_updates("Mình không còn làm backend engineer nữa, giờ chuyển sang MLOps engineer.")
    assert updates["profession"] == "MLOps engineer"


def test_extraction_ignores_questions_jokes_and_noise() -> None:
    assert extract_profile_updates("Hiện tại mình đang ở đâu?") == {}
    assert extract_profile_updates("Bạn thử nhớ lại xem đồ uống yêu thích của mình là gì.") == {}
    assert extract_profile_updates("Món ăn yêu thích của mình là gì và mình nuôi con gì?") == {}
    # a joke and a meeting trip are not facts about the user
    assert "profession" not in extract_profile_updates("Mình đùa là sẽ chuyển sang product manager cho vui.")
    assert "location" not in extract_profile_updates("Hà Nội chỉ là nơi mình vừa bay ra họp hai ngày.")
    # "corgi tên Bơ" must not overwrite the user's own name
    updates = extract_profile_updates("Mình nuôi một bé corgi tên Bơ.")
    assert "name" not in updates and updates["pet"] == "corgi tên Bơ"


def test_confidence_threshold_blocks_hedged_facts(tmp_path: Path) -> None:
    # "có thể" before the fact lowers confidence under the threshold -> nothing is persisted
    assert "profession" not in extract_profile_updates("Có thể mình sẽ làm data engineer sau này.")
    assert extract_profile_updates("Mình làm data engineer.")["profession"] == "data engineer"


def test_weak_statement_does_not_overwrite_confident_fact(tmp_path: Path) -> None:
    store = UserProfileStore(tmp_path / "profiles")
    store.upsert_fact("u", "location", "Huế", confidence=1.0)
    store.upsert_fact("u", "location", "Hà Nội", confidence=0.65)  # much weaker -> ignored
    assert store.facts("u")["location"] == "Huế"
    store.upsert_fact("u", "location", "Đà Nẵng", confidence=0.95)  # confident correction -> wins
    assert store.facts("u")["location"] == "Đà Nẵng"
    assert "Huế -> Đà Nẵng" in store.read_text("u")  # old value kept only as superseded audit trail
    assert "Superseded" not in store.prompt_view("u")


def test_memory_decay_prunes_stale_low_value_interests(tmp_path: Path) -> None:
    store = UserProfileStore(tmp_path / "profiles")
    for _ in range(3):
        store.apply_candidates("u", [Candidate("interest", "Python", 0.8)])
    for i in range(MAX_INTERESTS + 2):
        store.apply_candidates("u", [Candidate("interest", f"topic-{i}", 0.8)])
    interests = store.facts("u")["interest"].split(", ")
    assert len(interests) == MAX_INTERESTS  # file growth is bounded
    assert "Python" in interests  # repeated mentions protect an old fact from decay
    assert "topic-0" not in interests  # old one-off interest was dropped


# --- compact memory ----------------------------------------------------------------------------


def test_compact_trigger(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    manager = CompactMemoryManager(config.compact_threshold_tokens, config.compact_keep_messages)

    manager.append("t", "user", "ngắn")  # far below the threshold
    assert manager.compaction_count("t") == 0 and manager.context("t")["summary"] == ""

    for i in range(10):
        manager.append("t", "user", f"Tin số {i}: " + "nội dung khá dài " * 12)
    context = manager.context("t")
    assert manager.compaction_count("t") >= 1
    assert len(context["messages"]) <= config.compact_keep_messages  # recent messages kept verbatim
    assert context["summary"]  # older content moved into the summary
    assert context["messages"][-1]["content"].startswith("Tin số 9")  # newest message is never lost
    # compact memory is per thread
    assert manager.compaction_count("other-thread") == 0


def test_summary_stays_bounded() -> None:
    messages = [{"role": "user", "content": f"Câu số {i}. " + "x" * 400} for i in range(50)]
    summary = summarize_messages(messages, max_items=6)
    assert len(summary.splitlines()) == 6
    assert estimate_tokens(summary) < 6 * 40  # each item is truncated, so the summary cannot grow unbounded
    assert "Câu số 49" in summary and "Câu số 0." not in summary  # newest kept, oldest dropped


def test_compact_reduces_prompt_load_on_long_thread(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    baseline = BaselineAgent(replace(config, state_dir=tmp_path / "b"), force_offline=True)
    advanced = AdvancedAgent(replace(config, state_dir=tmp_path / "a"), force_offline=True)
    conv = _stress_conversation()

    for turn in conv["turns"]:
        baseline.reply(conv["user_id"], "long", turn)
        advanced.reply(conv["user_id"], "long", turn)

    assert advanced.compaction_count("long") >= 2
    assert advanced.prompt_token_usage("long") < baseline.prompt_token_usage("long") * 0.7
    # ... and the profile kept what matters despite the compaction
    facts = advanced.profile_store.facts(conv["user_id"])
    assert facts["name"] == "DũngCT Stress" and facts["location"] == "Đà Nẵng"


def test_compact_costs_more_on_short_threads(tmp_path: Path) -> None:
    """Trade-off check: on a short chat the advanced agent pays for User.md + a bigger system prompt."""

    config = replace(make_config(tmp_path), compact_threshold_tokens=900)
    baseline = BaselineAgent(replace(config, state_dir=tmp_path / "b"), force_offline=True)
    advanced = AdvancedAgent(replace(config, state_dir=tmp_path / "a"), force_offline=True)
    for turn in ["Mình tên là An.", "Mình ở Huế và làm data engineer.", "Mình thích Python và Rust."]:
        baseline.reply("an", "short", turn)
        advanced.reply("an", "short", turn)
    assert advanced.compaction_count("short") == 0
    assert advanced.prompt_token_usage("short") > baseline.prompt_token_usage("short")


# --- cross-session recall ----------------------------------------------------------------------


def test_cross_session_recall(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    baseline = BaselineAgent(replace(config, state_dir=tmp_path / "b"), force_offline=True)
    advanced = AdvancedAgent(replace(config, state_dir=tmp_path / "a"), force_offline=True)

    for agent in (baseline, advanced):
        agent.reply("dungct", "session-1", "Chào bạn, mình tên là DũngCT.")
        agent.reply("dungct", "session-1", "Đồ uống yêu thích là cà phê sữa đá.")

    question = "Mình tên gì và đồ uống yêu thích là gì?"
    expected = ["DũngCT", "cà phê sữa đá"]

    # same thread: both agents remember (short-term memory)
    assert recall_points(baseline.reply("dungct", "session-1", question)["response"], expected) == 1.0
    # new thread: only the agent with User.md remembers
    assert recall_points(baseline.reply("dungct", "session-2", question)["response"], expected) == 0.0
    assert recall_points(advanced.reply("dungct", "session-2", question)["response"], expected) == 1.0
    # persistence is per user, not global
    assert recall_points(advanced.reply("someone-else", "session-3", question)["response"], expected) == 0.0
    # baseline never writes a profile file
    assert not (tmp_path / "b" / "profiles").exists()
    assert (tmp_path / "a" / "profiles" / "dungct" / "User.md").exists()


def test_advanced_keeps_latest_correction_across_sessions(tmp_path: Path) -> None:
    advanced = AdvancedAgent(make_config(tmp_path), force_offline=True)
    advanced.reply("u", "s1", "Mình ở Đà Nẵng và đang làm backend engineer.")
    advanced.reply("u", "s2", "Mình đính chính: giờ mình ở Huế, và mình đã chuyển sang MLOps engineer.")
    answer = advanced.reply("u", "s3", "Hiện tại mình làm nghề gì và đang ở đâu?")["response"]
    assert "MLOps engineer" in answer and "Huế" in answer
    assert "backend" not in answer and "Đà Nẵng" not in answer


def test_questions_never_write_to_user_md(tmp_path: Path) -> None:
    advanced = AdvancedAgent(make_config(tmp_path), force_offline=True)
    advanced.reply("u", "s1", "Bạn có biết DũngCT không?")
    advanced.reply("u", "s1", "Tên mình là gì?")
    assert advanced.memory_file_size("u") == 0


# --- benchmark end-to-end & determinism --------------------------------------------------------


def test_benchmark_suites_tell_the_expected_story(tmp_path: Path) -> None:
    config = load_config(tmp_path)  # default threshold
    standard = run_suite("standard", DATA_DIR / "conversations.json", config, force_offline=True)
    stress = run_suite("stress", DATA_DIR / "advanced_long_context.json", config, force_offline=True)

    for baseline, advanced in (standard, stress):
        assert baseline.recall_score < advanced.recall_score
        assert advanced.recall_score == 1.0
        assert baseline.memory_growth_bytes == 0 < advanced.memory_growth_bytes
    assert standard[1].prompt_tokens_processed > standard[0].prompt_tokens_processed  # short: advanced pricier
    assert stress[1].prompt_tokens_processed < stress[0].prompt_tokens_processed  # long: compact wins
    assert stress[1].compactions >= 2 and stress[0].compactions == 0

    # deterministic: same input -> identical rows
    assert run_suite("stress", DATA_DIR / "advanced_long_context.json", config, force_offline=True) == stress


def test_recall_and_quality_scoring() -> None:
    assert recall_points("Tên: DũngCT; Đồ uống: cà phê sữa đá", ["DũngCT", "cà phê sữa đá"]) == 1.0
    assert recall_points("Tên: DũngCT", ["DũngCT", "cà phê sữa đá"]) == 0.5
    assert recall_points("Mình chưa có thông tin, hai bạn ạ", ["AI"]) == 0.0  # "AI" must not match inside "hai"
    assert heuristic_quality("DũngCT", ["DũngCT"]) > heuristic_quality("Mình chưa có thông tin.", ["DũngCT"])


# --- live path (wiring checked with a fake chat model, no API key needed) ----------------------


def _fake_model():
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    from langchain_core.messages import AIMessage

    class FakeToolModel(GenericFakeChatModel):
        def bind_tools(self, tools, **kwargs):  # tools are never called by the scripted replies
            return self

    return FakeToolModel(messages=iter(AIMessage(content=f"live-{i}") for i in range(100)))


@pytest.fixture
def live_config(tmp_path: Path):
    config = make_config(tmp_path)
    return replace(config, model=replace(config.model, provider="openai", api_key="test-key"))


def test_live_agents_wire_up_with_a_fake_model(tmp_path: Path, live_config, monkeypatch) -> None:
    pytest.importorskip("langchain")
    for module in (agent_baseline, agent_advanced):
        monkeypatch.setattr(module, "build_chat_model", lambda _cfg: _fake_model())

    baseline = BaselineAgent(live_config)
    advanced = AdvancedAgent(live_config)
    assert baseline.langchain_agent is not None and advanced.langchain_agent is not None

    assert baseline.reply("u", "t1", "xin chào")["response"] == "live-0"
    reply = advanced.reply("u", "t1", "Mình tên là DũngCT.")
    assert reply["response"].startswith("live-0")
    assert advanced.profile_store.facts("u")["name"] == "DũngCT"  # User.md still persisted in live mode
    assert advanced.prompt_token_usage("t1") > 0


def test_force_offline_never_builds_a_live_agent(live_config) -> None:
    assert BaselineAgent(live_config, force_offline=True).langchain_agent is None
    assert AdvancedAgent(live_config, force_offline=True).langchain_agent is None

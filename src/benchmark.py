from __future__ import annotations

import json
import re
import shutil
import sys
import unicodedata
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import LabConfig, load_config

COLUMNS = [
    "Agent",
    "Agent tokens only",
    "Prompt tokens processed",
    "Cross-session recall",
    "Response quality",
    "Memory growth (bytes)",
    "Compactions",
]


@dataclass
class BenchmarkRow:
    agent_name: str
    agent_tokens_only: int
    prompt_tokens_processed: int
    recall_score: float
    response_quality: float
    memory_growth_bytes: int
    compactions: int


def load_conversations(path: Path) -> list[dict[str, Any]]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _contains(answer: str, expected: str) -> bool:
    """Case-insensitive containment on whole words, so "AI" does not match inside "hai"."""

    answer = unicodedata.normalize("NFC", answer).casefold()
    expected = unicodedata.normalize("NFC", expected).casefold()
    return re.search(rf"(?<!\w){re.escape(expected)}(?!\w)", answer) is not None


def recall_points(answer: str, expected: list[str]) -> float:
    """1 if every expected fact appears, 0.5 if only some do, 0 if none."""

    if not expected:
        return 1.0
    hits = sum(_contains(answer, item) for item in expected)
    if hits == len(expected):
        return 1.0
    return 0.5 if hits else 0.0


_REFUSAL_RE = re.compile(r"chưa có thông tin|không biết|không nhớ", re.IGNORECASE)


def heuristic_quality(answer: str, expected: list[str]) -> float:
    """Offline quality score in [0, 1]: 70% fact coverage, 15% concise (<=300 chars), 15% not a refusal."""

    coverage = sum(_contains(answer, item) for item in expected) / len(expected) if expected else 1.0
    concise = 1.0 if len(answer) <= 300 else max(0.0, 1 - (len(answer) - 300) / 600)
    not_refusal = 0.0 if _REFUSAL_RE.search(answer) else 1.0
    return round(0.7 * coverage + 0.15 * concise + 0.15 * not_refusal, 3)


def run_agent_benchmark(agent_name: str, agent, conversations: list[dict[str, Any]], config) -> BenchmarkRow:
    """Evaluate one agent over many conversations.

    For each conversation: feed every turn in a session thread, then ask the recall questions in a
    *fresh* thread (same user) to measure cross-session recall.
    """

    size_of = getattr(agent, "memory_file_size", lambda user_id: 0)
    users = sorted({conv["user_id"] for conv in conversations})
    size_before = {user: size_of(user) for user in users}

    threads: list[str] = []
    recall_scores: list[float] = []
    quality_scores: list[float] = []

    for conv in conversations:
        user_id = conv["user_id"]
        session_thread = f"{conv['id']}-session"
        recall_thread = f"{conv['id']}-recall"
        threads += [session_thread, recall_thread]

        for turn in conv["turns"]:
            agent.reply(user_id, session_thread, turn)

        for item in conv.get("recall_questions", []):
            answer = agent.reply(user_id, recall_thread, item["question"])["response"]
            recall_scores.append(recall_points(answer, item["expected_contains"]))
            quality_scores.append(heuristic_quality(answer, item["expected_contains"]))

    def mean(values: list[float]) -> float:
        return round(sum(values) / len(values), 3) if values else 0.0

    return BenchmarkRow(
        agent_name=agent_name,
        agent_tokens_only=sum(agent.token_usage(t) for t in threads),
        prompt_tokens_processed=sum(agent.prompt_token_usage(t) for t in threads),
        recall_score=mean(recall_scores),
        response_quality=mean(quality_scores),
        memory_growth_bytes=sum(size_of(user) - size_before[user] for user in users),
        compactions=sum(agent.compaction_count(t) for t in threads),
    )


def format_rows(rows: list[BenchmarkRow]) -> str:
    """Render rows as a markdown table (uses `tabulate` when installed)."""

    table = [
        [
            row.agent_name,
            row.agent_tokens_only,
            row.prompt_tokens_processed,
            f"{row.recall_score:.2f}",
            f"{row.response_quality:.2f}",
            row.memory_growth_bytes,
            row.compactions,
        ]
        for row in rows
    ]
    try:
        from tabulate import tabulate

        return tabulate(table, headers=COLUMNS, tablefmt="github")
    except ImportError:
        lines = ["| " + " | ".join(COLUMNS) + " |", "|" + "|".join("---" for _ in COLUMNS) + "|"]
        lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in table]
        return "\n".join(lines)


def run_suite(name: str, dataset: Path, config: LabConfig, force_offline: bool) -> list[BenchmarkRow]:
    """Run baseline + advanced on one dataset, each against a clean state directory."""

    conversations = load_conversations(dataset)
    rows = []
    for agent_name, agent_cls in (("Baseline", BaselineAgent), ("Advanced", AdvancedAgent)):
        state_dir = config.state_dir / "benchmark" / name / agent_name.lower()
        if state_dir.exists():
            shutil.rmtree(state_dir)  # only ever our own benchmark sub-folder
        agent = agent_cls(replace(config, state_dir=state_dir), force_offline=force_offline)
        rows.append(run_agent_benchmark(agent_name, agent, conversations, config))
    return rows


def _delta_line(rows: list[BenchmarkRow]) -> str:
    base, adv = rows
    if not base.prompt_tokens_processed:
        return ""
    change = (adv.prompt_tokens_processed - base.prompt_tokens_processed) / base.prompt_tokens_processed * 100
    verb = "cao hơn" if change > 0 else "thấp hơn"
    return f"Prompt tokens processed của Advanced {verb} Baseline {abs(change):.1f}%."


def main() -> None:
    """Run both benchmark suites and print the two comparison tables."""

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    force_offline = "--live" not in sys.argv[1:]
    config = load_config(Path(__file__).resolve().parent.parent)

    suites = (
        ("standard", "Standard Benchmark", config.data_dir / "conversations.json"),
        ("stress", "Long-Context Stress Benchmark", config.data_dir / "advanced_long_context.json"),
    )
    mode = "offline (deterministic)" if force_offline else "live"
    print(f"Mode: {mode} | compact threshold = {config.compact_threshold_tokens} tokens, "
          f"keep = {config.compact_keep_messages} messages\n")
    for name, title, dataset in suites:
        rows = run_suite(name, dataset, config, force_offline)
        print(f"## {title}\n")
        print(format_rows(rows))
        print(f"\n{_delta_line(rows)}\n")


if __name__ == "__main__":
    main()

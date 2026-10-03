from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple


def estimate_tokens(text: str) -> int:
    """Cheap, deterministic token estimator: ~4 characters per token (0 for empty text)."""

    text = (text or "").strip()
    if not text:
        return 0
    return max(1, math.ceil(len(text) / 4))


# ---------------------------------------------------------------------------
# Profile model: structured facts with confidence / mentions / recency
# ---------------------------------------------------------------------------

# Bonus knobs (documented in README "Phân tích kết quả"):
CONFIDENCE_THRESHOLD = 0.6   # facts below this are never written to User.md
HALF_LIFE_SEQ = 40           # memory decay: a fact loses half its weight every 40 profile updates
MAX_INTERESTS = 6            # multi-valued facts are capped; lowest-scoring entries are pruned
MAX_STYLE_TAGS = 6
MAX_SUPERSEDED = 5           # how many replaced values we keep for auditing

SINGLE_KEYS = ("name", "location", "profession", "drink", "food", "pet")
MULTI_KEYS = ("style", "interest")
LABELS = {
    "name": "Tên",
    "location": "Nơi ở hiện tại",
    "profession": "Nghề nghiệp hiện tại",
    "style": "Style trả lời",
    "drink": "Đồ uống yêu thích",
    "food": "Món ăn yêu thích",
    "pet": "Thú cưng",
    "interest": "Mối quan tâm",
}
_CAPS = {"style": MAX_STYLE_TAGS, "interest": MAX_INTERESTS}
_SECTION_TITLES = {"style": "Style", "interest": "Interests"}


@dataclass
class Fact:
    key: str
    value: str
    confidence: float = 0.9
    mentions: int = 1
    seq: int = 0

    def score(self, now_seq: int) -> float:
        """Priority used to rank/prune multi-valued facts: confidence x repetition x recency decay."""

        age = max(0, now_seq - self.seq)
        return self.confidence * (1 + math.log2(max(1, self.mentions))) * 0.5 ** (age / HALF_LIFE_SEQ)


class Candidate(NamedTuple):
    key: str
    value: str
    confidence: float


def _norm(value: str) -> str:
    return unicodedata.normalize("NFC", value).strip().casefold()


# ---------------------------------------------------------------------------
# Extraction (heuristic, Vietnamese)
# ---------------------------------------------------------------------------

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
_QUERY_RE = re.compile(
    r"^\s*(?:nhắc lại|tóm tắt|hãy nhắc|bạn có (?:thể )?nhắc|bạn thử nhớ|bạn có biết|bạn biết)\b"
    r"|\b(?:là gì|là ai|ở đâu|con gì|nghề gì|làm gì)\b"
    r"|\bnhắc lại (?:giúp |cho )?mình\b",
    re.IGNORECASE,
)
_JOKE_RE = re.compile(r"\b(?:đùa|giả sử|tưởng tượng|hay là)\b", re.IGNORECASE)
_HEDGE_RE = re.compile(r"\b(?:có thể|chắc là|có lẽ|hình như|đang cân nhắc|đang nghĩ|thử|tạm|nếu)\b", re.IGNORECASE)
_CORRECTION_RE = re.compile(r"đính chính|cập nhật|giờ (?:mình|tôi)|không còn|chuyển sang|thực ra", re.IGNORECASE)
_FIRST_PERSON_RE = re.compile(r"\b(?:mình|tôi|tớ)\b", re.IGNORECASE)
_NEGATION_TAIL_RE = re.compile(r"(?:không còn|không phải|chứ không|không)\s*$", re.IGNORECASE)
_STOP_VALUES = {"gì", "nào", "ai", "đâu", "sao", "con", "bé", "một", ""}
_VALUE_BOUNDARY_RE = re.compile(r"\s(?:như|nhưng|để|mỗi|vì|lúc|khi|rồi|và|thì|nên|cho|ở|mà)\b|[,.;:!?]")
_DRINK_WORD_RE = re.compile(r"cà phê|\btrà\b|\bbia\b|sinh tố|nước ép", re.IGNORECASE)
_EN_TITLES = "engineer|developer|manager|scientist|analyst|designer|architect|researcher"
_VI_TITLES = "kỹ sư|lập trình viên|giảng viên|giáo viên|sinh viên|bác sĩ"
# English titles may be preceded by up to two qualifier words ("MLOps engineer"); Vietnamese titles
# must follow the verb directly so "đây là phần sinh viên học được" is not read as a profession.
_PROFESSION_RE = re.compile(
    rf"\b(?:làm|chuyển sang|là)\s+((?:[\w+#.\-]+\s+){{0,2}}?(?:{_EN_TITLES})\b|(?:{_VI_TITLES})\b(?:\s+\w+)?)",
    re.IGNORECASE,
)
_NAME_RES = (
    re.compile(r"\b(?:mình|tôi)\s+tên\s+(?:là\s+)?", re.IGNORECASE),
    re.compile(r"\btên\s+(?:của\s+)?(?:mình|tôi)\s+là\s+", re.IGNORECASE),
)
_DRINK_RES = (
    (re.compile(r"đồ uống(?:\s+yêu thích)?(?:\s+của\s+mình)?\s+là\s+", re.IGNORECASE), 0.9),
    (re.compile(r"\b(?:vẫn|thường|hay|luôn)\s+uống\s+", re.IGNORECASE), 0.75),
)
_FOOD_RE = re.compile(r"\bmón(?: ăn)?\s+(?:yêu thích|ưa thích)(?:\s+của\s+mình)?\s+là\s+", re.IGNORECASE)
_PET_RES = (
    re.compile(r"\bnuôi\s+(?:một\s+)?(?:bé\s+|con\s+)?(\w+)", re.IGNORECASE),
    re.compile(r"\bcon\s+(\w+)(?=\s+tên\s)", re.IGNORECASE),
)
_INTEREST_RES = (
    re.compile(r"(?<!yêu )(?<!ưa )\bthích\s+(.+)$", re.IGNORECASE),
    re.compile(r"\bquan tâm(?:\s+(?:nhiều|nhất))?\s+(?:đến|tới)\s+(.+)$", re.IGNORECASE),
)
_INTEREST_CUT_RE = re.compile(r"\s(?:vì|nhưng|hơn|để|nên|khi|mà)\s")
_INTEREST_BAD_START = ("cách", "kiểu", "việc", "các", "những", "một", "bạn", "cái", "điều", "và")
_INTEREST_BAD_END = ("này", "đó", "kia", "ấy")
_STYLE_CUE_RE = re.compile(r"\b(?:trả lời|giải thích|câu trả lời|trình bày|style)\b", re.IGNORECASE)
_STYLE_SHORT_RE = re.compile(
    r"(?:trả lời|giải thích|trình bày|giữ câu|câu trả lời|style)[^.!?]*?\b(?:ngắn gọn|ngắn|gọn)\b", re.IGNORECASE
)
_STYLE_TRADEOFF_RE = re.compile(
    r"(?:giải thích|trả lời|so sánh|nhấn|bám)[^.!?]*trade-off|trade-off[^.!?]*(?:giải thích|trả lời)", re.IGNORECASE
)


def _take_capitalized(text: str, pos: int) -> str:
    """Read a run of capitalized words ("Đà Nẵng", "DũngCT Stress") starting at `pos`."""

    words: list[str] = []
    token = re.compile(r"\s*([^\s,.;:!?]+)")
    while True:
        m = token.match(text, pos)
        if not m or not m.group(1)[0].isupper():
            break
        words.append(m.group(1))
        pos = m.end()
        if pos < len(text) and text[pos] in ",.;:!?":
            break
    return " ".join(words)


def _take_phrase(text: str, pos: int, max_words: int = 5) -> str:
    """Read a lowercase-friendly value (drink/food) up to the next boundary word or punctuation."""

    rest = text[pos:]
    m = _VALUE_BOUNDARY_RE.search(rest)
    value = (rest[: m.start()] if m else rest).strip()
    if not value or len(value.split()) > max_words or _norm(value) in _STOP_VALUES:
        return ""
    return value


def _confidence(base: float, sentence: str, fact_pos: int, hedgeable: bool = True) -> float:
    """Confidence threshold logic: explicit/correcting statements score high, hedged ones low, jokes zero."""

    if _JOKE_RE.search(sentence):
        return 0.0
    conf = base
    if _CORRECTION_RE.search(sentence):
        conf += 0.05
    if hedgeable and _HEDGE_RE.search(sentence[:fact_pos]):
        conf -= 0.35
    return round(min(conf, 1.0), 2)


def _sentence_candidates(sentence: str) -> list[Candidate]:
    sentence = sentence.strip()
    if not sentence or sentence.endswith("?") or _QUERY_RE.search(sentence):
        return []  # questions are requests, not facts (guards against storing wrong facts)

    out: list[Candidate] = []
    first_person = bool(_FIRST_PERSON_RE.search(sentence))

    # name
    for rx in _NAME_RES:
        m = rx.search(sentence)
        if m:
            name = _take_capitalized(sentence, m.end())
            if name:
                out.append(Candidate("name", name, _confidence(0.95, sentence, m.start())))
            break

    # location: "ở <Place>" unless negated ("không còn ở ..."); last valid mention wins
    if first_person:
        place, place_pos = "", 0
        for m in re.finditer(r"\bở\s+", sentence, re.IGNORECASE):
            if _NEGATION_TAIL_RE.search(sentence[: m.start()]):
                continue
            cand = _take_capitalized(sentence, m.end())
            if cand:
                place, place_pos = cand, m.start()
        m = re.search(r"nơi ở(?: hiện tại)?\s+là\s+", sentence, re.IGNORECASE)
        if m and _take_capitalized(sentence, m.end()):
            place, place_pos = _take_capitalized(sentence, m.end()), m.start()
        if place:
            out.append(Candidate("location", place, _confidence(0.85, sentence, place_pos)))

    # profession: "làm / chuyển sang / là <role>" unless negated ("không còn làm backend engineer")
    if first_person or re.search(r"\bnghề\b", sentence, re.IGNORECASE):
        role, role_pos = "", 0
        for m in _PROFESSION_RE.finditer(sentence):
            if _NEGATION_TAIL_RE.search(sentence[: m.start()]):
                continue
            role, role_pos = re.sub(r"^(?:một|1)\s+", "", m.group(1).strip(), flags=re.IGNORECASE), m.start()
        if role:
            out.append(Candidate("profession", role, _confidence(0.9, sentence, role_pos)))

    # drink / food (explicit "yêu thích" or habitual "vẫn uống ...")
    for rx, base in _DRINK_RES:
        m = rx.search(sentence)
        if m:
            value = _take_phrase(sentence, m.end())
            if value:
                out.append(Candidate("drink", value, _confidence(base, sentence, m.start())))
            break
    m = _FOOD_RE.search(sentence)
    if m:
        value = _take_phrase(sentence, m.end())
        if value:
            out.append(Candidate("food", value, _confidence(0.9, sentence, m.start())))

    # pet: "nuôi một bé corgi tên Bơ" / "con corgi tên Bơ"
    if first_person:
        for rx in _PET_RES:
            m = rx.search(sentence)
            if not m or _norm(m.group(1)) in _STOP_VALUES:
                continue
            value = m.group(1)
            tail = re.match(r"\s+tên\s+", sentence[m.end():])
            if tail:
                pet_name = _take_capitalized(sentence, m.end() + tail.end())
                if pet_name:
                    value = f"{value} tên {pet_name}"
            out.append(Candidate("pet", value, _confidence(0.9, sentence, m.start())))
            break

    # interests ("thích X, Y và Z", "quan tâm đến X")
    if first_person:
        for rx in _INTEREST_RES:
            m = rx.search(sentence)
            if not m:
                continue
            clause = _INTEREST_CUT_RE.split(" " + m.group(1))[0]
            for raw in re.split(r",|\svà\s", clause):
                item = re.sub(r"^\s*và\s+", "", raw).strip().rstrip(".!;:")
                words = item.split()
                if not (1 <= len(words) <= 4) or _norm(item) in _STOP_VALUES:
                    continue
                if words[0].casefold() in _INTEREST_BAD_START or words[-1].casefold() in _INTEREST_BAD_END:
                    continue
                key = "drink" if _DRINK_WORD_RE.search(item) else "interest"
                conf = 0.7 if key == "drink" else 0.8
                out.append(Candidate(key, item, _confidence(conf, sentence, m.start())))
            break

    # response style: normalized tags so a later turn can add to (not erase) earlier preferences
    if _STYLE_CUE_RE.search(sentence):
        style_conf = _confidence(0.85, sentence, 0, hedgeable=False)
        tags: list[str] = []
        if _STYLE_SHORT_RE.search(sentence):
            tags.append("ngắn gọn")
        bullet = re.search(r"(\d+)\s*bullet", sentence, re.IGNORECASE)
        if bullet:
            tags.append(f"{bullet.group(1)} bullet")
        elif re.search(r"\bbullet\b", sentence, re.IGNORECASE):
            tags.append("bullet")
        if re.search(r"ví dụ\s+(?:thực tế|thực chiến)", sentence, re.IGNORECASE):
            tags.append("ví dụ thực tế")
        if re.search(r"ví dụ\s+số liệu", sentence, re.IGNORECASE):
            tags.append("ví dụ số liệu")
        if _STYLE_TRADEOFF_RE.search(sentence):
            tags.append("nhấn trade-off")
        out.extend(Candidate("style", tag, style_conf) for tag in tags)

    return out


def extract_profile_candidates(message: str) -> list[Candidate]:
    """All (key, value, confidence) facts found in a message, before the confidence threshold."""

    message = unicodedata.normalize("NFC", message or "")
    candidates: list[Candidate] = []
    for sentence in _SENTENCE_SPLIT.split(message):
        candidates.extend(_sentence_candidates(sentence))
    return candidates


def extract_profile_updates(message: str, min_confidence: float = CONFIDENCE_THRESHOLD) -> dict[str, str]:
    """Convert raw user text into stable profile facts (only those above the confidence threshold).

    Single-valued facts keep the last mention in the message; style/interest are joined with ", ".
    """

    updates: dict[str, str] = {}
    for key, value, conf in extract_profile_candidates(message):
        if conf < min_confidence:
            continue
        if key in MULTI_KEYS:
            existing = [v for v in updates.get(key, "").split(", ") if v]
            if _norm(value) not in {_norm(v) for v in existing}:
                existing.append(value)
            updates[key] = ", ".join(existing)
        else:
            updates[key] = value
    return updates


# ---------------------------------------------------------------------------
# Applying candidates to a profile (conflict handling + decay), pure functions
# ---------------------------------------------------------------------------


def apply_candidates(
    facts: list[Fact],
    superseded: list[tuple[str, str, str, int]],
    candidates: list[Candidate],
    seq: int,
    min_confidence: float = CONFIDENCE_THRESHOLD,
) -> list[str]:
    """Merge candidates into `facts` in place and return a list of human-readable changes.

    Conflict handling: a new single-valued fact replaces the old one (the old value moves to
    `superseded`), unless it is much less certain than what we already hold.
    """

    changes: list[str] = []
    for key, value, conf in candidates:
        if conf < min_confidence:
            continue
        if key in SINGLE_KEYS:
            current = next((f for f in facts if f.key == key), None)
            if current is None:
                facts.append(Fact(key, value, conf, 1, seq))
                changes.append(f"{key}")
            elif _norm(current.value) == _norm(value):
                current.mentions += 1
                current.seq = seq
                current.confidence = max(current.confidence, conf)
            elif conf + 0.2 >= current.confidence:
                superseded.append((key, current.value, value, seq))
                del superseded[:-MAX_SUPERSEDED]
                current.value, current.confidence, current.mentions, current.seq = value, conf, 1, seq
                changes.append(f"{key}")
            # else: weak contradicting statement -> keep the confident value
        else:
            current = next((f for f in facts if f.key == key and _norm(f.value) == _norm(value)), None)
            if current is not None:
                current.mentions += 1
                current.seq = seq
                current.confidence = max(current.confidence, conf)
                continue
            facts.append(Fact(key, value, conf, 1, seq))
            changes.append(f"{key}")
            if key == "style" and re.match(r"\d+ bullet$", value):
                facts[:] = [f for f in facts if not (f.key == "style" and f.value == "bullet")]
            same = [f for f in facts if f.key == key]
            if len(same) > _CAPS[key]:
                # memory decay: drop the lowest-priority entry (old, rarely repeated, low confidence)
                weakest = min(same, key=lambda f: f.score(seq))
                facts.remove(weakest)
    return changes


def facts_dict(facts: list[Fact], now_seq: int) -> dict[str, str]:
    """Flatten facts to {key: value}; multi-valued keys are ranked by score and joined."""

    result: dict[str, str] = {}
    for key in SINGLE_KEYS:
        current = next((f for f in facts if f.key == key), None)
        if current:
            result[key] = current.value
    for key in MULTI_KEYS:
        items = sorted((f for f in facts if f.key == key), key=lambda f: -f.score(now_seq))
        if items:
            result[key] = ", ".join(f.value for f in items)
    return result


def facts_from_messages(user_messages: list[str]) -> dict[str, str]:
    """In-memory profile built only from one thread's messages (used by the baseline agent)."""

    facts: list[Fact] = []
    superseded: list[tuple[str, str, str, int]] = []
    for seq, message in enumerate(user_messages, start=1):
        apply_candidates(facts, superseded, extract_profile_candidates(message), seq)
    return facts_dict(facts, len(user_messages))


# ---------------------------------------------------------------------------
# Answering from facts (shared by both offline agents)
# ---------------------------------------------------------------------------

_FIELD_PATTERNS = (
    ("name", re.compile(r"\btên\b", re.IGNORECASE)),
    ("location", re.compile(r"nơi ở|ở đâu|đang ở|còn ở|hiện ở|sống ở", re.IGNORECASE)),
    ("profession", re.compile(r"\bnghề\b|công việc|làm gì", re.IGNORECASE)),
    ("style", re.compile(r"\bstyle\b|kiểu trả lời|trả lời như thế nào|trả lời mình thích|cách trả lời", re.IGNORECASE)),
    ("drink", re.compile(r"đồ uống", re.IGNORECASE)),
    ("food", re.compile(r"món ăn", re.IGNORECASE)),
    ("pet", re.compile(r"nuôi|thú cưng|con gì", re.IGNORECASE)),
    ("interest", re.compile(r"quan tâm|sở thích|thích gì", re.IGNORECASE)),
)
_SUMMARY_RE = re.compile(r"là ai|tóm tắt|mô tả mình|bạn có biết|bạn biết", re.IGNORECASE)
_SUMMARY_FIELDS = ("name", "profession", "location", "interest")
_INTEREST_ANSWER_LIMIT = 3
NO_INFO = "Mình chưa có thông tin về bạn trong cuộc trò chuyện này."
ACK = "Mình đã ghi nhận."


def is_query(message: str) -> bool:
    message = (message or "").strip()
    return message.endswith("?") or bool(_QUERY_RE.search(message))


def answer_from_facts(message: str, facts: dict[str, str]) -> str:
    """Deterministic answer to a recall question from `facts`; plain acknowledgement otherwise."""

    if not is_query(message):
        return ACK
    asked = sorted(
        (m.start(), key) for key, rx in _FIELD_PATTERNS if (m := rx.search(message))
    )
    keys = [key for _, key in asked]
    if not keys and _SUMMARY_RE.search(message):
        keys = list(_SUMMARY_FIELDS)
    if not keys:
        return ACK

    known, unknown = [], []
    for key in keys:
        value = facts.get(key)
        if value and key == "interest":
            value = ", ".join(value.split(", ")[:_INTEREST_ANSWER_LIMIT])
        (known if value else unknown).append(f"{LABELS[key]}: {value}" if value else LABELS[key].lower())
    if not known:
        return NO_INFO
    answer = "; ".join(known) + "."
    if unknown:
        answer += f" Chưa có thông tin: {', '.join(unknown)}."
    return answer


# ---------------------------------------------------------------------------
# Persistent User.md store
# ---------------------------------------------------------------------------

_LINE_RE = re.compile(
    r"^- (?:(?P<key>[a-z]+): )?(?P<value>.+?)(?:\s*<!-- conf=(?P<conf>[\d.]+) n=(?P<n>\d+) seq=(?P<seq>\d+) -->)?\s*$"
)
_SEQ_RE = re.compile(r"<!-- seq: (\d+) -->")
_SUPERSEDED_RE = re.compile(r"^- (?P<key>[a-z]+): (?P<old>.+?) -> (?P<new>.+?) \(seq (?P<seq>\d+)\)\s*$")


@dataclass
class UserProfileStore:
    """Persistent storage for `User.md` (one markdown file per user).

    File layout (human readable, but also machine-parsable so edits are round-tripped):

        # User Profile: <user>
        <!-- seq: N -->
        ## Facts          single-valued facts (name, location, profession, ...)
        ## Style          style tags
        ## Interests      interests, ranked by score
        ## Superseded     values replaced by corrections (audit only, never injected in prompts)
    """

    root_dir: Path

    def path_for(self, user_id: str) -> Path:
        slug = re.sub(r"[^\w\-]+", "_", unicodedata.normalize("NFC", user_id or ""), flags=re.UNICODE).strip("_")
        return self.root_dir / (slug or "anonymous") / "User.md"

    def read_text(self, user_id: str) -> str:
        path = self.path_for(user_id)
        if path.exists():
            return path.read_text(encoding="utf-8")
        return self._render(user_id, [], [], 0)

    def write_text(self, user_id: str, content: str) -> Path:
        path = self.path_for(user_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def edit_text(self, user_id: str, search_text: str, replacement: str) -> bool:
        """Replace one occurrence of `search_text`; return whether the file changed."""

        if not search_text:
            return False
        content = self.read_text(user_id)
        if search_text not in content:
            return False
        updated = content.replace(search_text, replacement, 1)
        if updated == content:
            return False
        self.write_text(user_id, updated)
        return True

    def file_size(self, user_id: str) -> int:
        path = self.path_for(user_id)
        return path.stat().st_size if path.exists() else 0

    # -- structured helpers ------------------------------------------------

    def facts(self, user_id: str) -> dict[str, str]:
        facts, _, seq = self._load(user_id)
        return facts_dict(facts, seq)

    def upsert_fact(self, user_id: str, key: str, value: str, confidence: float = 0.9) -> bool:
        """Insert/update one fact through the same conflict handling as extracted facts."""

        return bool(self.apply_candidates(user_id, [Candidate(key, value, confidence)]))

    def apply_candidates(self, user_id: str, candidates: list[Candidate]) -> list[str]:
        """Merge extracted candidates into User.md; write only when something changed."""

        facts, superseded, seq = self._load(user_id)
        before = (
            [(f.key, f.value, f.confidence, f.mentions, f.seq) for f in facts],
            list(superseded),
        )
        changes = apply_candidates(facts, superseded, candidates, seq + 1)
        after = (
            [(f.key, f.value, f.confidence, f.mentions, f.seq) for f in facts],
            list(superseded),
        )
        if after != before:
            self.write_text(user_id, self._render(user_id, facts, superseded, seq + 1))
        return changes

    def prompt_view(self, user_id: str) -> str:
        """Lean version injected into the prompt: current facts only, no metadata, no history."""

        facts, _, seq = self._load(user_id)
        data = facts_dict(facts, seq)
        if not data:
            return ""
        lines = ["Hồ sơ người dùng (User.md):"]
        lines += [f"- {LABELS[k]}: {data[k]}" for k in (*SINGLE_KEYS, *MULTI_KEYS) if k in data]
        return "\n".join(lines)

    # -- (de)serialization --------------------------------------------------

    def _load(self, user_id: str) -> tuple[list[Fact], list[tuple[str, str, str, int]], int]:
        path = self.path_for(user_id)
        if not path.exists():
            return [], [], 0
        text = path.read_text(encoding="utf-8")
        seq_match = _SEQ_RE.search(text)
        seq = int(seq_match.group(1)) if seq_match else 0
        facts: list[Fact] = []
        superseded: list[tuple[str, str, str, int]] = []
        section = ""
        for raw in text.splitlines():
            line = raw.strip()
            if line.startswith("## "):
                section = line[3:].strip().lower()
                continue
            if not line.startswith("- "):
                continue
            if section.startswith("superseded"):
                m = _SUPERSEDED_RE.match(line)
                if m:
                    superseded.append((m["key"], m["old"], m["new"], int(m["seq"])))
                continue
            m = _LINE_RE.match(line)
            if not m:
                continue
            key = m["key"] or {"style": "style", "interests": "interest"}.get(section)
            if key not in (*SINGLE_KEYS, *MULTI_KEYS):
                continue  # unknown, hand-written lines stay in the file but are not parsed as facts
            facts.append(
                Fact(
                    key,
                    m["value"].strip(),
                    float(m["conf"]) if m["conf"] else 0.9,
                    int(m["n"]) if m["n"] else 1,
                    int(m["seq"]) if m["seq"] else seq,
                )
            )
        return facts, superseded, seq

    def _render(
        self, user_id: str, facts: list[Fact], superseded: list[tuple[str, str, str, int]], seq: int
    ) -> str:
        def meta(f: Fact) -> str:
            return f"<!-- conf={f.confidence:.2f} n={f.mentions} seq={f.seq} -->"

        lines = [f"# User Profile: {user_id}", f"<!-- seq: {seq} -->", "", "## Facts"]
        singles = [f for key in SINGLE_KEYS for f in facts if f.key == key]
        lines += [f"- {f.key}: {f.value} {meta(f)}" for f in singles] or ["_(chưa có thông tin)_"]
        for key in MULTI_KEYS:
            items = sorted((f for f in facts if f.key == key), key=lambda f: -f.score(seq))
            if items:
                lines += ["", f"## {_SECTION_TITLES[key]}"] + [f"- {f.value} {meta(f)}" for f in items]
        if superseded:
            lines += ["", "## Superseded (đã bị thay thế, không dùng làm thông tin hiện tại)"]
            lines += [f"- {k}: {old} -> {new} (seq {s})" for k, old, new, s in superseded]
        return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Compact memory
# ---------------------------------------------------------------------------


def _salient(content: str, limit: int = 110) -> str:
    first = _SENTENCE_SPLIT.split(content.strip())[0].strip()
    return first if len(first) <= limit else first[: limit - 1].rstrip() + "…"


def summarize_messages(messages: list[dict[str, str]], max_items: int = 6) -> str:
    """Heuristic summary of older messages: one salient sentence per user turn, newest `max_items` kept.

    A message with role "summary" carries a previous summary so summaries can be chained. Assistant
    turns are dropped (acks / answers re-derivable from User.md). This is lossy by design.
    """

    lines: list[str] = []
    for message in messages:
        role, content = message.get("role", ""), message.get("content", "")
        if role == "summary":
            lines.extend(line.strip() for line in content.splitlines() if line.strip())
        elif role == "user" and content.strip():
            lines.append(f"- {_salient(content)}")
    deduped = list(dict.fromkeys(lines))
    return "\n".join(deduped[-max_items:])


@dataclass
class CompactMemoryManager:
    """Compact memory for long threads.

    - keeps the `keep_messages` most recent messages verbatim
    - when summary + messages exceed `threshold_tokens`, older messages are folded into the summary
    - counts compactions per thread for the benchmark
    """

    threshold_tokens: int
    keep_messages: int
    state: dict[str, dict[str, object]] = field(default_factory=dict)
    summary_items: int = 6

    def append(self, thread_id: str, role: str, content: str) -> None:
        thread = self.context(thread_id)
        messages: list[dict[str, str]] = thread["messages"]  # type: ignore[assignment]
        messages.append({"role": role, "content": content})
        if self._tokens(thread) > self.threshold_tokens and len(messages) > self.keep_messages:
            older, recent = messages[: -self.keep_messages], messages[-self.keep_messages :]
            thread["summary"] = summarize_messages(
                [{"role": "summary", "content": str(thread["summary"])}, *older], self.summary_items
            )
            thread["messages"] = recent
            thread["compactions"] = int(thread["compactions"]) + 1  # type: ignore[arg-type]

    def context(self, thread_id: str) -> dict[str, object]:
        if thread_id not in self.state:
            self.state[thread_id] = {"messages": [], "summary": "", "compactions": 0}
        return self.state[thread_id]

    def compaction_count(self, thread_id: str) -> int:
        return int(self.context(thread_id)["compactions"])  # type: ignore[arg-type]

    @staticmethod
    def _tokens(thread: dict[str, object]) -> int:
        messages: list[dict[str, str]] = thread["messages"]  # type: ignore[assignment]
        return estimate_tokens(str(thread["summary"])) + sum(estimate_tokens(m["content"]) for m in messages)

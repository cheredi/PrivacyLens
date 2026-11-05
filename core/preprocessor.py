"""Text cleaning, validation, section detection and chunking."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

from config import Settings, get_settings

WORDS_PER_TOKEN = 0.75  # ~1.33 tokens per English word (GPT-style tokenizers)


class PolicyValidationError(Exception):
    """Raised when submitted text is not a usable policy."""


@dataclass(frozen=True)
class Chunk:
    id: str
    index: int
    heading: str | None
    text: str
    approx_tokens: int = field(default=0)

    @property
    def source_number(self) -> int:
        return self.index + 1

    @property
    def label(self) -> str:
        return f"Source {self.source_number}" + (f" — {self.heading}" if self.heading else "")


@dataclass
class Section:
    heading: str | None
    paragraphs: list[str]

    @property
    def text(self) -> str:
        return "\n".join(self.paragraphs)

    @property
    def words(self) -> int:
        return len(self.text.split())


def approx_tokens(text: str) -> int:
    return int(round(len(text.split()) / WORDS_PER_TOKEN))


# ---------------------------------------------------------------------------
# Cleaning & validation
# ---------------------------------------------------------------------------

_CONTROL_RE = re.compile(r"[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f​-‏  ﻿]")


def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\t", " ")
    text = _CONTROL_RE.sub("", text)
    lines = [re.sub(r"[  ]+", " ", ln).strip() for ln in text.split("\n")]
    out: list[str] = []
    blank = 0
    for ln in lines:
        if not ln:
            blank += 1
            if blank <= 1:
                out.append("")
            continue
        blank = 0
        out.append(ln)
    return "\n".join(out).strip()


def validate_policy_text(text: str, settings: Settings | None = None) -> str:
    settings = settings or get_settings()
    if not text or not text.strip():
        raise PolicyValidationError("Please provide privacy policy text to analyze.")
    if len(text) > settings.max_policy_chars:
        raise PolicyValidationError(
            f"That policy is very long ({len(text):,} characters). Please submit under "
            f"{settings.max_policy_chars:,} characters."
        )
    cleaned = clean_text(text)
    words = re.findall(r"[A-Za-z][A-Za-z'\-]+", cleaned)
    if len(words) < settings.min_policy_words:
        raise PolicyValidationError(
            f"That text is too short to be a privacy policy ({len(words)} words found; "
            f"at least {settings.min_policy_words} are needed)."
        )
    letters = sum(c.isalpha() for c in cleaned)
    if letters / max(len(cleaned), 1) < 0.5:
        raise PolicyValidationError("That text doesn't look like readable prose. Please paste the policy itself.")
    return cleaned


# ---------------------------------------------------------------------------
# Section detection
# ---------------------------------------------------------------------------

_MD_HEADING = re.compile(r"^#{1,6}\s+(.+?)\s*#*$")
_NUMBERED_HEADING = re.compile(r"^(?:section\s+)?(\d+(?:\.\d+)*|[IVXLC]+)[.)]?\s+([A-Z].{1,90})$", re.I)
_SMALL_WORDS = {"a", "an", "and", "or", "of", "the", "to", "in", "on", "for", "with", "we", "your", "you", "how", "our", "by", "at", "is", "do"}


def _heading_text(line: str, next_line: str | None) -> str | None:
    """Return the heading text if `line` looks like a section heading."""
    m = _MD_HEADING.match(line)
    if m:
        return m.group(1).strip()
    if line.startswith(("-", "*", "•")) or len(line) > 100:
        return None
    if line.endswith((".", ",", ";")) and not _NUMBERED_HEADING.match(line):
        return None
    words = line.split()
    if not words or len(words) > 12:
        return None
    m = _NUMBERED_HEADING.match(line)
    if m and not line.endswith("."):
        return line
    if line.endswith((".", ",", ";")):
        return None
    has_follow = bool(next_line and next_line.strip())
    letters = [c for c in line if c.isalpha()]
    if letters and line.upper() == line and len(words) <= 10 and len(letters) >= 4:
        return line.title() if has_follow else None
    significant = [w for w in words if w.lower().strip(":?") not in _SMALL_WORDS]
    if significant and len(words) <= 10 and has_follow:
        capped = sum(1 for w in significant if w[:1].isupper())
        if capped / len(significant) >= 0.8 and (line.endswith((":", "?")) or not re.search(r"[.!]$", line)):
            return line.rstrip(":")
    return None


def split_sections(text: str) -> list[Section]:
    lines = text.split("\n")
    sections: list[Section] = [Section(heading=None, paragraphs=[])]
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        nxt = next((ln for ln in lines[i + 1:] if ln.strip()), None)
        heading = _heading_text(line.strip(), nxt)
        if heading:
            sections.append(Section(heading=heading, paragraphs=[]))
        else:
            sections[-1].paragraphs.append(line.strip())
    return [s for s in sections if s.paragraphs or s.heading]


def _merge_small_sections(sections: list[Section], min_words: int) -> list[Section]:
    merged: list[Section] = []
    for sec in sections:
        if not sec.paragraphs:  # heading with no body: keep as context for next section
            if merged and not merged[-1].paragraphs:
                merged[-1] = Section(heading=f"{merged[-1].heading} › {sec.heading}", paragraphs=[])
            else:
                merged.append(Section(heading=sec.heading, paragraphs=[]))
            continue
        if merged and not merged[-1].paragraphs:
            parent = merged.pop()
            heading = sec.heading if sec.heading else parent.heading
            if parent.heading and sec.heading:
                heading = f"{parent.heading} › {sec.heading}"
            sec = Section(heading=heading, paragraphs=sec.paragraphs)
        if merged and merged[-1].words < min_words:
            prev = merged.pop()
            heading = prev.heading or sec.heading
            if prev.heading and sec.heading:
                heading = f"{prev.heading} / {sec.heading}"
            paragraphs = list(prev.paragraphs)
            if sec.heading and prev.heading:
                paragraphs.append(sec.heading + ":")
            paragraphs += sec.paragraphs
            sec = Section(heading=heading, paragraphs=paragraphs)
        merged.append(sec)
    if len(merged) > 1 and merged[-1].words < min_words and merged[-1].paragraphs:
        last = merged.pop()
        prev = merged.pop()
        heading = prev.heading if not last.heading else (f"{prev.heading} / {last.heading}" if prev.heading else last.heading)
        extra = ([last.heading + ":"] if last.heading else []) + last.paragraphs
        merged.append(Section(heading=heading, paragraphs=prev.paragraphs + extra))
    return [s for s in merged if s.paragraphs]


def _window_words(words: list[str], size: int, overlap: int) -> list[list[str]]:
    if len(words) <= size:
        return [words]
    step = max(size - overlap, 1)
    windows = []
    start = 0
    while start < len(words):
        window = words[start:start + size]
        windows.append(window)
        if start + size >= len(words):
            break
        start += step
    # Avoid a tiny trailing window: fold it into the previous one.
    if len(windows) > 1 and len(windows[-1]) < overlap + size // 4:
        tail = windows.pop()
        windows[-1] = windows[-1] + tail[overlap:]
    return windows


def chunk_policy(text: str, settings: Settings | None = None) -> list[Chunk]:
    """Split cleaned policy text into heading-aware, overlapping chunks.

    Chunk IDs (chunk_1, chunk_2, ...) are assigned in document order, so the
    same input always yields the same IDs.
    """
    settings = settings or get_settings()
    target_words = int(settings.chunk_target_tokens * WORDS_PER_TOKEN)
    overlap_words = int(settings.chunk_overlap_tokens * WORDS_PER_TOKEN)
    min_words = int(settings.min_section_tokens * WORDS_PER_TOKEN)

    sections = _merge_small_sections(split_sections(text), min_words)
    chunks: list[Chunk] = []
    for sec in sections:
        # Keep paragraph boundaries where possible by greedily packing paragraphs.
        paragraphs = sec.paragraphs
        if sec.words <= target_words:
            groups = [paragraphs]
        else:
            groups = []
            current: list[str] = []
            count = 0
            for para in paragraphs:
                n = len(para.split())
                if n > target_words:  # very long paragraph: hard window it
                    if current:
                        groups.append(current)
                        current, count = [], 0
                    for w in _window_words(para.split(), target_words, overlap_words):
                        groups.append([" ".join(w)])
                    continue
                if count + n > target_words and current:
                    groups.append(current)
                    # carry the tail of the previous group as overlap
                    tail_words = " ".join(current).split()[-overlap_words:]
                    current = ["… " + " ".join(tail_words)] if overlap_words else []
                    count = len(tail_words)
                current.append(para)
                count += n
            if current:
                groups.append(current)
        for group in groups:
            body = "\n".join(group).strip()
            if not body:
                continue
            idx = len(chunks)
            chunks.append(
                Chunk(id=f"chunk_{idx + 1}", index=idx, heading=sec.heading, text=body, approx_tokens=approx_tokens(body))
            )
    return chunks


def guess_title(text: str) -> str | None:
    for line in text.split("\n")[:8]:
        m = _MD_HEADING.match(line.strip())
        if m:
            return m.group(1).strip()[:120]
    first = next((ln.strip() for ln in text.split("\n") if ln.strip()), "")
    if first and len(first.split()) <= 12 and not first.endswith("."):
        return first[:120]
    return None

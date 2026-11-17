"""Split a document into qualifying prose sentences and overlapping windows.

Mirrors the published outline of Turnitin's approach: only prose sentences
qualify (headings, lists and other non-sentence text are skipped), and the
qualifying sentences are grouped into overlapping segments of a few hundred
words so every sentence is scored in context, usually more than once.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_WORD = re.compile(r"[^\W_]+(?:['’\-][^\W_]+)*")
_BOUNDARY = re.compile(r"[.!?]+[\"'”’)\]]*\s+")
_BULLET = re.compile(r"^\s*(?:[-*•–—·▪◦‣>]|\(?\d{1,3}[.)]|\(?[a-zA-Z][.)]|\[\d+\])\s+")
_ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc", "fig", "figs",
    "eq", "eqs", "no", "nos", "vol", "pp", "ed", "eds", "al", "cf", "ca", "approx",
    "inc", "ltd", "co", "e.g", "i.e", "u.s", "u.k", "ph.d", "jan", "feb", "mar",
    "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec",
}

MIN_PROSE_WORDS = 8


@dataclass(frozen=True)
class Sentence:
    index: int
    start: int  # character offsets into the original text
    end: int
    text: str
    words: int
    block: int


@dataclass(frozen=True)
class Window:
    first: int  # index of first sentence (inclusive)
    last: int  # index of last sentence (inclusive)
    text: str
    words: int


def count_words(text: str) -> int:
    return len(_WORD.findall(text))


def _is_boundary(text: str, match: re.Match[str]) -> bool:
    before = text[: match.start()]
    token = before.rsplit(None, 1)[-1].lower().lstrip("(\"'“‘[") if before.strip() else ""
    if token in _ABBREVIATIONS or (len(token) == 1 and token.isalpha()):
        return False  # "Dr." / "e.g." / an initial like "J."
    following = text[match.end():].lstrip("\"'“‘([")
    return not following or not following[0].islower()


def _split_sentences(text: str, offset: int) -> list[tuple[int, int]]:
    spans, start = [], 0
    for match in _BOUNDARY.finditer(text):
        if _is_boundary(text, match):
            end = match.end() - (len(match.group()) - len(match.group().rstrip()))
            spans.append((offset + start, offset + end))
            start = match.end()
    if text[start:].strip():
        spans.append((offset + start, offset + len(text.rstrip())))
    return spans


def _is_prose(block: str) -> bool:
    """Standard grammatical sentences only: no list items, headings or tables."""
    if _BULLET.match(block) or "\t" in block or block.count("|") >= 2:
        return False
    if count_words(block) < MIN_PROSE_WORDS:
        return False
    if not re.search(r"[.!?][\"'”’)\]]*(?:\s|$)", block):
        return False  # no sentence-ending punctuation: heading or caption
    letters = sum(ch.isalpha() for ch in block)
    return letters / max(len(block.replace(" ", "")), 1) >= 0.6


def qualifying_sentences(text: str) -> list[Sentence]:
    sentences: list[Sentence] = []
    for block_index, match in enumerate(re.finditer(r"[^\n]+", text)):
        raw = match.group()
        lead = len(raw) - len(raw.lstrip())
        block = raw.strip()
        if not _is_prose(block):
            continue
        for start, end in _split_sentences(block, match.start() + lead):
            words = count_words(text[start:end])
            if words:
                sentences.append(
                    Sentence(len(sentences), start, end, text[start:end], words, block_index)
                )
    return sentences


def build_windows(
    sentences: list[Sentence], window_words: int = 250, stride_words: int = 125
) -> list[Window]:
    """Overlapping windows of roughly `window_words`, advancing ~`stride_words`."""
    if not sentences:
        return []
    n = len(sentences)
    spans: list[tuple[int, int]] = []
    start = 0
    while True:
        end, words = start, sentences[start].words
        while end + 1 < n and words < window_words:
            end += 1
            words += sentences[end].words
        if end == n - 1:
            # Final window: reach backwards so the tail is scored with full context.
            while start > 0 and words < window_words:
                start -= 1
                words += sentences[start].words
            if not spans or spans[-1] != (start, end):
                spans.append((start, end))
            break
        spans.append((start, end))
        advanced, nxt = 0, start
        while nxt < end and advanced < stride_words:
            advanced += sentences[nxt].words
            nxt += 1
        start = max(nxt, start + 1)

    return [
        Window(
            first,
            last,
            " ".join(s.text for s in sentences[first : last + 1]),
            sum(s.words for s in sentences[first : last + 1]),
        )
        for first, last in spans
    ]

from aicheck.pipeline import Config, analyze
from aicheck.segment import build_windows, qualifying_sentences

HUMAN = "The old harbour was quiet that morning and nobody had come for the boats. "
ROBOT = "Furthermore, robot synergy leverages a multifaceted tapestry of holistic paradigms. "


class KeywordDetector:
    """Stand-in model: a window's score is the share of its sentences mentioning 'robot'."""

    name = "keyword"
    threshold = 0.5

    def score(self, texts):
        return [t.count("robot") / max(t.count("."), 1) for t in texts]


def test_only_prose_sentences_qualify():
    text = (
        "Introduction\n"
        "- a bullet point that should never be analysed, even though it ends with a full stop.\n"
        "1. A numbered item that is also long enough but is still a list entry here.\n"
        "Dr. Smith arrived at 9 a.m. on Tuesday, e.g. before anyone else. Was he early? Yes.\n"
    )
    sentences = qualifying_sentences(text)
    assert [s.text for s in sentences] == [
        "Dr. Smith arrived at 9 a.m. on Tuesday, e.g. before anyone else.",
        "Was he early?",
        "Yes.",
    ]
    assert all(text[s.start : s.end] == s.text for s in sentences)


def test_windows_overlap_and_cover_every_sentence():
    sentences = qualifying_sentences(HUMAN * 120)
    windows = build_windows(sentences, window_words=250, stride_words=125)
    covered = [0] * len(sentences)
    for w in windows:
        assert 250 <= w.words < 250 + 20
        for i in range(w.first, w.last + 1):
            covered[i] += 1
    assert min(covered) >= 1
    assert sum(c >= 2 for c in covered) > len(sentences) * 0.8


def test_short_document_gets_no_score():
    report = analyze(HUMAN * 5, KeywordDetector())
    assert report.status == "insufficient_text"
    assert report.percent is None and report.sentences == []


def test_fully_human_and_fully_ai():
    assert analyze(HUMAN * 60, KeywordDetector()).percent == 0
    report = analyze(ROBOT * 60, KeywordDetector())
    assert report.percent == 100 and all(s.flagged for s in report.sentences)


def test_mixed_document_highlights_the_ai_half():
    text = HUMAN * 60 + "\n" + ROBOT * 60
    report = analyze(text, KeywordDetector())
    assert 35 <= report.percent <= 60
    boundary = text.index("\n")
    flagged = [s for s in report.sentences if s.flagged]
    # Only sentences next to the seam can be mislabelled, never ones deep in a half.
    assert all(s.start > boundary - 1500 for s in flagged)
    assert all(s.flagged for s in report.sentences if s.start > boundary + 1500)


def test_scores_below_display_floor_are_withheld():
    text = HUMAN * 150 + "\n" + ROBOT * 14
    report = analyze(text, KeywordDetector(), Config())
    assert 0 < report.raw_percent < 20
    assert report.display == "*%" and report.percent is None
    assert not report.highlights_shown and not any(s.flagged for s in report.sentences)

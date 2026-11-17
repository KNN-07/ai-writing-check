"""Document-level AI writing report built from overlapping segment scores."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

from .detector import Detector
from .segment import build_windows, count_words, qualifying_sentences


@dataclass(frozen=True)
class Config:
    window_words: int = 250  # "a few hundred words" per segment
    stride_words: int = 125  # ~50% overlap, so most sentences get 2+ scores
    sentence_threshold: float | None = None  # None: use the detector's calibrated value
    min_words: int = 300  # below this no score is produced
    max_words: int = 30_000
    display_floor: int = 20  # 1..floor-1 % is shown as "*%" with no highlights
    paraphrase_threshold: float = 0.5  # second stage: flagged text at/above this was reworded


@dataclass
class SentenceResult:
    start: int
    end: int
    score: float
    flagged: bool
    category: str | None = None  # "generated" | "paraphrased" for flagged sentences


@dataclass
class Report:
    status: str  # "ok" | "insufficient_text" | "too_long"
    message: str
    model: str
    total_words: int
    qualifying_words: int
    percent: int | None = None  # None when withheld or not computed
    display: str = "--"
    highlights_shown: bool = False
    raw_percent: float | None = None  # unrounded, never withheld; for evaluation
    paraphrase_checked: bool = False
    generated_percent: int | None = None  # AI-generated, wording untouched
    paraphrased_percent: int | None = None  # AI-generated, then run through a paraphraser
    sentences: list[SentenceResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def analyze(
    text: str,
    detector: Detector,
    config: Config = Config(),
    paraphrase_detector: Detector | None = None,
) -> Report:
    total_words = count_words(text)
    sentences = qualifying_sentences(text)
    qualifying_words = sum(s.words for s in sentences)
    report = Report("ok", "", detector.name, total_words, qualifying_words)

    if total_words > config.max_words:
        report.status = "too_long"
        report.message = f"Document exceeds {config.max_words:,} words."
        return report
    if qualifying_words < config.min_words:
        report.status = "insufficient_text"
        report.message = (
            f"Only {qualifying_words} words of qualifying prose; at least "
            f"{config.min_words} are needed for a reliable result."
        )
        return report

    windows = build_windows(sentences, config.window_words, config.stride_words)
    window_scores = detector.score([w.text for w in windows])

    # Pool: each sentence takes the mean score of every window that contains it.
    totals = [0.0] * len(sentences)
    counts = [0] * len(sentences)
    for window, score in zip(windows, window_scores):
        for i in range(window.first, window.last + 1):
            totals[i] += score
            counts[i] += 1
    pooled = [total / count for total, count in zip(totals, counts)]

    threshold = detector.threshold if config.sentence_threshold is None else config.sentence_threshold
    flagged = [score >= threshold for score in pooled]
    flagged_words = sum(s.words for s, hit in zip(sentences, flagged) if hit)
    report.raw_percent = 100 * flagged_words / qualifying_words
    percent = round(report.raw_percent)

    if 0 < percent < config.display_floor:
        # Low scores are the least reliable, so the number and highlights are withheld.
        report.display = "*%"
        report.message = (
            f"Some AI-like text was detected, but below {config.display_floor}% the "
            "result is too unreliable to show a number or highlights."
        )
    else:
        report.percent = percent
        report.display = f"{percent}%"
        report.highlights_shown = percent > 0
        report.message = f"{percent}% of the qualifying text is likely AI-generated."

    shown = report.highlights_shown
    report.sentences = [
        SentenceResult(s.start, s.end, round(score, 4), hit and shown, "generated" if hit and shown else None)
        for s, score, hit in zip(sentences, pooled, flagged)
    ]

    if shown and paraphrase_detector is not None:
        # Second stage, only over text already flagged as AI: was it reworded by a tool?
        hot = [w for w in windows if any(flagged[w.first : w.last + 1])]
        totals, counts = [0.0] * len(sentences), [0] * len(sentences)
        for window, score in zip(hot, paraphrase_detector.score([w.text for w in hot])):
            for i in range(window.first, window.last + 1):
                totals[i] += score
                counts[i] += 1
        paraphrased_words = 0
        for i, result in enumerate(report.sentences):
            if result.flagged and totals[i] / counts[i] >= config.paraphrase_threshold:
                result.category = "paraphrased"
                paraphrased_words += sentences[i].words
        report.paraphrase_checked = True
        report.paraphrased_percent = min(percent, round(100 * paraphrased_words / qualifying_words))
        report.generated_percent = percent - report.paraphrased_percent
    return report

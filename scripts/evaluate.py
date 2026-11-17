"""Measure document-level accuracy on the HAP-E human/AI parallel corpus.

Human text is the second ~500-word chunk of a real document; AI text is an
LLM's continuation of the same first chunk, so topics and genres are matched.
A document counts as flagged when its AI percentage reaches the display floor
(20%). Each model is scored once and every sentence threshold is swept from
the stored sentence scores.

    python scripts/evaluate.py --per-class 300 --models <hf-model> [<hf-model> ...]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aicheck.detector import DEFAULT_MODEL, HFDetector  # noqa: E402
from aicheck.pipeline import Config, analyze  # noqa: E402
from aicheck.segment import count_words  # noqa: E402

REPO = "browndw/human-ai-parallel-corpus"
SOURCES = {
    "human": "text_data/hape-text_human-chunk-2.parquet",
    "gpt-4o": "text_data/hape-text_gpt-4o-2024-08-06.parquet",
    "gpt-4o-mini": "text_data/hape-text_gpt-4o-mini-2024-07-18.parquet",
    "llama-3-70B-Inst": "text_data/hape-text_llama-3-70B-Instruct.parquet",
}
THRESHOLDS = (0.5, 0.7, 0.9, 0.95, 0.99, 0.999)


def load(path: str, n: int, seed: int) -> list[str]:
    frame = pd.read_parquet(hf_hub_download(REPO, path, repo_type="dataset"))
    return frame.sample(n=min(n, len(frame)), random_state=seed)["text"].tolist()


def sentence_scores(text: str, detector: HFDetector) -> list[tuple[float, int]] | None:
    report = analyze(text, detector)
    if report.status != "ok":
        return None
    return [(s.score, count_words(text[s.start : s.end])) for s in report.sentences]


def percent_at(document: list[tuple[float, int]], threshold: float) -> float:
    flagged = sum(words for score, words in document if score >= threshold)
    return 100 * flagged / sum(words for _, words in document)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=[DEFAULT_MODEL])
    parser.add_argument("--per-class", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    floor = Config().display_floor
    texts = {name: load(path, args.per_class, args.seed) for name, path in SOURCES.items()}

    for model in args.models:
        started = time.time()
        detector = HFDetector(model, batch_size=32)
        scored: dict[str, list] = {}
        total, done = sum(len(docs) for docs in texts.values()), 0
        for name, docs in texts.items():
            scored[name] = []
            for text in docs:
                result = sentence_scores(text, detector)
                if result:
                    scored[name].append(result)
                done += 1
                if done % 25 == 0 or done == total:
                    print(
                        f"[{model}] {done}/{total} documents ({name}) "
                        f"{time.time() - started:.0f}s",
                        flush=True,
                    )
        print(f"\n== {model}  ({time.time() - started:.0f}s)")
        print("documents:", {name: len(docs) for name, docs in scored.items()})
        print(f"% of documents flagged (AI share >= {floor}%); the human row is the false positive rate")
        print("threshold".ljust(18) + "".join(f"{t:>8}" for t in THRESHOLDS))
        for name, docs in scored.items():
            rates = [
                100 * sum(percent_at(d, t) >= floor for d in docs) / len(docs) for t in THRESHOLDS
            ]
            print(name.ljust(18) + "".join(f"{rate:8.1f}" for rate in rates))


if __name__ == "__main__":
    main()

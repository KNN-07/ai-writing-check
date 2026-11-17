"""Segment-level classifier: P(AI-written) for a few hundred words of prose."""
from __future__ import annotations

import os
import re
import unicodedata
from typing import Protocol

DEFAULT_MODEL = os.environ.get("AICHECK_MODEL", "tabularisai/ai-text-detection")

# Sentence-flagging threshold per model, chosen with scripts/evaluate.py so that
# under 1% of human documents are flagged. Raw classifier scores are not calibrated,
# so the right cut-off differs a lot between models.
MODEL_THRESHOLDS = {
    "tabularisai/ai-text-detection": 0.95,
    "fakespot-ai/roberta-base-ai-text-detection-v1": 0.999,
}
_AI_LABEL_HINTS = ("ai", "fake", "machine", "generated", "gpt", "llm", "label_1")


_INVISIBLE = re.compile("[​-‏⁠﻿­]")


def _normalize(text: str) -> str:
    """Undo cheap evasion tricks: zero-width characters, look-alike glyph forms."""
    text = _INVISIBLE.sub("", unicodedata.normalize("NFKC", text))
    return re.sub(r"\s+", " ", text).strip()


class Detector(Protocol):
    name: str
    threshold: float  # pooled sentence score at/above this counts as AI-written

    def score(self, texts: list[str]) -> list[float]:
        """Return P(AI-written) in [0, 1] for each text."""
        ...


class HFDetector:
    """Any Hugging Face sequence-classification model with a human/AI label pair."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        ai_label: int | None = None,
        batch_size: int = 8,
        max_tokens: int = 512,
    ) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        self.name = model_name
        self.threshold = float(
            os.environ.get("AICHECK_THRESHOLD", MODEL_THRESHOLDS.get(model_name, 0.5))
        )
        self.batch_size = batch_size
        self.max_tokens = max_tokens
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device).eval()

        if ai_label is None and os.environ.get("AICHECK_AI_LABEL"):
            ai_label = int(os.environ["AICHECK_AI_LABEL"])
        self.ai_label = self._find_ai_label() if ai_label is None else ai_label

    def _find_ai_label(self) -> int:
        labels = {int(i): str(name).lower() for i, name in self.model.config.id2label.items()}
        for index, name in labels.items():
            tokens = name.replace("-", " ").replace("_", " ").split()
            if name == "label_1" or any(hint in tokens for hint in _AI_LABEL_HINTS):
                return index
        raise ValueError(
            f"Cannot tell which label of {self.name} means AI-written ({labels}); "
            "pass ai_label= or set AICHECK_AI_LABEL."
        )

    def score(self, texts: list[str]) -> list[float]:
        torch = self._torch
        texts = [_normalize(text) for text in texts]
        scores: list[float] = []
        for i in range(0, len(texts), self.batch_size):
            batch = self.tokenizer(
                texts[i : i + self.batch_size],
                truncation=True,
                max_length=self.max_tokens,
                padding=True,
                return_tensors="pt",
            ).to(self.device)
            with torch.inference_mode():
                logits = self.model(**batch).logits
            if logits.shape[-1] == 1:
                probs = torch.sigmoid(logits[:, 0])
            else:
                probs = torch.softmax(logits, dim=-1)[:, self.ai_label]
            scores.extend(probs.float().cpu().tolist())
        return scores

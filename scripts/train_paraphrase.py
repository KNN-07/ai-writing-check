"""Train and test the second-stage classifier: AI-generated vs AI-generated-then-paraphrased.

Needs a GPU. Steps, each cached under --out so a rerun resumes:
  1. Take LLM-written documents from the HAP-E corpus, split into train and test by topic.
  2. Paraphrase them sentence by sentence with a T5 paraphraser (train and test), and
     paragraph by paragraph with an instruction-tuned LLM (test only, never trained on).
  3. Measure how often the first-stage detector still flags paraphrased documents.
  4. Fine-tune a classifier on ~250-word windows: original AI text vs paraphrased AI text.
  5. Report its accuracy on both paraphrasers and on an existing public 3-class model.

    python scripts/train_paraphrase.py --out /content/exp
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aicheck.detector import HFDetector  # noqa: E402
from aicheck.pipeline import analyze  # noqa: E402
from aicheck.segment import build_windows, qualifying_sentences  # noqa: E402

REPO = "browndw/human-ai-parallel-corpus"
HUMAN = "text_data/hape-text_human-chunk-2.parquet"
AI_SOURCES = {
    "gpt-4o": "text_data/hape-text_gpt-4o-2024-08-06.parquet",
    "gpt-4o-mini": "text_data/hape-text_gpt-4o-mini-2024-07-18.parquet",
    "llama-3-70B-Inst": "text_data/hape-text_llama-3-70B-Instruct.parquet",
}
T5_PARAPHRASER = "humarin/chatgpt_paraphraser_on_T5_base"
LLM_PARAPHRASER = "Qwen/Qwen2.5-1.5B-Instruct"
THREE_CLASS = "MUSKAN17/AI-DETECTOR"  # labels: 0 human, 1 AI-generated, 2 paraphrased
DEVICE = "cuda"


def log(*parts) -> None:
    print(time.strftime("%H:%M:%S"), *parts, flush=True)


def cached(path: Path, build):
    if path.exists():
        return json.loads(path.read_text())
    value = build()
    path.write_text(json.dumps(value))
    return value


def load_corpus(path: str) -> dict[str, str]:
    frame = pd.read_parquet(hf_hub_download(REPO, path, repo_type="dataset"))
    return dict(zip(frame["doc_id"].str.split("@").str[0], frame["text"]))


def windows_of(text: str) -> list[str]:
    return [w.text for w in build_windows(qualifying_sentences(text)) if w.words >= 120]


def t5_paraphrase(docs: list[str]) -> list[str]:
    """Sentence-level rewriting, the way 'word spinner' tools work."""
    tokenizer = AutoTokenizer.from_pretrained(T5_PARAPHRASER)
    model = AutoModelForSeq2SeqLM.from_pretrained(T5_PARAPHRASER).half().to(DEVICE).eval()
    split = [[s.text for s in qualifying_sentences(doc)] for doc in docs]
    flat = [s for sentences in split for s in sentences]
    out: list[str] = []
    for i in range(0, len(flat), 96):
        batch = tokenizer(
            ["paraphrase: " + s for s in flat[i : i + 96]],
            return_tensors="pt", padding=True, truncation=True, max_length=160,
        ).to(DEVICE)
        with torch.inference_mode():
            ids = model.generate(
                **batch, max_new_tokens=160, do_sample=True, top_p=0.92, temperature=0.8,
                repetition_penalty=1.2,
            )
        out.extend(tokenizer.batch_decode(ids, skip_special_tokens=True))
        if (i // 96) % 20 == 0:
            log(f"  T5 paraphraser: {min(i + 96, len(flat))}/{len(flat)} sentences")
    del model
    torch.cuda.empty_cache()
    rebuilt, position = [], 0
    for sentences in split:
        rebuilt.append(" ".join(out[position : position + len(sentences)]))
        position += len(sentences)
    return rebuilt


def llm_paraphrase(docs: list[str]) -> list[str]:
    """Whole-passage rewriting by a chat model, the way people use an LLM to reword."""
    tokenizer = AutoTokenizer.from_pretrained(LLM_PARAPHRASER, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(LLM_PARAPHRASER, torch_dtype=torch.float16)
    model = model.to(DEVICE).eval()
    instruction = (
        "Paraphrase the following text. Keep every idea and roughly the same length, but "
        "change the wording and sentence structure throughout. Output only the paraphrase.\n\n"
    )
    out: list[str] = []
    for i in range(0, len(docs), 12):
        prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": instruction + " ".join(doc.split()[:420])}],
                tokenize=False, add_generation_prompt=True,
            )
            for doc in docs[i : i + 12]
        ]
        batch = tokenizer(prompts, return_tensors="pt", padding=True).to(DEVICE)
        with torch.inference_mode():
            ids = model.generate(**batch, max_new_tokens=800, do_sample=True, temperature=0.7, top_p=0.9)
        out.extend(tokenizer.batch_decode(ids[:, batch["input_ids"].shape[1] :], skip_special_tokens=True))
        log(f"  LLM paraphraser: {min(i + 12, len(docs))}/{len(docs)} documents")
    del model
    torch.cuda.empty_cache()
    return [text.strip() for text in out]


def encode(tokenizer, texts: list[str]):
    return tokenizer(texts, truncation=True, max_length=512, padding=True, return_tensors="pt").to(DEVICE)


def predict(model, tokenizer, texts: list[str]) -> torch.Tensor:
    """Class probabilities for each text."""
    model.eval()
    chunks = []
    for i in range(0, len(texts), 48):
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            chunks.append(torch.softmax(model(**encode(tokenizer, texts[i : i + 48])).logits.float(), -1).cpu())
    return torch.cat(chunks)


def train(base: str, texts: list[str], labels: list[int], epochs: int, out_dir: Path):
    tokenizer = AutoTokenizer.from_pretrained(base)
    model = AutoModelForSequenceClassification.from_pretrained(
        base, num_labels=2, id2label={0: "AI-generated", 1: "AI-paraphrased"},
        label2id={"AI-generated": 0, "AI-paraphrased": 1},
    ).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    steps = epochs * (len(texts) // 16)
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=2e-5, total_steps=steps, pct_start=0.06)
    scaler = torch.amp.GradScaler("cuda")
    step = 0
    for epoch in range(epochs):
        order = list(range(len(texts)))
        random.shuffle(order)
        model.train()
        for i in range(0, len(order) - 15, 16):
            index = order[i : i + 16]
            batch = encode(tokenizer, [texts[j] for j in index])
            target = torch.tensor([labels[j] for j in index], device=DEVICE)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = torch.nn.functional.cross_entropy(model(**batch).logits.float(), target)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            schedule.step()
            step += 1
            if step % 50 == 0 or step == steps:
                log(f"  training: step {step}/{steps} (epoch {epoch + 1}/{epochs}) loss {loss.item():.4f}")
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    return model, tokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="/content/exp")
    parser.add_argument("--train-docs", type=int, default=500, help="per AI source")
    parser.add_argument("--test-docs", type=int, default=150, help="per AI source")
    parser.add_argument("--base", default="FacebookAI/roberta-base")
    parser.add_argument("--epochs", type=int, default=2)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    random.seed(0)
    torch.manual_seed(0)

    human = load_corpus(HUMAN)
    corpora = {name: load_corpus(path) for name, path in AI_SOURCES.items()}
    topics = sorted(set(human).intersection(*corpora.values()))
    random.shuffle(topics)
    test_topics = topics[: args.test_docs]
    train_topics = topics[args.test_docs : args.test_docs + args.train_docs]
    log(f"topics: {len(topics)} shared, {len(train_topics)} train, {len(test_topics)} test (disjoint)")

    train_ai = [corpora[name][t] for name in corpora for t in train_topics]
    test_ai = {name: [corpora[name][t] for t in test_topics] for name in corpora}
    test_ai_flat = [doc for docs in test_ai.values() for doc in docs]
    test_human = [human[t] for t in test_topics]

    log(f"paraphrasing {len(train_ai)} training documents with the T5 paraphraser")
    train_para = cached(out / "train_para_t5.json", lambda: t5_paraphrase(train_ai))
    log(f"paraphrasing {len(test_ai_flat)} test documents with the T5 paraphraser")
    test_para_t5 = cached(out / "test_para_t5.json", lambda: t5_paraphrase(test_ai_flat))
    log(f"paraphrasing {len(test_ai_flat)} test documents with the held-out LLM paraphraser")
    test_para_llm = cached(out / "test_para_llm.json", lambda: llm_paraphrase(test_ai_flat))
    (out / "examples.json").write_text(json.dumps(
        {"original": test_ai_flat[0], "t5": test_para_t5[0], "llm": test_para_llm[0]}, indent=1))

    results: dict = {}
    test_sets = {
        "human": test_human,
        "AI original": test_ai_flat,
        "AI + T5 paraphraser": test_para_t5,
        "AI + LLM paraphraser (held out)": test_para_llm,
    }

    log("stage 1: how often is each kind of document still flagged as AI (>= 20%)?")
    detector = HFDetector(batch_size=32)
    results["stage1_flagged_percent"] = {}
    for name, docs in test_sets.items():
        reports = [r for r in (analyze(doc, detector) for doc in docs) if r.status == "ok"]
        rate = 100 * sum(r.raw_percent >= 20 for r in reports) / len(reports)
        results["stage1_flagged_percent"][name] = {"documents": len(reports), "flagged": round(rate, 1)}
        log(f"  {name:34s} n={len(reports):4d}  flagged {rate:5.1f}%")
    del detector
    torch.cuda.empty_cache()

    test_windows = {name: [w for doc in docs for w in windows_of(doc)] for name, docs in test_sets.items()}

    log(f"existing public 3-class model {THREE_CLASS} on ~250-word windows")
    tokenizer = AutoTokenizer.from_pretrained(THREE_CLASS)
    model = AutoModelForSequenceClassification.from_pretrained(THREE_CLASS).to(DEVICE)
    results["three_class_predictions_percent"] = {}
    for name, windows in test_windows.items():
        picks = predict(model, tokenizer, windows).argmax(-1)
        share = [round(100 * (picks == k).float().mean().item(), 1) for k in range(3)]
        results["three_class_predictions_percent"][name] = dict(zip(["human", "ai", "paraphrased"], share))
        log(f"  {name:34s} n={len(windows):4d}  human {share[0]:5.1f}%  AI {share[1]:5.1f}%  paraphrased {share[2]:5.1f}%")
    del model
    torch.cuda.empty_cache()

    originals = [w for doc in train_ai for w in windows_of(doc)]
    paraphrased = [w for doc in train_para for w in windows_of(doc)]
    log(f"stage 2 training on {len(originals)} original and {len(paraphrased)} paraphrased windows ({args.base})")
    model, tokenizer = train(
        args.base, originals + paraphrased, [0] * len(originals) + [1] * len(paraphrased),
        args.epochs, out / "paraphrase-model",
    )

    log("stage 2: share of windows called 'paraphrased' (P >= 0.5)")
    results["stage2_called_paraphrased_percent"] = {}
    for name, windows in test_windows.items():
        rate = 100 * (predict(model, tokenizer, windows)[:, 1] >= 0.5).float().mean().item()
        results["stage2_called_paraphrased_percent"][name] = {"windows": len(windows), "rate": round(rate, 1)}
        log(f"  {name:34s} n={len(windows):4d}  called paraphrased {rate:5.1f}%")

    (out / "results.json").write_text(json.dumps(results, indent=1))
    log("DONE", out / "results.json")


if __name__ == "__main__":
    main()

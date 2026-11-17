"""Web app: `uvicorn aicheck.server:app` then open http://127.0.0.1:8000"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse

from .detector import Detector, HFDetector
from .extract import extract_text
from .pipeline import analyze

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
STATIC = Path(__file__).parent / "static"

app = FastAPI(title="AI Writing Check")


@lru_cache(maxsize=1)
def get_detector() -> Detector:
    return HFDetector()


@lru_cache(maxsize=1)
def get_paraphrase_detector() -> Detector | None:
    """Optional second stage: set AICHECK_PARAPHRASE_MODEL to a model trained by
    scripts/train_paraphrase.py (label 1 = AI-paraphrased)."""
    model = os.environ.get("AICHECK_PARAPHRASE_MODEL")
    return HFDetector(model, ai_label=1) if model else None


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.post("/api/analyze")
async def analyze_endpoint(
    text: str | None = Form(None), file: UploadFile | None = File(None)
) -> dict:
    if file is not None and file.filename:
        data = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "File is larger than 20 MB.")
        try:
            text = await run_in_threadpool(extract_text, file.filename, data)
        except ValueError as error:
            raise HTTPException(415, str(error)) from error
        except Exception as error:  # corrupt or encrypted document
            raise HTTPException(422, f"Could not read that file: {error}") from error
    if not text or not text.strip():
        raise HTTPException(400, "Provide some text or upload a file.")

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    report = await run_in_threadpool(lambda: analyze(text, get_detector(), paraphrase_detector=get_paraphrase_detector()))
    return {"text": text, **report.to_dict()}

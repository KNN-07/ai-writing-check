"""Turn an uploaded .txt / .md / .docx / .pdf file into plain text."""
from __future__ import annotations

import io
import re

SUPPORTED = (".txt", ".md", ".docx", ".pdf")


def _unwrap_pdf_lines(page_text: str) -> str:
    """PDF text comes one visual line at a time; rejoin wrapped paragraph lines."""
    out: list[str] = []
    for line in (ln.strip() for ln in page_text.splitlines()):
        previous = out[-1] if out else ""
        wrapped = len(previous) > 60 and not re.search(r"[.!?:][\"'”’)\]]*$", previous)
        if line and previous and wrapped:
            joiner = "" if previous.endswith("-") else " "
            out[-1] = previous.rstrip("-") + joiner + line if joiner == "" else previous + " " + line
        else:
            out.append(line)
    return "\n".join(out)


def extract_text(filename: str, data: bytes) -> str:
    name = filename.lower()
    if name.endswith(".pdf"):
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        return "\n\n".join(_unwrap_pdf_lines(page.extract_text() or "") for page in reader.pages)
    if name.endswith(".docx"):
        from docx import Document

        return "\n".join(p.text for p in Document(io.BytesIO(data)).paragraphs)
    if name.endswith((".txt", ".md")):
        return data.decode("utf-8", errors="replace")
    raise ValueError(f"Unsupported file type. Use one of: {', '.join(SUPPORTED)}")

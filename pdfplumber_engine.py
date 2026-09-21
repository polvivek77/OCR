"""pdfplumber runner — PDF only, no OCR (extracts embedded text + tables)."""
from __future__ import annotations

import io
from pathlib import Path

from engines.logging_setup import get_logger
from engines.utils import pdf_embedded_text_length, timer

log = get_logger("ocr.pdfplumber")


def _empty(err: str) -> dict:
    return {
        "engine": "pdfplumber",
        "device": "cpu",
        "elapsed_sec": 0.0,
        "pages": 0,
        "text": "",
        "lines": [],
        "tables": [],
        "error": err,
    }


def extract(file_bytes: bytes, filename: str) -> dict:
    log.info("pdfplumber → %s (%d bytes)", filename, len(file_bytes))
    if not filename.lower().endswith(".pdf"):
        msg = "pdfplumber only supports PDF files."
        log.warning(msg)
        return _empty(msg)

    embedded = pdf_embedded_text_length(file_bytes)
    log.info("pdfplumber: embedded text chars = %d", embedded)
    if embedded == 0:
        msg = ("This PDF has no embedded text (looks scanned/image-based). "
               "pdfplumber cannot OCR — use PaddleOCR / Surya / docTR / "
               "Tesseract / EasyOCR / Docling / Marker instead.")
        log.warning(msg)
        return _empty(msg)

    import pdfplumber

    text_all: list[str] = []
    lines: list[dict] = []
    tables: list[dict] = []
    try:
        with timer() as t:
            with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                for page_idx, page in enumerate(pdf.pages):
                    txt = page.extract_text() or ""
                    if txt:
                        text_all.append(txt)
                        for row in txt.splitlines():
                            if row.strip():
                                lines.append({
                                    "page": page_idx + 1,
                                    "text": row,
                                    "conf": 1.0,
                                    "bbox": None,
                                })
                    for tbl_idx, tbl in enumerate(page.extract_tables() or []):
                        tables.append({
                            "page": page_idx + 1,
                            "index": tbl_idx,
                            "rows": tbl,
                        })
                n_pages = len(pdf.pages)
    except Exception as exc:
        log.exception("pdfplumber failed")
        return _empty(f"{type(exc).__name__}: {exc}")

    log.info(
        "pdfplumber ✓ %.2fs · %d pages · %d lines · %d tables",
        t["elapsed"], n_pages, len(lines), len(tables),
    )
    return {
        "engine": "pdfplumber",
        "device": "cpu",
        "elapsed_sec": round(t["elapsed"], 3),
        "pages": n_pages,
        "text": "\n".join(text_all),
        "lines": lines,
        "tables": tables,
    }


if __name__ == "__main__":
    import sys
    from engines.logging_setup import setup_logging
    setup_logging()
    p = Path(sys.argv[1])
    out = extract(p.read_bytes(), p.name)
    print(out["text"][:500])
    print(f"[{out['engine']}] — {out['elapsed_sec']}s, {len(out['lines'])} tables")

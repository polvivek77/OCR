"""Tesseract OCR runner (CPU-only; requires system Tesseract binary)."""
from __future__ import annotations

import shutil
from pathlib import Path

from engines.logging_setup import get_logger
from engines.utils import load_as_pil, timer

log = get_logger("ocr.tesseract")


def _find_tesseract() -> str | None:
    exe = shutil.which("tesseract")
    if exe:
        return exe
    for candidate in (
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    ):
        if Path(candidate).exists():
            return candidate
    return None


def extract(file_bytes: bytes, filename: str) -> dict:
    log.info("Tesseract → %s (%d bytes)", filename, len(file_bytes))
    import pytesseract

    tess = _find_tesseract()
    if tess:
        pytesseract.pytesseract.tesseract_cmd = tess
        log.info("Tesseract binary: %s", tess)
    else:
        msg = ("Tesseract binary not found. Install from "
               "https://github.com/UB-Mannheim/tesseract/wiki and add it to PATH.")
        log.error(msg)
        return {
            "engine": "Tesseract", "device": "cpu", "elapsed_sec": 0.0, "pages": 0,
            "text": "", "lines": [], "tables": [], "error": msg,
        }

    try:
        pages = load_as_pil(file_bytes, filename)
        log.info("Tesseract: rasterised %d page(s)", len(pages))
        text_all: list[str] = []
        lines: list[dict] = []
        with timer() as t:
            for page_idx, img in enumerate(pages):
                data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
                page_lines = 0
                for i, txt in enumerate(data["text"]):
                    if not txt.strip():
                        continue
                    conf_raw = data["conf"][i]
                    try:
                        conf = float(conf_raw) / 100.0
                    except (ValueError, TypeError):
                        conf = 0.0
                    lines.append(
                        {
                            "page": page_idx + 1,
                            "text": txt,
                            "conf": conf,
                            "bbox": [
                                data["left"][i],
                                data["top"][i],
                                data["left"][i] + data["width"][i],
                                data["top"][i] + data["height"][i],
                            ],
                        }
                    )
                    text_all.append(txt)
                    page_lines += 1
                log.debug("Tesseract: page %d → %d tokens", page_idx + 1, page_lines)
    except Exception as e:
        log.exception("Tesseract failed")
        return {
            "engine": "Tesseract", "device": "cpu", "elapsed_sec": 0.0, "pages": 0,
            "text": "", "lines": [], "tables": [], "error": f"{type(e).__name__}: {e}",
        }

    log.info("Tesseract ✓ %.2fs · %d pages · %d tokens",
             t["elapsed"], len(pages), len(lines))
    return {
        "engine": "Tesseract",
        "device": "cpu",
        "elapsed_sec": round(t["elapsed"], 3),
        "pages": len(pages),
        "text": " ".join(text_all),
        "lines": lines,
        "tables": [],
    }


if __name__ == "__main__":
    import sys
    from engines.logging_setup import setup_logging
    setup_logging()
    p = Path(sys.argv[1])
    out = extract(p.read_bytes(), p.name)
    print(out["text"][:500])
    print(f"[{out['engine']}] {out['device']} — {out['elapsed_sec']}s, {len(out['lines'])} lines")

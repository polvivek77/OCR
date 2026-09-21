"""EasyOCR runner."""
from __future__ import annotations

from pathlib import Path

from engines.logging_setup import get_logger
from engines.utils import has_torch_gpu, load_as_pil, pil_to_np, timer

log = get_logger("ocr.easyocr")

_reader = None


def _get_reader(use_gpu: bool):
    global _reader
    if _reader is None:
        log.info("EasyOCR: initialising (gpu=%s)…", use_gpu)
        import easyocr
        _reader = easyocr.Reader(["en"], gpu=use_gpu, verbose=False)
        log.info("EasyOCR: ready")
    return _reader


def extract(file_bytes: bytes, filename: str) -> dict:
    log.info("EasyOCR → %s (%d bytes)", filename, len(file_bytes))
    gpu = has_torch_gpu()
    device = "cuda" if gpu else "cpu"
    try:
        reader = _get_reader(use_gpu=gpu)
        pages = load_as_pil(file_bytes, filename)
        log.info("EasyOCR: rasterised %d page(s)", len(pages))

        text_all: list[str] = []
        lines: list[dict] = []
        with timer() as t:
            for page_idx, img in enumerate(pages):
                arr = pil_to_np(img)
                page_hits = 0
                for bbox, txt, conf in reader.readtext(arr):
                    lines.append({"page": page_idx + 1, "text": txt, "conf": float(conf), "bbox": bbox})
                    text_all.append(txt)
                    page_hits += 1
                log.debug("EasyOCR: page %d → %d boxes", page_idx + 1, page_hits)
    except Exception as e:
        log.exception("EasyOCR failed")
        return {
            "engine": "EasyOCR", "device": device, "elapsed_sec": 0.0, "pages": 0,
            "text": "", "lines": [], "tables": [], "error": f"{type(e).__name__}: {e}",
        }

    log.info("EasyOCR ✓ %.2fs · %d pages · %d lines · device=%s",
             t["elapsed"], len(pages), len(lines), device)
    return {
        "engine": "EasyOCR",
        "device": device,
        "elapsed_sec": round(t["elapsed"], 3),
        "pages": len(pages),
        "text": "\n".join(text_all),
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

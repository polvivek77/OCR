"""docTR runner (PyTorch backend)."""
from __future__ import annotations

import os
from pathlib import Path

from engines.logging_setup import get_logger
from engines.utils import has_torch_gpu, timer

os.environ.setdefault("USE_TORCH", "1")
log = get_logger("ocr.doctr")

_predictor = None


def _get_predictor():
    global _predictor
    if _predictor is None:
        log.info("docTR: loading ocr_predictor (pretrained)…")
        from doctr.models import ocr_predictor
        _predictor = ocr_predictor(pretrained=True)
        if has_torch_gpu():
            try:
                _predictor = _predictor.cuda()
                log.info("docTR: moved predictor to CUDA")
            except Exception as e:
                log.warning("docTR: could not move to CUDA (%s), staying on CPU", e)
        log.info("docTR: ready")
    return _predictor


def extract(file_bytes: bytes, filename: str) -> dict:
    log.info("docTR → %s (%d bytes)", filename, len(file_bytes))
    device = "cuda" if has_torch_gpu() else "cpu"
    try:
        from doctr.io import DocumentFile

        if filename.lower().endswith(".pdf"):
            doc = DocumentFile.from_pdf(file_bytes)
        else:
            doc = DocumentFile.from_images([file_bytes])
        log.info("docTR: loaded %d page(s)", len(doc))

        predictor = _get_predictor()
        with timer() as t:
            result = predictor(doc)
        export = result.export()
    except Exception as e:
        log.exception("docTR failed")
        return {
            "engine": "docTR", "device": device, "elapsed_sec": 0.0, "pages": 0,
            "text": "", "lines": [], "tables": [], "error": f"{type(e).__name__}: {e}",
        }

    text_all: list[str] = []
    lines: list[dict] = []
    for page_idx, page in enumerate(export["pages"]):
        for block in page.get("blocks", []):
            for line in block.get("lines", []):
                words = [w["value"] for w in line.get("words", [])]
                if not words:
                    continue
                txt = " ".join(words)
                confs = [w.get("confidence", 0.0) for w in line["words"]]
                lines.append(
                    {
                        "page": page_idx + 1,
                        "text": txt,
                        "conf": float(sum(confs) / len(confs)) if confs else 0.0,
                        "bbox": line.get("geometry"),
                    }
                )
                text_all.append(txt)
    log.info("docTR ✓ %.2fs · %d pages · %d lines · device=%s",
             t["elapsed"], len(export["pages"]), len(lines), device)
    return {
        "engine": "docTR",
        "device": device,
        "elapsed_sec": round(t["elapsed"], 3),
        "pages": len(export["pages"]),
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

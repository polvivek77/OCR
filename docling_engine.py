"""Docling runner (IBM) — layout-aware PDF/image parsing."""
from __future__ import annotations

import tempfile
from pathlib import Path

from engines.logging_setup import get_logger
from engines.utils import has_torch_gpu, timer

log = get_logger("ocr.docling")

_converter = None


def _get_converter():
    global _converter
    if _converter is None:
        device = "cuda" if has_torch_gpu() else "cpu"
        log.info("Docling: initialising DocumentConverter (device=%s)…", device)
        from docling.datamodel.accelerator_options import AcceleratorOptions
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption

        # Pin ML pipeline stages to CUDA when it is available. PDF rasterisation
        # and extracting embedded text are inherently CPU-side operations.
        options = PdfPipelineOptions()
        options.accelerator_options = AcceleratorOptions(device=device)
        _converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=options),
            }
        )
        log.info("Docling: ready")
    return _converter


def extract(file_bytes: bytes, filename: str) -> dict:
    log.info("Docling → %s (%d bytes)", filename, len(file_bytes))
    device = "cuda" if has_torch_gpu() else "cpu"
    try:
        converter = _get_converter()
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            p = Path(td) / filename
            p.write_bytes(file_bytes)
            with timer() as t:
                result = converter.convert(str(p))
    except Exception as e:
        log.exception("Docling failed")
        return {
            "engine": "Docling", "device": device, "elapsed_sec": 0.0, "pages": 0,
            "text": "", "lines": [], "tables": [], "error": f"{type(e).__name__}: {e}",
        }

    doc = result.document
    md_text = doc.export_to_markdown()
    tables_out: list[dict] = []
    for i, tbl in enumerate(getattr(doc, "tables", []) or []):
        try:
            # Newer Docling API takes doc=... explicitly; fall back for older versions.
            try:
                df = tbl.export_to_dataframe(doc=doc)
            except TypeError:
                df = tbl.export_to_dataframe()
            tables_out.append({"page": i + 1, "index": i, "rows": df.values.tolist()})
        except Exception as ex:
            log.warning("Docling: could not export table %d — %s", i, ex)
            continue

    n_pages = getattr(doc, "num_pages", lambda: 1)() if callable(getattr(doc, "num_pages", None)) else 1
    lines = [{"page": 1, "text": line, "conf": 1.0, "bbox": None} for line in md_text.splitlines() if line.strip()]
    log.info("Docling ✓ %.2fs · %d pages · %d lines · %d tables · device=%s",
             t["elapsed"], n_pages, len(lines), len(tables_out), device)
    return {
        "engine": "Docling",
        "device": device,
        "elapsed_sec": round(t["elapsed"], 3),
        "pages": n_pages,
        "text": md_text,
        "lines": lines,
        "tables": tables_out,
    }


if __name__ == "__main__":
    import sys
    from engines.logging_setup import setup_logging
    setup_logging()
    p = Path(sys.argv[1])
    out = extract(p.read_bytes(), p.name)
    print(out["text"][:500])
    print(f"[{out['engine']}] {out['device']} — {out['elapsed_sec']}s")

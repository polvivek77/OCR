"""Camelot runner — PDF tables only. Needs Ghostscript installed on the system."""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

from engines.logging_setup import get_logger
from engines.utils import pdf_embedded_text_length, timer

log = get_logger("ocr.camelot")


def _ensure_ghostscript_on_path() -> str | None:
    """Windows: locate an installed Ghostscript and add its bin folder to the DLL
    search path so Camelot's ctypes binding finds gsdll64.dll without a shell restart."""
    if os.name != "nt":
        return None
    roots = [Path(r"C:/Program Files/gs"), Path(r"C:/Program Files (x86)/gs")]
    for root in roots:
        if not root.exists():
            continue
        for child in sorted(root.iterdir(), reverse=True):
            bin_dir = child / "bin"
            if (bin_dir / "gsdll64.dll").exists() or (bin_dir / "gsdll32.dll").exists():
                os.environ["PATH"] = f"{bin_dir};{os.environ.get('PATH', '')}"
                if hasattr(os, "add_dll_directory") and sys.version_info >= (3, 8):
                    try:
                        os.add_dll_directory(str(bin_dir))
                    except (FileNotFoundError, OSError):
                        pass
                log.debug("Ghostscript located at %s", bin_dir)
                return str(bin_dir)
    log.warning("Ghostscript not found under C:/Program Files/gs or (x86)/gs")
    return None


def _empty(err: str) -> dict:
    return {
        "engine": "Camelot",
        "device": "cpu",
        "elapsed_sec": 0.0,
        "pages": 0,
        "text": "",
        "lines": [],
        "tables": [],
        "error": err,
    }


def extract(file_bytes: bytes, filename: str) -> dict:
    log.info("Camelot → %s (%d bytes)", filename, len(file_bytes))
    if not filename.lower().endswith(".pdf"):
        msg = "Camelot only supports PDF files."
        log.warning(msg)
        return _empty(msg)

    embedded = pdf_embedded_text_length(file_bytes)
    log.info("Camelot: embedded text chars = %d", embedded)
    if embedded == 0:
        msg = ("This PDF is image-based (scanned). Camelot only works on text-based PDFs. "
               "Use Docling / Marker for OCR-driven table extraction instead.")
        log.warning(msg)
        return _empty(msg)

    _ensure_ghostscript_on_path()
    import camelot

    # ignore_cleanup_errors: Camelot's PyPDF2/Ghostscript backend may keep the file
    # handle open past our `with` block on Windows, which crashes rmtree.
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        pdf_path = Path(td) / filename
        pdf_path.write_bytes(file_bytes)

        tables_out: list[dict] = []
        text_all: list[str] = []
        with timer() as t:
            try:
                log.info("Camelot: trying lattice flavor…")
                tables = camelot.read_pdf(str(pdf_path), pages="all", flavor="lattice")
                if len(tables) == 0:
                    log.info("Camelot: lattice found 0 tables, trying stream…")
                    tables = camelot.read_pdf(str(pdf_path), pages="all", flavor="stream")
            except Exception as e:
                log.exception("Camelot failed")
                return _empty(f"{type(e).__name__}: {e}. Ghostscript may be missing.")
            for i, tbl in enumerate(tables):
                df = tbl.df
                tables_out.append(
                    {
                        "page": int(tbl.page) if hasattr(tbl, "page") else i + 1,
                        "index": i,
                        "rows": df.values.tolist(),
                        "accuracy": getattr(tbl, "accuracy", None),
                    }
                )
                text_all.append(df.to_csv(index=False))
    log.info("Camelot ✓ %.2fs · %d tables", t["elapsed"], len(tables_out))
    return {
        "engine": "Camelot",
        "device": "cpu",
        "elapsed_sec": round(t["elapsed"], 3),
        "pages": len(tables_out),
        "text": "\n\n".join(text_all),
        "lines": [],
        "tables": tables_out,
    }


if __name__ == "__main__":
    from engines.logging_setup import setup_logging
    setup_logging()
    p = Path(sys.argv[1])
    out = extract(p.read_bytes(), p.name)
    print(f"[{out['engine']}] — {out['elapsed_sec']}s, {len(out['tables'])} tables")

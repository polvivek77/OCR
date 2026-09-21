"""Marker runner — PDF/image → Markdown, layout & table aware."""
from __future__ import annotations

import tempfile
from pathlib import Path

from engines.llamacpp_helper import LlamaServerNotFound, ensure_llama_server
from engines.logging_setup import get_logger, preload_windows_dlls
from engines.utils import has_torch_gpu, timer

preload_windows_dlls()

log = get_logger("ocr.marker")

_state: dict = {}


def _load(device: str):
    if _state:
        return
    # Marker delegates OCR to Surya, which spawns `llama-server`. Point it at
    # our project-local install if we have one.
    ensure_llama_server(auto_install=False)
    log.info("Marker: loading models…")
    from marker.converters.pdf import PdfConverter
    from marker.models import create_model_dict
    _state["models"] = create_model_dict(device=device)
    _state["Converter"] = PdfConverter
    log.info("Marker: ready")


def extract(file_bytes: bytes, filename: str) -> dict:
    log.info("Marker → %s (%d bytes)", filename, len(file_bytes))
    device = "cuda" if has_torch_gpu() else "cpu"
    try:
        from marker.output import text_from_rendered
        _load(device)

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            p = Path(td) / filename
            p.write_bytes(file_bytes)
            with timer() as t:
                converter = _state["Converter"](artifact_dict=_state["models"])
                rendered = converter(str(p))
                md_text, _meta, _images = text_from_rendered(rendered)
    except LlamaServerNotFound as e:
        log.error("Marker: %s", e)
        return {
            "engine": "Marker", "device": device, "elapsed_sec": 0.0, "pages": 0,
            "text": "", "lines": [], "tables": [],
            "error": (
                "Marker uses Surya for OCR, which needs the llama.cpp `llama-server` "
                "binary. Install it once by running:\n"
                "    python scripts/install_llama_cpp.py"
            ),
        }
    except Exception as e:
        log.exception("Marker failed")
        return {
            "engine": "Marker", "device": device, "elapsed_sec": 0.0, "pages": 0,
            "text": "", "lines": [], "tables": [], "error": f"{type(e).__name__}: {e}",
        }

    lines = [{"page": 1, "text": line, "conf": 1.0, "bbox": None} for line in md_text.splitlines() if line.strip()]
    log.info("Marker ✓ %.2fs · %d lines · device=%s", t["elapsed"], len(lines), device)
    return {
        "engine": "Marker",
        "device": device,
        "elapsed_sec": round(t["elapsed"], 3),
        "pages": 1,
        "text": md_text,
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
    print(f"[{out['engine']}] {out['device']} — {out['elapsed_sec']}s")

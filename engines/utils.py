"""Shared helpers: GPU detection, file loading, timing."""
from __future__ import annotations

import io
import subprocess
import sys
import time
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Iterator

import numpy as np
from PIL import Image


def has_torch_gpu() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


@lru_cache(maxsize=1)
def has_paddle_gpu() -> bool:
    """Probe Paddle CUDA without loading it beside Torch on Windows.

    Current Windows CUDA wheels for Torch and Paddle bundle incompatible cuDNN
    DLLs.  The Streamlit process imports Torch for several engines, so probe
    Paddle in a clean child interpreter instead of poisoning the parent process
    with a failed import.

    Escape hatch: PADDLE_FORCE_CPU=1 forces the answer to False without any
    import — needed on GPUs whose compute capability isn't in Paddle's wheel
    (Blackwell / RTX 50-series / sm_120).
    """
    import os as _os
    if _os.environ.get("PADDLE_FORCE_CPU", "").lower() in ("1", "true", "yes"):
        return False
    if sys.platform == "win32":
        probe = "import paddle; print(int(paddle.device.is_compiled_with_cuda()))"
        try:
            result = subprocess.run(
                [sys.executable, "-c", probe],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            return result.returncode == 0 and result.stdout.rstrip().endswith("1")
        except (OSError, subprocess.TimeoutExpired):
            return False
    try:
        import paddle  # type: ignore
        return bool(paddle.device.is_compiled_with_cuda())
    except Exception:
        return False


def device_str(prefer: str = "torch") -> str:
    """Return a friendly device string for the selected backend."""
    if prefer == "paddle":
        return "gpu" if has_paddle_gpu() else "cpu"
    return "cuda" if has_torch_gpu() else "cpu"


@contextmanager
def timer() -> Iterator[dict]:
    t = {"start": time.perf_counter(), "elapsed": 0.0}
    try:
        yield t
    finally:
        t["elapsed"] = time.perf_counter() - t["start"]


def load_as_pil(file_bytes: bytes, filename: str) -> list[Image.Image]:
    """Return a list of PIL images (1+ pages) from an uploaded file."""
    name = filename.lower()
    if name.endswith(".pdf"):
        import pymupdf  # PyMuPDF (import via the modern name, not the `fitz` alias)
        doc = pymupdf.open(stream=file_bytes, filetype="pdf")
        pages = []
        for page in doc:
            pix = page.get_pixmap(dpi=200)
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            pages.append(img)
        doc.close()
        return pages
    return [Image.open(io.BytesIO(file_bytes)).convert("RGB")]


def pil_to_np(img: Image.Image) -> np.ndarray:
    return np.array(img)


def save_temp(file_bytes: bytes, filename: str, tmpdir: Path) -> Path:
    tmpdir.mkdir(parents=True, exist_ok=True)
    p = tmpdir / filename
    p.write_bytes(file_bytes)
    return p


def pdf_embedded_text_length(file_bytes: bytes) -> int:
    """How many characters of embedded text a PDF contains. 0 usually means
    it's a scanned/image-only PDF and OCR is required. Non-PDF returns -1."""
    try:
        import pymupdf
        doc = pymupdf.open(stream=file_bytes, filetype="pdf")
        total = 0
        for page in doc:
            total += len(page.get_text("text") or "")
        doc.close()
        return total
    except Exception:
        return -1


def pdf_page_count(file_bytes: bytes) -> int:
    """Number of pages in a PDF, or 1 for images, or 0 on failure."""
    try:
        import pymupdf
        doc = pymupdf.open(stream=file_bytes, filetype="pdf")
        n = len(doc)
        doc.close()
        return n
    except Exception:
        return 0


def slice_pdf(file_bytes: bytes, pages: list[int]) -> bytes:
    """Return a new PDF containing only the given 1-indexed page numbers, in order.

    ``pages`` is a de-duplicated, sorted, clamped list. Passing an empty list or
    the full page set returns a re-emitted PDF of every page (identity slice)."""
    import pymupdf
    src = pymupdf.open(stream=file_bytes, filetype="pdf")
    total = len(src)
    if not pages:
        pages = list(range(1, total + 1))
    # normalise: clamp, dedupe, sort, 1-indexed
    keep = sorted({p for p in pages if 1 <= p <= total})
    dst = pymupdf.open()
    for p in keep:
        dst.insert_pdf(src, from_page=p - 1, to_page=p - 1)
    buf = dst.tobytes()
    dst.close()
    src.close()
    return buf

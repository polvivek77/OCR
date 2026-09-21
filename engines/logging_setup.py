"""Centralised logging: writes to ./app.log (rotating) and also mirrors to console.

Call `setup_logging()` once at process start (app.py does this). Everywhere else
use `get_logger(__name__)`.
"""
from __future__ import annotations

import logging
import logging.handlers
import os
import warnings
from pathlib import Path

LOG_FILE = Path(__file__).resolve().parent.parent / "app.log"
_configured = False
_dll_preloaded = False


def preload_windows_dlls() -> None:
    _preload_windows_dlls()


def _preload_windows_dlls() -> None:
    """Import torch BEFORE paddle so their OpenMP DLLs don't clash.

    Windows-specific: if PaddlePaddle 3.3 is imported first, `torch\\lib\\shm.dll`
    later fails with WinError 127 (missing procedure). Torch's DLLs are compatible
    when it loads first, so we always preload it here. Also silences a few
    other Windows-specific noises at the environment level.
    """
    global _dll_preloaded
    if _dll_preloaded or os.name != "nt":
        _dll_preloaded = True
        return
    # HuggingFace symlinks warning on non-dev-mode Windows
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    try:
        import torch  # noqa: F401
    except Exception:
        # No torch available — nothing to do
        pass
    _dll_preloaded = True


def setup_logging(level: int | str = None) -> Path:
    """Configure the root logger to write to app.log + console. Idempotent."""
    global _configured
    _preload_windows_dlls()  # must happen before paddle/paddleocr get imported
    if _configured:
        return LOG_FILE

    level = level or os.environ.get("OCR_LOG_LEVEL", "INFO")
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    fmt = logging.Formatter(
        "[%(asctime)s] [%(levelname)-8s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Rotating file: 5 MB × 3 backups
    file_handler = logging.handlers.RotatingFileHandler(
        LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(fmt)
    file_handler.setLevel(level)

    # Console mirror
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    console.setLevel(level)

    root = logging.getLogger()
    root.setLevel(level)
    # Drop any pre-existing handlers so we don't double-log. Close them first
    # so their file handles are released — otherwise the Streamlit rerun leaks
    # them and emits a ResourceWarning next time setup_logging runs.
    for h in list(root.handlers):
        root.removeHandler(h)
        try:
            h.close()
        except Exception:
            pass
    root.addHandler(file_handler)
    root.addHandler(console)

    # Route warnings.warn(...) through logging
    logging.captureWarnings(True)
    warnings.simplefilter("default")

    # Silence third-party noise we can't influence from user code.
    # Every filter here corresponds to a real warning observed in app.log.
    _noisy_warning_patterns = [
        # PaddlePaddle
        (r".*No ccache found.*",                       UserWarning),
        # Surya (Pydantic V2 class-based config)
        (r".*class-based `config` is deprecated.*",    DeprecationWarning),
        # EasyOCR / torch quantization
        (r".*torch\.ao\.quantization is deprecated.*", DeprecationWarning),
        (r".*torch\.quantize_per_tensor.*",            UserWarning),
        (r".*torch\.quantize_per_tensor.*",            DeprecationWarning),
        # Torch DataLoader on CPU-only machines
        (r".*'pin_memory' argument is set as true.*",  UserWarning),
        # Docling internal API deprecations
        (r".*force_full_page_ocr.*",                   DeprecationWarning),
        (r".*generate_page_images=True.*",             DeprecationWarning),
        (r".*TableItem\.export_to_dataframe.*",        DeprecationWarning),
        (r".*ResourceWarning: unclosed.*",             ResourceWarning),
        # Surya cleanup slop we can't fix from user code
        (r".*unclosed file.*ocr_error_server\.log.*",  ResourceWarning),
        (r".*subprocess \d+ is still running.*",       ResourceWarning),
        (r".*unclosed event loop.*",                   ResourceWarning),
        (r".*unclosed file.*app\.log.*",               ResourceWarning),
        # PyMuPDF fitz alias
        (r".*`fitz` API is deprecated.*",              DeprecationWarning),
    ]
    for pattern, category in _noisy_warning_patterns:
        warnings.filterwarnings("ignore", message=pattern, category=category)

    # Tame chatty libraries so app.log stays readable
    for noisy in (
        "urllib3", "PIL", "matplotlib", "httpx", "httpcore",
        "huggingface_hub", "filelock", "streamlit.runtime.caching",
        "MatchingPostProcessor",  # docling table post-processor spam
        "docling_core.types.doc.items.table.table",
        "docling_ibm_models",
        "paddle", "paddlex", "paddleocr",  # ANSI-coloured "model already cached" prints
        "transformers", "surya", "marker",
        "doctr", "doctr.utils.data",       # doctr's "Using downloaded & verified file"
        "easyocr",
        "playa", "playa.font",             # camelot pdfminer/playa font-glyph noise
        "pdfminer", "pdfminer.pdfinterp",
        "camelot", "camelot.core", "camelot.parsers",
    ):
        logging.getLogger(noisy).setLevel(logging.ERROR)

    _configured = True
    log = logging.getLogger("ocr.setup")
    log.info("Logging initialised → %s (level=%s)", LOG_FILE, logging.getLevelName(level))
    return LOG_FILE


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def tail_log(lines: int = 200) -> str:
    """Return the last `lines` lines of app.log, or a placeholder."""
    if not LOG_FILE.exists():
        return "(app.log not created yet)"
    try:
        with open(LOG_FILE, encoding="utf-8", errors="replace") as f:
            data = f.readlines()
        return "".join(data[-lines:])
    except Exception as e:
        return f"(failed to read {LOG_FILE}: {e})"

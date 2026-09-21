"""Streamlit UI to compare OCR / PDF extraction engines on the metrics that matter:
text accuracy, number accuracy, tables, speed, GPU, layout understanding."""
from __future__ import annotations

import os
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("PYTHONUTF8", "1")

import importlib
import shutil
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


def _docker_running() -> bool:
    """Return True if `docker info` succeeds within 2 s. Used by the sidebar
    to warn when Surya's vllm backend has no daemon to talk to."""
    exe = shutil.which("docker")
    if not exe:
        return False
    try:
        return subprocess.run(
            [exe, "info"],
            capture_output=True,
            timeout=2,
        ).returncode == 0
    except (subprocess.TimeoutExpired, OSError):
        return False

# Force UTF-8 on Windows stdio so libraries printing progress bars don't crash.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import pandas as pd
import streamlit as st

from engines.logging_setup import LOG_FILE, get_logger, setup_logging, tail_log

setup_logging()  # writes ./app.log; must run before other engine imports emit logs
log = get_logger("ocr.app")

from engines.llamacpp_helper import (
    LlamaServerNotFound,
    install_llama_cpp_windows,
    locate_llama_server,
)
from engines.metrics import evaluate
from engines.utils import (
    has_paddle_gpu,
    has_torch_gpu,
    load_as_pil,
    pdf_embedded_text_length,
    pdf_page_count,
    slice_pdf,
)
from engines.viz import draw_boxes

st.set_page_config(page_title="OCR Benchmark", layout="wide")


# ---------------------------------------------------------------------------
# Engine registry — capability flags drive the "Capabilities" matrix + summary
# ---------------------------------------------------------------------------
@dataclass
class Engine:
    name: str
    module: str
    pdf: bool
    image: bool
    tables: str          # "no" | "basic" | "good" | "best"
    layout: str          # "no" | "basic" | "good" | "best"
    gpu_capable: bool
    gpu_backend: str     # "torch" | "paddle" | "-"
    notes: str


ENGINES: list[Engine] = [
    Engine("PaddleOCR",  "paddleocr_engine",  True, True,  "no",    "basic", True,  "paddle", "Fast multilingual OCR"),
    Engine("Surya OCR",  "suryaocr_engine",   True, True,  "no",    "good",  True,  "torch",  "Modern transformer OCR"),
    Engine("docTR",      "doctr_engine",      True, True,  "no",    "good",  True,  "torch",  "Torch or TF backend"),
    Engine("Tesseract",  "tesseract_engine",  True, True,  "no",    "basic", False, "-",      "CPU-only classic OCR"),
    Engine("EasyOCR",    "easyocr_engine",    True, True,  "no",    "basic", True,  "torch",  "80+ languages"),
    Engine("pdfplumber", "pdfplumber_engine", True, False, "good",  "text",  False, "-",      "CPU-only digital PDF text + tables"),
    Engine("Camelot",    "camelot_engine",    True, False, "best",  "tables",False, "-",      "CPU-only PDF tables via Ghostscript"),
    Engine("Docling",    "docling_engine",    True, True,  "good",  "best",  True,  "torch",  "IBM structured output"),
    Engine("Marker",     "marker_engine",     True, True,  "good",  "best",  True,  "torch",  "Excellent PDF→Markdown"),
]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def _run(engine: Engine, file_bytes: bytes, filename: str) -> dict:
    log.info("─── run engine=%s file=%s size=%d ───", engine.name, filename, len(file_bytes))
    started = time.perf_counter()
    try:
        mod = importlib.import_module(engine.module)
        fn: Callable = mod.extract
        result = fn(file_bytes, filename)
        elapsed = time.perf_counter() - started
        if result.get("error"):
            log.warning("%s finished with reported error after %.2fs: %s",
                        engine.name, elapsed, str(result["error"])[:200])
        else:
            log.info("%s finished OK in %.2fs (device=%s, lines=%d, tables=%d)",
                     engine.name, elapsed, result.get("device"),
                     len(result.get("lines", [])), len(result.get("tables", [])))
        return result
    except Exception as e:
        log.exception("%s crashed uncaught", engine.name)
        return {
            "engine": engine.name,
            "device": "-",
            "elapsed_sec": 0.0,
            "pages": 0,
            "text": "",
            "lines": [],
            "tables": [],
            "error": f"{type(e).__name__}: {e}\n\n{traceback.format_exc()[:1200]}",
        }


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def _sidebar() -> list[Engine]:
    st.sidebar.title("OCR benchmark")

    torch_gpu = has_torch_gpu()
    paddle_gpu = has_paddle_gpu()
    st.sidebar.subheader("GPU status")
    st.sidebar.markdown(
        f"- PyTorch CUDA: {'🟢 yes' if torch_gpu else '⚪ no (CPU)'}\n"
        f"- Paddle CUDA: {'🟢 yes' if paddle_gpu else '⚪ no (CPU)'}"
    )

    st.sidebar.subheader("Engines to run")
    if st.sidebar.button("Select all"):
        for eng in ENGINES:
            st.session_state[f"eng_{eng.name}"] = True
    if st.sidebar.button("Clear all"):
        for eng in ENGINES:
            st.session_state[f"eng_{eng.name}"] = False

    # PaddleOCR speed tweak — cuts CPU time roughly in half by skipping
    # optional preprocessing models (doc orientation, unwarping, textline).
    fast_paddle = st.sidebar.checkbox(
        "PaddleOCR fast mode (skip preprocessing)",
        value=os.environ.get("PADDLE_FAST_MODE", "").lower() in ("1", "true", "yes"),
        help="Halves CPU inference time; slightly lower accuracy on rotated/warped pages.",
    )
    os.environ["PADDLE_FAST_MODE"] = "1" if fast_paddle else "0"

    selected: list[Engine] = []
    defaults = {"Tesseract", "EasyOCR", "pdfplumber"}
    for eng in ENGINES:
        key = f"eng_{eng.name}"
        if key not in st.session_state:
            st.session_state[key] = eng.name in defaults
        checked = st.sidebar.checkbox(
            f"{eng.name}",
            key=key,
            help=f"{eng.notes} · GPU: {eng.gpu_backend if eng.gpu_capable else 'no'}",
        )
        if checked:
            selected.append(eng)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Optional dependencies")

    # llama-server for Surya llamacpp backend
    llama = locate_llama_server()
    if llama:
        st.sidebar.markdown(f"- llama-server: 🟢 `{Path(llama).name}`")
    else:
        st.sidebar.markdown("- llama-server: ⚪ missing (needed by Surya llamacpp backend)")
        if st.sidebar.button("Install llama.cpp now"):
            with st.spinner("Downloading llama.cpp… (≈18 MB)"):
                try:
                    install_llama_cpp_windows(cuda=has_torch_gpu())
                    st.sidebar.success("llama-server installed. Re-run engines.")
                except LlamaServerNotFound as e:
                    st.sidebar.error(str(e))

    # Docker for Surya vllm backend (auto-picked when CUDA is present)
    if has_torch_gpu():
        docker_ok = _docker_running()
        st.sidebar.markdown(
            "- Docker Desktop: "
            + ("🟢 running (Surya vllm backend available)"
               if docker_ok
               else "⚪ not detected (Surya will fail unless SURYA_INFERENCE_BACKEND=llamacpp)")
        )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Log file")
    st.sidebar.caption(f"`{LOG_FILE}`")
    if LOG_FILE.exists():
        try:
            size_kb = LOG_FILE.stat().st_size / 1024
            st.sidebar.caption(f"Current size: {size_kb:.1f} KB")
        except Exception:
            pass
    return selected


# ---------------------------------------------------------------------------
# Capabilities matrix (static, informational)
# ---------------------------------------------------------------------------
def _render_capabilities() -> None:
    rows = []
    torch_gpu = has_torch_gpu()
    paddle_gpu = has_paddle_gpu()
    for e in ENGINES:
        gpu_available = (
            (e.gpu_backend == "torch" and torch_gpu)
            or (e.gpu_backend == "paddle" and paddle_gpu)
        )
        rows.append(
            {
                "Engine": e.name,
                "PDF": "✅" if e.pdf else "—",
                "Image": "✅" if e.image else "—",
                "Tables": e.tables,
                "Layout": e.layout,
                "GPU-capable": "✅" if e.gpu_capable else "—",
                "GPU backend": e.gpu_backend,
                "GPU available now": "✅" if gpu_available else "—",
                "Notes": e.notes,
            }
        )
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


# ---------------------------------------------------------------------------
# Per-engine result rendering
# ---------------------------------------------------------------------------
def _render_result(eng: Engine, res: dict, ground_truth: str, first_page_image) -> dict:
    if res.get("error"):
        err_text = str(res["error"])
        # Split "TypeName: message\n\ntraceback…" into a headline + full details
        headline, _sep, rest = err_text.partition("\n\n")
        st.error(headline)
        if rest.strip():
            with st.expander("Full traceback"):
                st.code(rest.strip(), language="text")
        return {
            "engine": eng.name, "device": "-", "gpu_used": False,
            "time_sec": 0.0, "pages": 0, "lines": 0, "tables": 0,
            "char_acc": 0.0, "word_acc": 0.0,
            "num_precision": 0.0, "num_recall": 0.0, "num_f1": 0.0,
            "error": True,
        }

    device = res.get("device", "-")
    gpu_used = device in ("cuda", "gpu")
    metrics = evaluate(ground_truth, res.get("text", "")) if ground_truth else None

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Device", f"🟢 {device.upper()}" if gpu_used else device.upper())
    c2.metric("Time (s)", res.get("elapsed_sec", 0.0))
    c3.metric("Pages", res.get("pages", 0))
    c4.metric("Lines", len(res.get("lines", [])))
    c5.metric("Tables", len(res.get("tables", [])))
    if metrics:
        c6.metric("Char acc", f"{metrics['char_acc'] * 100:.1f}%")

    if metrics:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Word accuracy", f"{metrics['word_acc'] * 100:.1f}%")
        n = metrics["numbers"]
        m2.metric("Numbers matched", f"{n['matched']} / {n['ref_count']}")
        m3.metric("Number precision", f"{n['precision'] * 100:.1f}%")
        m4.metric("Number recall", f"{n['recall'] * 100:.1f}%")
        if n["missed"] or n["extra"]:
            with st.expander("Number diff (first 20 each)"):
                st.markdown(f"**Missed (in ref, not in output):** `{n['missed']}`")
                st.markdown(f"**Extra (in output, not in ref):** `{n['extra']}`")

    text_tab, layout_tab, lines_tab, tables_tab = st.tabs(
        ["Text", "Layout (boxes)", "Lines", "Tables"]
    )
    with text_tab:
        st.text_area("Extracted text", res.get("text", ""), height=280, key=f"txt_{eng.name}")
    with layout_tab:
        if first_page_image is not None and res.get("lines"):
            annotated = draw_boxes(first_page_image, [ln for ln in res["lines"] if ln.get("page") == 1])
            st.image(annotated, caption=f"{eng.name} — page 1 detections", width="stretch")
        else:
            st.info("No bounding boxes available for this engine (or no image page).")
    with lines_tab:
        if res.get("lines"):
            df = pd.DataFrame(res["lines"])
            st.dataframe(df, width="stretch", height=280)
        else:
            st.info("No line-level output.")
    with tables_tab:
        if res.get("tables"):
            for tbl in res["tables"]:
                st.caption(f"Page {tbl['page']} — table #{tbl['index']}"
                           + (f" · accuracy {tbl.get('accuracy'):.1f}%" if tbl.get("accuracy") else ""))
                df = pd.DataFrame(tbl["rows"])
                st.dataframe(df, width="stretch")
        else:
            st.info("No tables detected.")

    return {
        "engine": res.get("engine", eng.name),
        "device": device,
        "gpu_used": gpu_used,
        "time_sec": res.get("elapsed_sec", 0.0),
        "pages": res.get("pages", 0),
        "lines": len(res.get("lines", [])),
        "tables": len(res.get("tables", [])),
        "char_acc": metrics["char_acc"] if metrics else None,
        "word_acc": metrics["word_acc"] if metrics else None,
        "num_precision": metrics["numbers"]["precision"] if metrics else None,
        "num_recall": metrics["numbers"]["recall"] if metrics else None,
        "num_f1": metrics["numbers"]["f1"] if metrics else None,
        "error": False,
    }


# ---------------------------------------------------------------------------
# Logs panel
# ---------------------------------------------------------------------------
def _render_logs() -> None:
    st.caption(f"Live tail of `{LOG_FILE.name}` — last 200 lines. Full file at `{LOG_FILE}`.")
    c1, c2 = st.columns([1, 5])
    with c1:
        if st.button("🔄 Refresh logs"):
            st.rerun()
    with c2:
        if LOG_FILE.exists():
            st.download_button(
                "⬇ Download app.log",
                LOG_FILE.read_bytes(),
                file_name="app.log",
                mime="text/plain",
            )
    st.code(tail_log(200), language="log")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    st.title("OCR / PDF extraction benchmark")
    st.caption(
        "Upload a PDF or image, optionally paste the ground-truth text, "
        "and compare engines on text/number accuracy, tables, layout, and speed."
    )

    with st.expander("📊 Engine capabilities", expanded=False):
        _render_capabilities()

    selected = _sidebar()

    col_up, col_gt = st.columns([1, 1])
    with col_up:
        uploaded = st.file_uploader(
            "Upload a PDF or image",
            type=["pdf", "png", "jpg", "jpeg", "tif", "tiff", "bmp", "webp"],
        )
    with col_gt:
        ground_truth = st.text_area(
            "Ground-truth text (optional — enables accuracy scoring)",
            height=180,
            placeholder="Paste the exact text that should appear in the document…",
        )

    if not uploaded:
        st.info("Upload a file to begin.")
        with st.expander(f"📄 Logs ({LOG_FILE.name})", expanded=False):
            _render_logs()
        return
    if not selected:
        st.warning("Pick at least one engine from the sidebar.")
        return

    file_bytes = uploaded.read()
    filename = uploaded.name
    log.info("=== new upload: %s (%d bytes) ===", filename, len(file_bytes))

    is_pdf = filename.lower().endswith(".pdf")
    embedded_chars = pdf_embedded_text_length(file_bytes) if is_pdf else -1
    total_pages = pdf_page_count(file_bytes) if is_pdf else 1
    if is_pdf:
        log.info("PDF embedded-text char count = %d, page count = %d",
                 embedded_chars, total_pages)

    try:
        all_pages = load_as_pil(file_bytes, filename)
    except Exception:
        log.exception("Failed to rasterise upload for preview")
        all_pages = []
    n_pages = len(all_pages) if all_pages else total_pages

    st.markdown("### Preview")
    if is_pdf and embedded_chars == 0:
        st.warning(
            "⚠ This PDF appears to be **scanned / image-based** (0 embedded characters). "
            "**pdfplumber** and **Camelot** will not extract anything — they need selectable text. "
            "For OCR-driven text + tables use **Docling** or **Marker**; for plain OCR text "
            "any of PaddleOCR / Surya / docTR / Tesseract / EasyOCR will work."
        )
    elif is_pdf and embedded_chars > 0:
        st.caption(f"PDF has {embedded_chars} embedded characters — text is selectable.")

    # Preview navigator — pick which page to show + which page(s) to send to engines
    preview_col, mode_col = st.columns([1, 2])
    with preview_col:
        preview_page = st.number_input(
            "Preview page",
            min_value=1,
            max_value=max(n_pages, 1),
            value=1,
            step=1,
            key="preview_page",
        )
    with mode_col:
        page_mode = st.radio(
            "Pages to send to engines",
            options=["All pages", "Current page only", "Range"],
            horizontal=True,
            key="page_mode",
        )
    selected_pages: list[int] = list(range(1, n_pages + 1))
    if page_mode == "Current page only":
        selected_pages = [int(preview_page)]
    elif page_mode == "Range":
        r1, r2 = st.columns(2)
        with r1:
            page_from = st.number_input(
                "From page", min_value=1, max_value=max(n_pages, 1), value=1, step=1,
                key="page_from",
            )
        with r2:
            page_to = st.number_input(
                "To page", min_value=1, max_value=max(n_pages, 1),
                value=max(n_pages, 1), step=1, key="page_to",
            )
        lo, hi = sorted((int(page_from), int(page_to)))
        selected_pages = list(range(lo, hi + 1))

    st.caption(
        f"Total pages: **{n_pages}** · will run on **{len(selected_pages)}** "
        f"page(s): `{selected_pages if len(selected_pages) <= 20 else str(selected_pages[:20])+'…'}`"
    )

    # Show the previewed page image
    first_page_image = None
    if all_pages:
        pi = min(int(preview_page) - 1, len(all_pages) - 1)
        first_page_image = all_pages[pi]
        st.image(
            first_page_image,
            caption=f"{filename} · page {int(preview_page)} of {n_pages}",
            width="stretch",
        )
    else:
        st.caption(f"Uploaded: {filename} ({len(file_bytes) // 1024} KB)")

    if not st.button("▶ Run selected engines", type="primary"):
        return

    # If a subset of pages was chosen and this is a PDF, produce a sliced PDF
    # containing only those pages, and send THAT to the engines. Non-PDFs (images)
    # only have one "page", so the selector is a no-op.
    payload_bytes = file_bytes
    payload_filename = filename
    if is_pdf and page_mode != "All pages" and selected_pages != list(range(1, n_pages + 1)):
        try:
            payload_bytes = slice_pdf(file_bytes, selected_pages)
            payload_filename = f"pages-{'-'.join(str(p) for p in selected_pages[:6])}"
            payload_filename = (payload_filename[:60] + "-of-" + filename)
            log.info("Sliced PDF: %d selected page(s) → %d bytes",
                     len(selected_pages), len(payload_bytes))
        except Exception:
            log.exception("Failed to slice PDF, falling back to full document")
            payload_bytes = file_bytes
            payload_filename = filename

    log.info("Running %d engines: %s", len(selected), ", ".join(e.name for e in selected))

    st.markdown("### Results")
    progress = st.progress(0.0, text="Starting…")
    summaries: list[dict] = []
    for i, eng in enumerate(selected):
        progress.progress(i / len(selected), text=f"Running {eng.name}…")
        with st.expander(f"🔹 {eng.name} — {eng.notes}", expanded=True):
            with st.spinner(f"Running {eng.name}…"):
                res = _run(eng, payload_bytes, payload_filename)
            summary = _render_result(eng, res, ground_truth, first_page_image)
            summaries.append(summary)
    progress.progress(1.0, text="Done")

    st.markdown("### Overall comparison")
    df = pd.DataFrame(summaries)
    st.dataframe(df, width="stretch", hide_index=True)

    if not df.empty:
        st.markdown("#### ⏱ Speed (lower is better)")
        st.bar_chart(df.set_index("engine")["time_sec"])

        if ground_truth:
            acc_df = df[["engine", "char_acc", "word_acc", "num_f1"]].dropna()
            if not acc_df.empty:
                st.markdown("#### 🎯 Accuracy (higher is better)")
                st.bar_chart(acc_df.set_index("engine"))

        st.download_button(
            "⬇ Download summary CSV",
            df.to_csv(index=False).encode(),
            file_name="ocr_benchmark.csv",
            mime="text/csv",
        )

    with st.expander(f"📄 Logs ({LOG_FILE.name})", expanded=False):
        _render_logs()


if __name__ == "__main__":
    main()

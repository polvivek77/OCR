"""Clean-process PaddleOCR worker. Never import Torch here.

Windows-specific: Paddle's GPU wheel needs cuDNN + CUDA-runtime DLLs. On
Python 3.8+ Windows Python only searches directories added via
``os.add_dll_directory``; ``os.environ['PATH']`` alone is ignored. We probe
the common install locations and add every one that exists BEFORE importing
paddle. If cuDNN can still not be loaded, we fall back to CPU silently.
"""
from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path


# ---------------------------------------------------------------------------
# CUDA / cuDNN DLL discovery — must run BEFORE `import paddle`
# ---------------------------------------------------------------------------
def _register_dll_dirs() -> list[str]:
    """Register every CUDA / cuDNN bin folder we can find. Returns paths added."""
    if os.name != "nt" or not hasattr(os, "add_dll_directory"):
        return []

    candidates: list[Path] = []
    # 1. nvidia-* pip wheels living beside paddle in the same venv
    venv_site = Path(sys.executable).parent.parent / "Lib" / "site-packages"
    for sub in ("nvidia/cudnn/bin", "nvidia/cublas/bin", "nvidia/cuda_runtime/bin",
                "nvidia/cuda_nvrtc/bin", "nvidia/cufft/bin"):
        p = venv_site / sub
        if p.exists():
            candidates.append(p)

    # 2. Stand-alone NVIDIA installs (both v9.x + the CUDA Toolkit)
    program_files_roots = [
        Path(r"C:\Program Files\NVIDIA\CUDNN"),
        Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"),
    ]
    for root in program_files_roots:
        if not root.exists():
            continue
        for version_dir in sorted(root.iterdir(), reverse=True):
            for bin_dir in version_dir.rglob("bin"):
                if bin_dir.is_dir():
                    candidates.append(bin_dir)

    added: list[str] = []
    for path in candidates:
        try:
            os.add_dll_directory(str(path))
            os.environ["PATH"] = str(path) + os.pathsep + os.environ.get("PATH", "")
            added.append(str(path))
        except (FileNotFoundError, OSError):
            pass
    return added


_DLL_DIRS = _register_dll_dirs()

# Also make sure the venv itself sees these — helps subprocess CLI callers.
os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")

from engines.utils import load_as_pil, pil_to_np  # noqa: E402


def _json_value(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _lines(result, page: int) -> list[dict]:
    data = getattr(result, "json", result)
    if isinstance(data, dict) and "res" in data:
        data = data["res"]
    if isinstance(data, dict):
        texts = data.get("rec_texts", [])
        scores = data.get("rec_scores", [])
        boxes = data.get("rec_polys", data.get("dt_polys", []))
        return [
            {
                "page": page,
                "text": str(text),
                "conf": float(scores[idx]) if idx < len(scores) else 0.0,
                "bbox": _json_value(boxes[idx]) if idx < len(boxes) else None,
            }
            for idx, text in enumerate(texts)
        ]
    return []


def _log(msg: str) -> None:
    print(f"[paddle-worker] {msg}", file=sys.stderr, flush=True)


def _purge_paddle_modules() -> None:
    """After a failed `import paddle`, sys.modules keeps the half-initialised
    module which causes `partially initialized module 'paddle' has no attribute
    'tensor' (circular import)` on the next import. Drop every paddle* module
    so the next import re-runs paddle/__init__.py from scratch."""
    for name in list(sys.modules):
        if name == "paddle" or name.startswith(("paddle.", "paddlex", "paddleocr")):
            sys.modules.pop(name, None)


def _init_ocr(requested_device: str):
    """Try requested device first, fall back to CPU if cuDNN cannot load.

    Set env `PADDLE_FORCE_CPU=1` to skip GPU entirely — needed on Blackwell
    (RTX 50-series) GPUs where paddlepaddle-gpu cu126 wheels have no sm_120
    kernels and any GPU init corrupts the paddle module state.
    """
    if os.environ.get("PADDLE_FORCE_CPU", "").lower() in ("1", "true", "yes"):
        _log("PADDLE_FORCE_CPU set — skipping GPU attempt")
        requested_device = "cpu"

    # Optional fast mode: skip the doc-orientation, doc-unwarping and
    # textline-orientation preprocessing models. Cuts CPU inference time
    # roughly in half for typical single-column pages.
    fast = os.environ.get("PADDLE_FAST_MODE", "").lower() in ("1", "true", "yes")
    if fast:
        _log("PADDLE_FAST_MODE — disabling textline_orientation preprocessing")

    last_exc: Exception | None = None
    for device in (requested_device, "cpu"):
        try:
            from paddleocr import PaddleOCR
            return (
                PaddleOCR(
                    lang="en",
                    device=device,
                    enable_mkldnn=False,
                    use_textline_orientation=not fast,
                    use_doc_orientation_classify=not fast,
                    use_doc_unwarping=not fast,
                ),
                device,
            )
        except (OSError, RuntimeError, AttributeError) as exc:
            last_exc = exc
            msg = str(exc).lower()
            gpu_dll_problem = "cudnn" in msg or "winerror 127" in msg
            circular = "partially initialized module 'paddle'" in msg
            if device != "cpu" and (gpu_dll_problem or circular):
                print(
                    f"[paddle-worker] {device} init failed ({type(exc).__name__}); "
                    "purging paddle modules and retrying on CPU.",
                    file=sys.stderr,
                    flush=True,
                )
                _purge_paddle_modules()
                continue
            raise
    raise RuntimeError(f"PaddleOCR could not initialise on any device: {last_exc}")


def _run_ocr(ocr, pages: list) -> list[dict]:
    lines: list[dict] = []
    for page_idx, image in enumerate(pages, start=1):
        arr = pil_to_np(image)
        _log(f"page {page_idx}: input shape={arr.shape} dtype={arr.dtype}")
        raw = ocr.predict(arr)
        _log(f"page {page_idx}: got {len(raw)} raw result(s)")
        for result in raw:
            data = getattr(result, "json", None)
            if isinstance(data, dict) and "res" in data:
                inner = data["res"]
                _log(f"page {page_idx}: rec_texts={len(inner.get('rec_texts', []))} "
                     f"rec_polys={len(inner.get('rec_polys', inner.get('dt_polys', [])))}")
            page_lines = _lines(result, page_idx)
            lines.extend(page_lines)
    return lines


def main(input_name: str, output_name: str, device: str) -> None:
    if _DLL_DIRS:
        _log(f"DLL dirs registered: {_DLL_DIRS}")

    ocr, used_device = _init_ocr(device)
    try:
        import paddle as _paddle
        _log(f"paddle.device.get_device() = {_paddle.device.get_device()!r}")
    except Exception:
        pass

    pages = load_as_pil(Path(input_name).read_bytes(), input_name)
    _log(f"rasterised {len(pages)} page(s) at DPI=200")
    lines = _run_ocr(ocr, pages)
    _log(f"{used_device}: extracted {len(lines)} line(s) total")

    # Diagnostic: if GPU produced nothing, try CPU once. This has happened when
    # Paddle silently mis-loads a CUDA kernel — the pipeline runs but detection
    # returns an empty structure with no error.
    if used_device == "gpu" and not lines and pages:
        _log("GPU returned 0 lines — retrying once on CPU to check pipeline")
        _purge_paddle_modules()
        try:
            cpu_ocr, _ = _init_ocr("cpu")
            cpu_lines = _run_ocr(cpu_ocr, pages)
        except Exception as exc:
            _log(f"CPU diagnostic failed: {type(exc).__name__}: {exc}")
            cpu_lines = []
        if cpu_lines:
            _log(f"CPU diagnostic returned {len(cpu_lines)} lines — GPU pipeline is silently broken. Using CPU result.")
            lines = cpu_lines
            used_device = "cpu"

    payload = {
        "pages": len(pages),
        "text": "\n".join(line["text"] for line in lines),
        "lines": lines,
        "tables": [],
        "device": used_device,
    }
    Path(output_name).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    output_arg = sys.argv[2] if len(sys.argv) > 2 else None
    try:
        main(*sys.argv[1:])
    except Exception as exc:
        # Persist a machine-readable error so the parent engine can display a
        # single-line message instead of a 100-line traceback. Exit 0 so the
        # parent treats it as a graceful "engine reported error", not a crash.
        traceback.print_exc()
        payload = {
            "pages": 0,
            "text": "",
            "lines": [],
            "tables": [],
            "device": "cpu",
            "error": f"{type(exc).__name__}: {exc}",
        }
        if output_arg:
            try:
                Path(output_arg).write_text(
                    json.dumps(payload, ensure_ascii=False),
                    encoding="utf-8",
                )
            except OSError:
                pass
        sys.exit(0 if output_arg else 1)

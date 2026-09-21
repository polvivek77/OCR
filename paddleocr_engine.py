"""PaddleOCR runner isolated from Torch on Windows.

Torch and Paddle CUDA wheels bundle mutually incompatible cuDNN DLLs on
Windows. Paddle OCR therefore runs in a clean child interpreter; the main
Streamlit process remains free to use Torch-based OCR engines on the GPU.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

from engines.logging_setup import get_logger
from engines.utils import has_paddle_gpu, timer

log = get_logger("ocr.paddleocr")
_WORKER = Path(__file__).with_name("paddleocr_worker.py")


def extract(file_bytes: bytes, filename: str) -> dict:
    log.info("PaddleOCR → %s (%d bytes)", filename, len(file_bytes))
    requested_device = "gpu" if has_paddle_gpu() else "cpu"
    suffix = Path(filename).suffix or ".png"

    try:
        with tempfile.TemporaryDirectory(prefix="ocr-paddle-", ignore_cleanup_errors=True) as tmp:
            input_path = Path(tmp) / f"input{suffix}"
            output_path = Path(tmp) / "result.json"
            input_path.write_bytes(file_bytes)

            with timer() as t:
                proc = subprocess.run(
                    [sys.executable, str(_WORKER),
                     str(input_path), str(output_path), requested_device],
                    capture_output=True,
                    text=True,
                    timeout=600,
                    check=False,
                )
            if proc.returncode:
                detail = (proc.stderr or proc.stdout).strip()
                raise RuntimeError(detail or f"Paddle worker exited with code {proc.returncode}")
            if proc.stderr.strip():
                # Worker logs go through stderr — forward the [paddle-worker] lines
                # to the main log so users can see what the child process was doing
                # without opening a subprocess terminal.
                for line in proc.stderr.splitlines():
                    line = line.rstrip()
                    if line.startswith("[paddle-worker]"):
                        log.info(line)
                    elif line.strip():
                        log.debug("paddle-worker stderr: %s", line)
            result = json.loads(output_path.read_text(encoding="utf-8"))
            # Worker may have written a graceful error JSON on total failure.
            if result.get("error"):
                log.warning("PaddleOCR worker reported: %s", result["error"])
                return {
                    "engine": "PaddleOCR", "device": result.get("device", requested_device),
                    "elapsed_sec": round(t["elapsed"], 3), "pages": 0,
                    "text": "", "lines": [], "tables": [], "error": result["error"],
                }
    except Exception as exc:
        log.exception("PaddleOCR failed")
        return {
            "engine": "PaddleOCR", "device": requested_device,
            "elapsed_sec": 0.0, "pages": 0,
            "text": "", "lines": [], "tables": [],
            "error": f"{type(exc).__name__}: {exc}",
        }

    used_device = result.pop("device", requested_device)
    result.update(engine="PaddleOCR", device=used_device,
                  elapsed_sec=round(t["elapsed"], 3))
    log.info(
        "PaddleOCR ✓ %.2fs · %d pages · %d lines · device=%s",
        t["elapsed"], result.get("pages", 0), len(result.get("lines", [])), used_device,
    )
    return result


if __name__ == "__main__":
    path = Path(sys.argv[1])
    output = extract(path.read_bytes(), path.name)
    print(output["text"][:500])
    print(f"[{output['engine']}] {output['device']} — {output['elapsed_sec']}s, {len(output['lines'])} lines")

# OCR / PDF Extraction Benchmark

Compare text, number, table, layout, and speed across popular OCR & document-parsing tools:
**PaddleOCR, Surya OCR, docTR, Tesseract, EasyOCR, pdfplumber, Camelot, Docling, Marker**.

Each engine has its own file (e.g. `paddleocr_engine.py`, `suryaocr_engine.py`) so you can
run it standalone, and there is a Streamlit app (`app.py`) that lets you upload a PDF/image
and run any subset side-by-side.

Every engine that can use a GPU checks for one at runtime and gracefully falls back to CPU.

---

## 1. Set up the virtual environment (with `uv`)

```powershell
# From the project directory
uv venv --python 3.11
.venv\Scripts\activate

# Install everything
uv pip install -r requirements.txt
```

> **Windows / GPU note.** `torch` on PyPI is CPU-only. For CUDA, replace it with:
> ```powershell
> uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
> ```
> Same for `paddlepaddle` — CPU wheel by default. For GPU:
> ```powershell
> uv pip install paddlepaddle-gpu==3.2.2 --index https://www.paddlepaddle.org.cn/packages/stable/cu129/
> ```

### System dependencies (not installable via pip)
- **Tesseract** — https://github.com/UB-Mannheim/tesseract/wiki (Windows)
- **Ghostscript** (needed by Camelot) — https://ghostscript.com/releases/gsdnld.html
- **llama.cpp `llama-server` binary** (needed by Surya OCR 0.22+ and Marker 2.0+).
  There is a one-command installer that fetches the latest Windows build:
  ```powershell
  python scripts\install_llama_cpp.py --cuda
  ```
  This drops ~50 MB into `.tools/llama-cpp/` and the engines find it automatically.
  You can also click **"Install llama.cpp now"** in the Streamlit sidebar.

### Troubleshooting install

- **PaddleOCR: `NotImplementedError … onednn_instruction.cc:118`** — a Windows-only bug
  in PaddlePaddle 3.3's oneDNN executor. Fixed in this project by passing
  `enable_mkldnn=False`. If you see it, make sure you're on the latest `paddleocr_engine.py`.
- **`OSError: [WinError 127] … torch\lib\shm.dll`** — Windows-only. Happens when
  `paddle` is imported before `torch`. Fixed automatically by
  `engines/logging_setup.py:preload_windows_dlls()` which loads torch first.
- **Surya returns 0 characters / is extremely slow** — Surya 0.22 uses a VLM through
  llama.cpp. Installing the CPU llama.cpp build makes recognition CPU-bound even when
  PyTorch reports CUDA. Install the CUDA build with `python scripts\install_llama_cpp.py --cuda`,
  then set `LLAMA_CPP_BINARY` to `.tools\llama-cpp\cuda\llama-server.exe` and restart the app.

---

## 2. Run the Streamlit app

```powershell
streamlit run app.py
```

Pick engines in the sidebar, upload a PDF or image, and click **Run selected engines**.

---

## 3. Run any engine by itself

Each engine is directly runnable on a single file:

```powershell
python paddleocr_engine.py sample.pdf
python suryaocr_engine.py sample.png
python doctr_engine.py sample.pdf
python tesseract_engine.py sample.jpg
python easyocr_engine.py sample.png
python pdfplumber_engine.py sample.pdf
python camelot_engine.py sample.pdf
python docling_engine.py sample.pdf
python marker_engine.py sample.pdf
```

Each script prints the first 500 characters of the extracted text plus the device
used and elapsed time — a quick sanity check without launching the UI.

---

## 4. What each engine is good at

| Engine       | PDF | Image | Tables | Layout | GPU | Notes                                         |
|--------------|-----|-------|--------|--------|-----|-----------------------------------------------|
| PaddleOCR    | yes | yes   | okay   | okay   | yes | Fast, multilingual                            |
| Surya OCR    | yes | yes   | —      | good   | yes | Modern transformer OCR                        |
| docTR        | yes | yes   | —      | good   | yes | Torch or TF backend                           |
| Tesseract    | yes | yes   | —      | basic  | no  | CPU, needs system binary                      |
| EasyOCR      | yes | yes   | —      | basic  | yes | 80+ languages                                 |
| pdfplumber   | yes | no    | good   | text   | no  | Digital PDFs only, no OCR                     |
| Camelot      | yes | no    | best   | tables | no  | Needs Ghostscript                             |
| Docling      | yes | yes   | good   | best   | opt | IBM, structured output                        |
| Marker       | yes | yes   | good   | best   | opt | Excellent PDF → Markdown                      |

---

## 5. Troubleshooting

- **PaddlePaddle install fails** on Python 3.13/3.14 — use Python 3.11 (`uv venv --python 3.11`).
- **Surya / Marker first run is slow** — models are downloaded from HuggingFace and cached.
- **Camelot error** — install Ghostscript and restart your terminal.
- **Tesseract not found** — install the binary and either add it to `PATH` or edit the
  path in `tesseract_engine.py`.

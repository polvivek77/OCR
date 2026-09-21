#!/usr/bin/env python3
"""Debug Paddle GPU Configuration"""

import os
import sys
from pathlib import Path

print("=" * 70)
print("PADDLE GPU DEBUG")
print("=" * 70)

# Check cuDNN DLL paths
cudnn_paths = [
    r"C:\Program Files\NVIDIA\CUDNN\v9.26\bin\13.4\x64",
    r"C:\Program Files\NVIDIA\CUDNN\v9.26\bin\13.4",
    r"C:\Program Files\NVIDIA\CUDNN\v9.26",
]

print("\n[1] Checking cuDNN DLL locations:")
for path in cudnn_paths:
    p = Path(path)
    if p.exists():
        dlls = list(p.glob("*cudnn*.dll"))
        print(f"    ✓ {path}")
        if dlls:
            for dll in dlls[:3]:
                print(f"       - {dll.name}")
        else:
            print(f"       (no DLLs found)")
    else:
        print(f"    ✗ {path} (NOT FOUND)")

print("\n[2] Checking system PATH:")
path_env = os.environ.get('PATH', '').split(os.pathsep)
cuda_paths = [p for p in path_env if 'cuda' in p.lower() or 'cudnn' in p.lower()]

if cuda_paths:
    print("    ✓ CUDA/cuDNN paths found in PATH:")
    for p in cuda_paths[:5]:
        print(f"       - {p}")
    if len(cuda_paths) > 5:
        print(f"       ... and {len(cuda_paths)-5} more")
else:
    print("    ✗ No CUDA/cuDNN paths in system PATH")
    print("       Add: C:\\Program Files\\NVIDIA\\CUDNN\\v9.26\\bin\\13.4\\x64")

print("\n[3] Checking environment variables:")
for var in ['CUDA_PATH', 'CUDNN_PATH', 'CUDA_HOME', 'PADDLE_ENABLE_GPU']:
    val = os.environ.get(var)
    if val:
        print(f"    ✓ {var} = {val}")
    else:
        print(f"    - {var} (not set)")

print("\n[4] Importing Paddle:")
try:
    import paddle
    print(f"    ✓ Paddle {paddle.__version__} imported")
    
    device = paddle.device.get_device()
    print(f"    ✓ Device: {device}")
    
    if device == 'gpu':
        print(f"    ✓✓ GPU MODE ACTIVE!")
        try:
            import paddle.device as device_lib
            n_gpus = paddle.device.cuda.device_count()
            print(f"    ✓ GPUs available: {n_gpus}")
            
            for i in range(n_gpus):
                name = paddle.device.cuda.get_device_name(i)
                print(f"       GPU {i}: {name}")
        except:
            pass
    else:
        print(f"    ⚠ Running on CPU (expected GPU)")
        
except Exception as e:
    print(f"    ✗ Error importing Paddle:")
    print(f"       {type(e).__name__}: {str(e)[:200]}")
    import traceback
    print("\n    Full traceback:")
    traceback.print_exc()

print("\n[5] Testing PaddleOCR initialization:")
try:
    from paddleocr import PaddleOCR
    print(f"    ✓ PaddleOCR importable")
    
    # Try to initialize (don't download models)
    print(f"    ℹ Initializing PaddleOCR (this may take a moment)...")
    ocr = PaddleOCR(use_gpu=(paddle.device.get_device() == 'gpu'))
    print(f"    ✓ PaddleOCR initialized successfully!")
    
except Exception as e:
    print(f"    ✗ Error initializing PaddleOCR:")
    print(f"       {type(e).__name__}: {str(e)[:300]}")

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

if 'paddle' in sys.modules:
    import paddle
    device = paddle.device.get_device()
    
    if device == 'gpu':
        print("✓ Paddle is configured for GPU - your OCR will run on GPU!")
    else:
        print("✗ Paddle is on CPU - need to fix cuDNN path")
        print("\nTo fix:")
        print("  1. Win+R → sysdm.cpl → Advanced → Environment Variables")
        print("  2. Add Path: C:\\Program Files\\NVIDIA\\CUDNN\\v9.26\\bin\\13.4\\x64")
        print("  3. Restart your terminal/IDE")
        print("  4. Run this script again")
else:
    print("✗ Could not import Paddle - check installation")

print("=" * 70)
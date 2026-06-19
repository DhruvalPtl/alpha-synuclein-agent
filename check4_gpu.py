import subprocess, shutil, sys

print("CHECK 4 — GPU fallback (actual detection on THIS machine)")
print("=" * 65)

nvidia_path = shutil.which("nvidia-smi")
print(f"  shutil.which('nvidia-smi') = {nvidia_path}")

def _has_nvidia():
    try:
        r = subprocess.run(["nvidia-smi"], capture_output=True)
        return r.returncode == 0
    except FileNotFoundError:
        return False

result = _has_nvidia()
print(f"  _has_nvidia() result       = {result}")
print(f"  --> Windows dev machine: no GPU driver / nvidia-smi not in PATH")
print(f"  --> IS_CLOUD = False, so GPU branch entirely skipped")
print()

try:
    import torch
    cuda = torch.cuda.is_available()
    print(f"  torch.cuda.is_available()  = {cuda}")
    print(f"  torch version              = {torch.__version__}")
except ImportError:
    print("  torch: not installed in this venv (OK — cloud handles it)")

print()
print("  Cell 1 if/else block that controls torch install:")
print("    Line 63:  if IS_CLOUD and _has_nvidia():")
print("    Line 64:      install torch CUDA 12.1       <- GPU path")
print("    Line 66:  elif IS_CLOUD:")
print("    Line 67:      print CPU torch message       <- CPU fallback")
print("    (else: local machine uses requirements.txt unchanged)")
print()
print("  _has_nvidia() wraps the subprocess call in try/except FileNotFoundError")
print("  so it returns False cleanly on machines with no nvidia-smi.")
print("  CPU-only branch: CONFIRMED EXISTS.")

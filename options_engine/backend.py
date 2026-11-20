"""Array-backend helpers: CuPy on NVIDIA GPUs, NumPy everywhere else.

Every routine that can run on the GPU takes an ``xp`` module (``numpy`` or
``cupy``), so the *same* code path is what gets benchmarked on both devices.
"""
from __future__ import annotations

import os
import platform
import sys

import numpy as np

try:  # pragma: no cover - depends on the machine
    import cupy as _cp

    try:
        _N_GPU = int(_cp.cuda.runtime.getDeviceCount())
    except Exception:  # driver missing / no device
        _N_GPU = 0
except Exception:  # ImportError, or CUDA libraries missing
    _cp = None
    _N_GPU = 0

GPU_AVAILABLE: bool = _cp is not None and _N_GPU > 0


def get_xp(device: str = "auto"):
    """``cupy`` for device='gpu' (or 'auto' when a GPU exists), else ``numpy``."""
    device = device.lower()
    if device not in {"auto", "cpu", "gpu"}:
        raise ValueError("device must be 'auto', 'cpu' or 'gpu'")
    if device == "gpu" and not GPU_AVAILABLE:
        raise RuntimeError("GPU requested but CuPy/CUDA is not available on this machine")
    if device == "gpu" or (device == "auto" and GPU_AVAILABLE):
        return _cp
    return np


def is_gpu(xp) -> bool:
    return _cp is not None and xp is _cp


def to_numpy(a):
    if _cp is not None and isinstance(a, _cp.ndarray):
        return _cp.asnumpy(a)
    return np.asarray(a)


def synchronize(xp) -> None:
    """Block until queued GPU work is done (needed for honest timings)."""
    if is_gpu(xp):
        _cp.cuda.Stream.null.synchronize()


def cpu_name() -> str:
    if sys.platform.startswith("win"):
        try:
            import winreg

            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            )
            return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        except Exception:
            pass
    if os.path.exists("/proc/cpuinfo"):
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    return platform.processor() or platform.machine()


def gpu_name():
    if not GPU_AVAILABLE:
        return None
    name = _cp.cuda.runtime.getDeviceProperties(0)["name"]
    return name.decode() if isinstance(name, bytes) else str(name)


def hardware_info() -> dict:
    info = {
        "python": sys.version.split()[0],
        "os": platform.platform(),
        "numpy": np.__version__,
        "cpu": cpu_name(),
        "gpu": gpu_name(),
    }
    if GPU_AVAILABLE:
        info["cupy"] = _cp.__version__
        try:
            v = int(_cp.cuda.runtime.runtimeGetVersion())
            info["cuda_runtime"] = f"{v // 1000}.{(v % 1000) // 10}"
        except Exception:
            pass
        try:
            mem = _cp.cuda.runtime.getDeviceProperties(0)["totalGlobalMem"]
            info["gpu_memory_gb"] = round(mem / 1024**3, 1)
        except Exception:
            pass
    return info

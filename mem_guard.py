"""Memory safety for GPU (MPS) runs. IMPORT THIS BEFORE torch.

1. Hard MPS cap: PyTorch raises an out-of-memory error instead of starving macOS.
2. Watchdog thread: kills the process if its resident memory exceeds a limit.

After the 2026-11-12 kernel panic (a training process reached 46.7 GB on a 36 GB Mac).
"""
import os
import threading
import time

import psutil

MPS_CAP_RATIO = os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.3")  # ~8 GB here
# The low watermark (when the allocator starts freeing cached memory) must be <= the cap.
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.2")
RSS_LIMIT_GB = float(os.environ.get("RSS_LIMIT_GB", "10"))


def rss_gb():
    return psutil.Process().memory_info().rss / 2**30


def _watchdog():
    while True:
        if rss_gb() > RSS_LIMIT_GB:
            print(f"\n[mem_guard] RSS {rss_gb():.1f} GB > {RSS_LIMIT_GB} GB limit: killing run",
                  flush=True)
            os._exit(99)
        time.sleep(0.5)


threading.Thread(target=_watchdog, daemon=True).start()

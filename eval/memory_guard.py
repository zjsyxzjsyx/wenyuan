"""Pause an experiment while the machine is short of memory, instead of pushing it over the edge.

On the 16 GB development laptop a long LLM experiment ran Windows out of memory several times in
a few minutes (M2): Ollama's llama-server is charged about 15 GB of commit for an 8B model that sits
entirely on the GPU (Windows counts GPU allocations against system commit), and with Docker and a
desktop alongside, the machine starts paging and stops responding. The first version of this guard
watched only the share of the commit limit in use, which stayed under its threshold the whole time:
what froze the machine was physical memory running out. So `wait_for_memory` now pauses while
either free RAM is below `min_free_gb` or commit use is above `max_commit`, printing why. The
readings come from GlobalMemoryStatusEx; elsewhere the guard is a no-op.
"""
from __future__ import annotations

import ctypes
import os
import sys
import threading
import time
from dataclasses import dataclass


class _MemoryStatus(ctypes.Structure):
    _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]


@dataclass
class Memory:
    free_gb: float          # physical RAM available
    commit_used: float      # share of the commit limit (RAM + page file) in use


def read_memory() -> Memory | None:
    if sys.platform != "win32":
        return None
    status = _MemoryStatus()
    status.dwLength = ctypes.sizeof(_MemoryStatus)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return None
    return Memory(status.ullAvailPhys / 2**30, 1 - status.ullAvailPageFile / status.ullTotalPageFile)


_lock = threading.Lock()


def wait_for_memory(min_free_gb: float | None = None, max_commit: float | None = None, poll_s: float = 15,
                    read=read_memory) -> float:
    """Block while memory is short; returns the seconds spent waiting.

    The thresholds default to 2 GB free and 85% commit; EVAL_MIN_FREE_GB and EVAL_MAX_COMMIT override
    them for a run on a machine that is known to sit lower (Docker, both models and a desktop).
    """
    if min_free_gb is None:
        min_free_gb = float(os.environ.get("EVAL_MIN_FREE_GB", 2.0))
    if max_commit is None:
        max_commit = float(os.environ.get("EVAL_MAX_COMMIT", 0.85))
    waited = 0.0
    with _lock:        # one waiting thread reports; the others queue behind it
        while (m := read()) is not None and (m.free_gb < min_free_gb or m.commit_used > max_commit):
            print(f"[memory guard] free RAM {m.free_gb:.1f} GB, commit {m.commit_used:.0%}: pausing {poll_s:.0f}s",
                  flush=True)
            time.sleep(poll_s)
            waited += poll_s
    return waited

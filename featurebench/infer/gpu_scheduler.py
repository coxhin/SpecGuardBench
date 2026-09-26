"""GPU scheduling utilities for inference."""

from __future__ import annotations

import re
import subprocess
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence


_GPU_ID_RE = re.compile(r"^\d+$")


def parse_gpu_id_list(gpu_ids: str) -> List[str]:
    """Parse a comma-separated GPU id string into a normalized list of ids."""
    if gpu_ids is None:
        raise TypeError("gpu_ids must be a string")

    parts = [p.strip() for p in gpu_ids.split(",")]
    parts = [p for p in parts if p]

    if not parts:
        raise ValueError("gpu_ids is empty")

    for p in parts:
        if not _GPU_ID_RE.match(p):
            raise ValueError(f"Invalid GPU id '{p}' in --gpu-ids (expected digits like '0,1,2')")

    seen = set()
    normalized: List[str] = []
    for p in parts:
        if p not in seen:
            seen.add(p)
            normalized.append(p)
    return normalized


def detect_host_gpu_ids() -> Optional[List[str]]:
    """Best-effort detect host GPU ids via nvidia-smi."""
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--list-gpus"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return None
    except Exception:
        return None

    if proc.returncode != 0:
        return None

    lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
    if not lines:
        return None

    return [str(i) for i in range(len(lines))]


@dataclass(frozen=True)
class GpuLease:
    """A lease returned by the scheduler."""

    gpu_ids: List[str]

    @property
    def gpu_ids_str(self) -> str:
        return ",".join(self.gpu_ids)


class GpuScheduler:
    """Thread-safe GPU scheduler using a least-loaded policy."""

    def __init__(self, gpu_pool: Sequence[str]):
        if not gpu_pool:
            raise ValueError("gpu_pool must be non-empty")
        for gid in gpu_pool:
            if not _GPU_ID_RE.match(str(gid)):
                raise ValueError(f"Invalid GPU id in pool: {gid}")

        self._gpu_pool: List[str] = [str(g) for g in gpu_pool]
        self._lock = threading.Lock()
        self._usage: Dict[str, int] = {gid: 0 for gid in self._gpu_pool}

    @property
    def gpu_pool(self) -> List[str]:
        return list(self._gpu_pool)

    def snapshot_usage(self) -> Dict[str, int]:
        with self._lock:
            return dict(self._usage)

    def allocate(self, n: int) -> GpuLease:
        if not isinstance(n, int) or n <= 0:
            n = 1

        with self._lock:
            if n > len(self._gpu_pool):
                raise ValueError(
                    f"number_once={n} exceeds GPU pool size ({len(self._gpu_pool)}): {','.join(self._gpu_pool)}"
                )

            ranked = sorted(
                self._gpu_pool,
                key=lambda gid: (self._usage.get(gid, 0), int(gid)),
            )
            chosen = ranked[:n]

            for gid in chosen:
                self._usage[gid] = self._usage.get(gid, 0) + 1

        return GpuLease(gpu_ids=chosen)

    def release(self, lease: GpuLease) -> None:
        if lease is None:
            return

        with self._lock:
            for gid in lease.gpu_ids:
                if gid not in self._usage:
                    continue
                self._usage[gid] = max(0, self._usage[gid] - 1)

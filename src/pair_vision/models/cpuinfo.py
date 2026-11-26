"""실제로 쓸 수 있는 CPU 수(실효 CPU).

RunPod 같은 컨테이너는 os.cpu_count() 로 호스트 전체(예: 96)를 보여 주지만 cgroup 이 훨씬 적게(예: 8) 준다.
그 숫자로 스레드를 만들면 CPU 한도 안에서 스레드끼리 다투며 크게 느려진다. 그래서
  실효 CPU = min(os.cpu_count(), affinity 코어 수, cgroup v2 cpu.max, cgroup v1 cfs_quota/period) (올림, 최소 1)
을 모든 스레드 수의 기준으로 쓴다.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path

CGROUP_ROOT = Path("/sys/fs/cgroup")


@dataclass(frozen=True)
class CpuInfo:
    os_count: int  # os.cpu_count()
    affinity: int | None  # sched_getaffinity 코어 수(없으면 None)
    cgroup: float | None  # cgroup 이 허용한 CPU 수(제한 없음 None)
    cgroup_raw: str  # 읽은 값 원문(결과에 적으려고)
    effective: int

    def describe(self) -> str:
        return (f"실효 CPU {self.effective} (os {self.os_count}, affinity {self.affinity if self.affinity is not None else '-'}, "
                f"cgroup {self.cgroup_raw or '제한 없음/읽지 못함'})")


def _read(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return None


def cgroup_cpus(root: Path = CGROUP_ROOT) -> tuple[float | None, str]:
    """(허용 CPU 수 또는 None, 원문). v2: cpu.max("<quota|max> <period>"), v1: cpu/cpu.cfs_quota_us ÷ cfs_period_us."""
    text = _read(root / "cpu.max")
    if text:
        parts = text.split()
        if len(parts) == 2 and parts[0] != "max":
            try:
                return float(parts[0]) / float(parts[1]), f"cpu.max={text}"
            except (ValueError, ZeroDivisionError):
                pass
        return None, f"cpu.max={text}"
    quota, period = _read(root / "cpu" / "cpu.cfs_quota_us"), _read(root / "cpu" / "cpu.cfs_period_us")
    if quota and period:
        try:
            q, p = float(quota), float(period)
            if q > 0 and p > 0:
                return q / p, f"cfs_quota={quota}/period={period}"
        except ValueError:
            pass
        return None, f"cfs_quota={quota}/period={period}"
    return None, ""


def cpu_info(root: Path = CGROUP_ROOT) -> CpuInfo:
    os_count = os.cpu_count() or 1
    affinity = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    cg, raw = cgroup_cpus(root)
    limits = [float(os_count)]
    if affinity:
        limits.append(float(affinity))
    if cg is not None:
        limits.append(cg)
    return CpuInfo(os_count, affinity, cg, raw, max(1, math.ceil(min(limits) - 1e-9)))


def effective_cpus(root: Path = CGROUP_ROOT) -> int:
    return cpu_info(root).effective


def default_threads(workers: int, root: Path = CGROUP_ROOT) -> int:
    """워커 한 개가 쓸 스레드 수 = 실효 CPU ÷ 워커 수(최소 1)."""
    return max(1, effective_cpus(root) // max(1, workers))


def resolve_threads(requested: int, workers: int, root: Path = CGROUP_ROOT) -> int:
    """지정값(0=자동)을 실제 스레드 수로."""
    return requested if requested > 0 else default_threads(workers, root)

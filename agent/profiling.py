"""
Lightweight wall-clock profiler for the agent's pipeline steps.

Usage:
    prof = StepProfiler()
    with prof.track("planning"):
        plan = await planner.create_plan(goal)   # awaits are fine inside the block
    print(prof.summary())

Spans are accumulated by label (total seconds + call count). Note that some
steps run concurrently (the eager dependency install overlaps test generation),
so the per-step totals can sum to more than the real elapsed time.
"""

import time
from contextlib import contextmanager
from typing import Dict, List


class StepProfiler:
    def __init__(self):
        self._records: Dict[str, List[float]] = {}  # label -> [total_seconds, count]
        self._order: List[str] = []

    def reset(self) -> None:
        self._records = {}
        self._order = []

    @contextmanager
    def track(self, label: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.add(label, time.perf_counter() - start)

    def add(self, label: str, seconds: float) -> None:
        if label not in self._records:
            self._records[label] = [0.0, 0]
            self._order.append(label)
        self._records[label][0] += seconds
        self._records[label][1] += 1

    def summary(self, total_wall_clock: float = None) -> str:
        if not self._records:
            return ""
        rows = sorted(self._records.items(), key=lambda kv: kv[1][0], reverse=True)
        width = max(len(label) for label in self._records)
        measured = sum(total for total, _ in self._records.values())

        lines = [
            "",
            "⏱  Step profile (wall-clock per step, slowest first;",
            "   overlapping steps may sum above the total):",
        ]
        for label, (total, count) in rows:
            share = f"{(total / measured * 100):4.0f}%" if measured else "   -"
            lines.append(f"   {label.ljust(width)}  {total:8.2f}s  x{count:<3d} {share}")
        lines.append(f"   {'-' * (width + 22)}")
        lines.append(f"   {'measured steps'.ljust(width)}  {measured:8.2f}s")
        if total_wall_clock is not None:
            lines.append(f"   {'total wall-clock'.ljust(width)}  {total_wall_clock:8.2f}s")
        return "\n".join(lines)

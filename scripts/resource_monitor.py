"""Lightweight process, system, and NVIDIA resource monitoring."""

from __future__ import annotations

import csv
import subprocess
import threading
import time
from pathlib import Path

import psutil


class ResourceMonitor:
    def __init__(self, output_path: Path, interval_seconds: float = 5.0) -> None:
        self.output_path = output_path
        self.interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.samples: list[dict[str, float | str]] = []

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="resource-monitor", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval_seconds + 2)
        self._write()

    def _run(self) -> None:
        while not self._stop.is_set():
            self.samples.append(self._sample())
            self._stop.wait(self.interval_seconds)
        self.samples.append(self._sample())

    @staticmethod
    def _gpu_sample() -> tuple[float, float, float]:
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, check=True, timeout=3,
            )
            used, total, utilization = [float(value.strip()) for value in result.stdout.splitlines()[0].split(",")]
            return used, total, utilization
        except (FileNotFoundError, IndexError, ValueError, subprocess.SubprocessError):
            return 0.0, 0.0, 0.0

    def _sample(self) -> dict[str, float | str]:
        memory = psutil.virtual_memory()
        gpu_used, gpu_total, gpu_utilization = self._gpu_sample()
        return {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "ram_used_gb": round((memory.total - memory.available) / 1024**3, 4),
            "ram_percent": round(memory.percent, 2),
            "gpu_memory_used_mb": round(gpu_used, 2),
            "gpu_memory_total_mb": round(gpu_total, 2),
            "gpu_utilization_percent": round(gpu_utilization, 2),
            "cpu_percent": round(psutil.cpu_percent(interval=None), 2),
        }

    def _write(self) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        fields = ["timestamp", "ram_used_gb", "ram_percent", "gpu_memory_used_mb", "gpu_memory_total_mb", "gpu_utilization_percent", "cpu_percent"]
        with self.output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(self.samples)

    def summary(self) -> dict[str, float | str]:
        if not self.samples:
            return {"peak_ram_gb": 0.0, "peak_ram_percent": 0.0, "peak_gpu_memory_gb": 0.0}
        peak_ram = max(float(row["ram_used_gb"]) for row in self.samples)
        return {
            "peak_ram_gb": round(peak_ram, 4),
            "peak_ram_percent": round(max(float(row["ram_percent"]) for row in self.samples), 2),
            "peak_gpu_memory_gb": round(max(float(row["gpu_memory_used_mb"]) for row in self.samples) / 1024, 4),
            "peak_gpu_utilization_percent": round(max(float(row["gpu_utilization_percent"]) for row in self.samples), 2),
            "ram_limit_gb": 14.0,
            "ram_status": "PASS" if peak_ram < 14 else "FAIL",
        }
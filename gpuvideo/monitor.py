"""Monitores leves de GPU (nvidia-smi) e CPU (/proc) para o benchmark.

GpuMonitor usa um unico processo ``nvidia-smi ... -lms`` em streaming
(baixo overhead) e amostra: util GPU, util decoder (NVDEC), util encoder
(NVENC), memoria usada e potencia.

CpuMonitor le /proc/stat (sistema) e /proc/self/stat (processo) nas bordas
da janela medida e calcula o uso percentual.
"""
from __future__ import annotations

import os
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class GpuSummary:
    samples: int = 0
    gpu_util_mean: float = 0.0
    gpu_util_max: float = 0.0
    dec_util_mean: float = 0.0
    dec_util_max: float = 0.0
    enc_util_mean: float = 0.0
    enc_util_max: float = 0.0
    mem_used_max_mb: float = 0.0
    power_w_mean: float = 0.0


class GpuMonitor:
    """Amostra a GPU via nvidia-smi em streaming."""

    FIELDS = ("utilization.gpu", "utilization.decoder",
              "utilization.encoder", "memory.used", "power.draw")

    def __init__(self, interval_ms: int = 200, gpu_index: int = 0):
        self.interval_ms = interval_ms
        self.gpu_index = gpu_index
        self._proc: Optional[subprocess.Popen] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self.samples: List[tuple] = []  # (gpu, dec, enc, mem, power)
        self.available = self._check()

    @staticmethod
    def _check() -> bool:
        from shutil import which
        return which("nvidia-smi") is not None

    def _reader(self):
        assert self._proc and self._proc.stdout
        for line in self._proc.stdout:
            if self._stop.is_set():
                break
            line = line.strip()
            if not line:
                continue
            try:
                vals = [float(x.strip()) for x in line.split(",")]
                if len(vals) >= 5:
                    self.samples.append(tuple(vals[:5]))
            except ValueError:
                continue

    def start(self) -> "GpuMonitor":
        if not self.available:
            return self
        query = ",".join(self.FIELDS)
        cmd = [
            "nvidia-smi", f"--query-gpu={query}",
            "--format=csv,noheader,nounits",
            "-i", str(self.gpu_index), "-lms", str(self.interval_ms),
        ]
        self._proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> GpuSummary:
        self._stop.set()
        if self._proc is not None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        if self._thread is not None:
            self._thread.join(timeout=2)
        return self.summary()

    def summary(self) -> GpuSummary:
        s = self.samples
        if not s:
            return GpuSummary()
        gpu = [r[0] for r in s]
        dec = [r[1] for r in s]
        enc = [r[2] for r in s]
        mem = [r[3] for r in s]
        pwr = [r[4] for r in s]
        n = len(s)
        return GpuSummary(
            samples=n,
            gpu_util_mean=sum(gpu) / n, gpu_util_max=max(gpu),
            dec_util_mean=sum(dec) / n, dec_util_max=max(dec),
            enc_util_mean=sum(enc) / n, enc_util_max=max(enc),
            mem_used_max_mb=max(mem), power_w_mean=sum(pwr) / n,
        )

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()


@dataclass
class CpuSummary:
    n_cpus: int = 0
    system_pct: float = 0.0     # uso medio do sistema (0-100 por nucleo agregado)
    process_pct: float = 0.0    # uso do processo python (pode passar de 100 com threads)


def _read_proc_stat_total() -> tuple:
    with open("/proc/stat") as f:
        parts = f.readline().split()
    vals = list(map(int, parts[1:]))
    idle = vals[3] + (vals[4] if len(vals) > 4 else 0)  # idle + iowait
    total = sum(vals)
    return total, idle


def _read_self_cpu() -> float:
    with open("/proc/self/stat") as f:
        parts = f.read().split()
    utime = int(parts[13]); stime = int(parts[14])
    return utime + stime  # em clock ticks


class CpuMonitor:
    def __init__(self):
        self.n_cpus = os.cpu_count() or 1
        self._clk = os.sysconf("SC_CLK_TCK")
        self._t0 = None

    def start(self) -> "CpuMonitor":
        self._wall0 = time.perf_counter()
        self._sys0 = _read_proc_stat_total()
        self._proc0 = _read_self_cpu()
        return self

    def stop(self) -> CpuSummary:
        wall = time.perf_counter() - self._wall0
        total1, idle1 = _read_proc_stat_total()
        total0, idle0 = self._sys0
        dtotal = total1 - total0
        didle = idle1 - idle0
        system_pct = 100.0 * (1 - didle / dtotal) if dtotal else 0.0
        proc1 = _read_self_cpu()
        proc_secs = (proc1 - self._proc0) / self._clk
        process_pct = 100.0 * proc_secs / wall if wall else 0.0
        return CpuSummary(n_cpus=self.n_cpus, system_pct=system_pct,
                          process_pct=process_pct)

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()

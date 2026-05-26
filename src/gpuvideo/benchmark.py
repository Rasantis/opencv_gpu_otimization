"""Harness de benchmark comparativo entre backends/engines.

Mede, para cada modo (opencv-cpu, opencv-gpu, gstreamer-cpu, gstreamer-gpu):
  - vazao (FPS) e speedup relativo ao baseline opencv-cpu
  - latencia de aquisicao por frame (read): media, p50, p90, p99
  - latencia de processamento por frame (op aplicada igual a todos)
  - CPU% (sistema e processo)
  - GPU%, decoder% (NVDEC), encoder% (NVENC), memoria, potencia

A "op" de processamento e' identica em todos os backends, garantindo
comparacao justa do custo de decode+entrega.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from typing import Callable, List, Optional, Sequence

import numpy as np

from . import make_stream, MODES
from .monitor import GpuMonitor, CpuMonitor, GpuSummary, CpuSummary


# --------------------------------------------------------------------------
# Operacoes de processamento (mesmo trabalho pra todos os backends)
# --------------------------------------------------------------------------
def _op_none(_arr):
    return None


def _op_light(arr):
    import cv2
    small = cv2.resize(arr, (1280, 720))
    return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)


def _op_heavy(arr):
    import cv2
    small = cv2.resize(arr, (1280, 720))
    blur = cv2.GaussianBlur(small, (7, 7), 0)
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    return cv2.Canny(gray, 60, 160)


OPS = {"none": _op_none, "light": _op_light, "heavy": _op_heavy}


# --------------------------------------------------------------------------
@dataclass
class BenchmarkResult:
    mode: str
    ok: bool = True
    error: str = ""
    frames: int = 0
    wall_s: float = 0.0
    fps: float = 0.0
    speedup: float = 1.0            # vs baseline
    acq_ms_mean: float = 0.0
    acq_ms_p50: float = 0.0
    acq_ms_p90: float = 0.0
    acq_ms_p99: float = 0.0
    proc_ms_mean: float = 0.0
    cpu_system_pct: float = 0.0
    cpu_process_pct: float = 0.0
    gpu_util_mean: float = 0.0
    gpu_util_max: float = 0.0
    dec_util_mean: float = 0.0
    dec_util_max: float = 0.0
    enc_util_mean: float = 0.0
    mem_used_max_mb: float = 0.0
    power_w_mean: float = 0.0
    width: int = 0
    height: int = 0
    pipeline: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


class Benchmark:
    def __init__(self, source, *, frames: int = 300, warmup: int = 20,
                 op: str = "light", modes: Sequence[str] = MODES,
                 gpu_index: int = 0, monitor_interval_ms: int = 150,
                 verbose: bool = True):
        self.source = source
        self.frames = frames
        self.warmup = warmup
        self.op_name = op if isinstance(op, str) else "custom"
        self.op: Callable = OPS[op] if isinstance(op, str) else op
        self.modes = list(modes)
        self.gpu_index = gpu_index
        self.monitor_interval_ms = monitor_interval_ms
        self.verbose = verbose
        self.results: List[BenchmarkResult] = []

    # ----------------------------------------------------------------
    def run_mode(self, mode: str) -> BenchmarkResult:
        res = BenchmarkResult(mode=mode)
        try:
            stream = make_stream(mode, self.source)
            stream.open()
            res.width, res.height = stream.width, stream.height
            res.pipeline = getattr(stream, "pipeline_string", "") or ""

            # ---- warmup (nao cronometrado) ----
            warmed = 0
            for _ in range(self.warmup):
                f = stream.read()
                if f is None:
                    break
                self.op(f.array)
                warmed += 1

            # ---- medicao ----
            acq: List[float] = []
            proc: List[float] = []
            gpumon = GpuMonitor(self.monitor_interval_ms, self.gpu_index).start()
            cpumon = CpuMonitor().start()
            t0 = time.perf_counter()
            count = 0
            while count < self.frames:
                a0 = time.perf_counter()
                frame = stream.read()
                a1 = time.perf_counter()
                if frame is None:
                    break
                acq.append(a1 - a0)
                p0 = time.perf_counter()
                self.op(frame.array)
                proc.append(time.perf_counter() - p0)
                count += 1
            wall = time.perf_counter() - t0
            cpu = cpumon.stop()
            gpu = gpumon.stop()
            stream.close()

            res.frames = count
            res.wall_s = wall
            res.fps = count / wall if wall else 0.0
            if acq:
                arr = np.array(acq) * 1000.0
                res.acq_ms_mean = float(arr.mean())
                res.acq_ms_p50 = float(np.percentile(arr, 50))
                res.acq_ms_p90 = float(np.percentile(arr, 90))
                res.acq_ms_p99 = float(np.percentile(arr, 99))
            if proc:
                res.proc_ms_mean = float(np.mean(proc) * 1000.0)
            res.cpu_system_pct = cpu.system_pct
            res.cpu_process_pct = cpu.process_pct
            res.gpu_util_mean = gpu.gpu_util_mean
            res.gpu_util_max = gpu.gpu_util_max
            res.dec_util_mean = gpu.dec_util_mean
            res.dec_util_max = gpu.dec_util_max
            res.enc_util_mean = gpu.enc_util_mean
            res.mem_used_max_mb = gpu.mem_used_max_mb
            res.power_w_mean = gpu.power_w_mean
            if count == 0:
                res.ok = False
                res.error = "0 frames lidos"
        except Exception as e:  # noqa: BLE001
            res.ok = False
            res.error = f"{type(e).__name__}: {e}"
        return res

    # ----------------------------------------------------------------
    def compare(self, modes: Optional[Sequence[str]] = None) -> "Benchmark":
        modes = list(modes) if modes else self.modes
        self.results = []
        for mode in modes:
            if self.verbose:
                print(f"  -> rodando {mode} ...", flush=True)
            r = self.run_mode(mode)
            self.results.append(r)
            if self.verbose:
                if r.ok:
                    print(f"     {r.frames} frames | {r.fps:7.1f} fps | "
                          f"acq p50 {r.acq_ms_p50:.2f} ms | dec {r.dec_util_mean:.0f}% | "
                          f"cpu sys {r.cpu_system_pct:.0f}%", flush=True)
                else:
                    print(f"     FALHOU: {r.error}", flush=True)
        self._fill_speedup()
        return self

    def _fill_speedup(self):
        base = next((r for r in self.results
                     if r.mode == "opencv-cpu" and r.ok and r.fps > 0), None)
        if base is None:
            base = next((r for r in self.results if r.ok and r.fps > 0), None)
        if base is None:
            return
        for r in self.results:
            r.speedup = (r.fps / base.fps) if (r.ok and base.fps) else 0.0

    # ----------------------------------------------------------------
    def report(self) -> str:
        lines = []
        w, h = next(((r.width, r.height) for r in self.results if r.width), (0, 0))
        lines.append("")
        lines.append("=" * 104)
        lines.append(f" COMPARATIVO DE PERFORMANCE - captura/decode {w}x{h} | "
                     f"op={self.op_name} | alvo={self.frames} frames")
        lines.append("=" * 104)
        header = (f"{'modo':<16}{'fps':>9}{'speedup':>9}{'acq p50':>10}"
                  f"{'acq p99':>10}{'proc ms':>9}{'cpu sys%':>9}{'cpu prc%':>9}"
                  f"{'gpu%':>7}{'dec%':>7}{'enc%':>6}")
        lines.append(header)
        lines.append("-" * 104)
        for r in self.results:
            if not r.ok:
                lines.append(f"{r.mode:<16}  FALHOU: {r.error}")
                continue
            lines.append(
                f"{r.mode:<16}{r.fps:>9.1f}{r.speedup:>8.2f}x"
                f"{r.acq_ms_p50:>9.2f} {r.acq_ms_p99:>9.2f} {r.proc_ms_mean:>8.2f}"
                f"{r.cpu_system_pct:>9.0f}{r.cpu_process_pct:>9.0f}"
                f"{r.gpu_util_mean:>7.0f}{r.dec_util_mean:>7.0f}{r.enc_util_mean:>6.0f}"
            )
        lines.append("-" * 104)
        lines.append("acq = latencia de aquisicao por frame (decode+entrega) | "
                     "dec/enc = uso NVDEC/NVENC | speedup vs opencv-cpu")
        lines.append("=" * 104)
        out = "\n".join(lines)
        print(out)
        return out

    def to_json(self, path: str) -> None:
        payload = {
            "source": str(self.source),
            "op": self.op_name,
            "frames_target": self.frames,
            "results": [r.to_dict() for r in self.results],
        }
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)

    def to_csv(self, path: str) -> None:
        import csv
        if not self.results:
            return
        fields = list(self.results[0].to_dict().keys())
        with open(path, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=fields)
            wr.writeheader()
            for r in self.results:
                wr.writerow(r.to_dict())

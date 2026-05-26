#!/usr/bin/env python3
"""Teste de escala serio: aggregate FPS + CPU% + NVDEC% por N streams."""
import sys
from gpuvideo.multistream import MultiStream
from gpuvideo.monitor import GpuMonitor, CpuMonitor

VIDEO = sys.argv[1] if len(sys.argv) > 1 else "heavy_1080p.mp4"
NS = [int(x) for x in (sys.argv[2].split(",") if len(sys.argv) > 2 else ["4", "8", "16", "24"])]
MODES = ["opencv-cpu", "gstreamer-gpu"]

print(f"Fonte: {VIDEO} | streams: {NS}\n")
print(f"{'modo':<15}{'N':>4}{'agg fps':>10}{'fps/stream':>12}{'cpu sys%':>10}{'gpu%':>7}{'dec%':>7}")
print("-" * 65)
for n in NS:
    for mode in MODES:
        gmon = GpuMonitor(150).start()
        cmon = CpuMonitor().start()
        res = MultiStream.replicate(VIDEO, n, mode=mode).run(max_frames=120, warmup=5)
        cpu = cmon.stop()
        gpu = gmon.stop()
        err = "" if not res.errors else f"  ERRO:{len(res.errors)}"
        print(f"{mode:<15}{n:>4}{res.aggregate_fps:>10.0f}"
              f"{res.aggregate_fps/n:>12.1f}{cpu.system_pct:>10.0f}"
              f"{gpu.gpu_util_mean:>7.0f}{gpu.dec_util_mean:>7.0f}{err}")
    print("-" * 65)

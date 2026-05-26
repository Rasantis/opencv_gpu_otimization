#!/usr/bin/env python3
"""Teste de velocidade: gpuvideo (decode) + YOLO11 (Ultralytics) na GPU.

Mede, por backend de decode, o FPS de decode, de inferência e end-to-end, mais
o uso de CPU/GPU/NVDEC — mostrando o efeito de decodificar na GPU vs CPU quando
a inferência YOLO já está na GPU.

    # rode com o venv que tem torch+ultralytics:
    .venv-yolo/bin/python benchmarks/yolo_speedtest.py VIDEO.mp4
    .venv-yolo/bin/python benchmarks/yolo_speedtest.py VIDEO.mp4 --model yolo11s.pt --imgsz 640
    .venv-yolo/bin/python benchmarks/yolo_speedtest.py VIDEO.mp4 --record yolo_out.mp4

Doc YOLO11: https://docs.ultralytics.com/models/yolo11/
"""
from __future__ import annotations
import os, sys, time, argparse

try:
    import gpuvideo  # noqa: F401
except ModuleNotFoundError:
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import numpy as np
from gpuvideo import make_stream, ALL_MODES, MODES
from gpuvideo.monitor import GpuMonitor, CpuMonitor
from gpuvideo.cudacodec import cudacodec_available


def run_mode(mode, video, model, device, imgsz, frames, warmup):
    import time as _t
    stream = make_stream(mode, video)
    stream.open()
    # warmup (carrega/JIT do modelo + enche pipeline de decode)
    n = 0
    while n < warmup:
        f = stream.read()
        if f is None:
            stream.close(); stream = make_stream(mode, video); stream.open(); continue
        model.predict(f.array, device=device, imgsz=imgsz, verbose=False)
        n += 1

    dec_ms, inf_ms = [], []
    gmon = GpuMonitor(150).start(); cmon = CpuMonitor().start()
    t0 = _t.perf_counter(); count = 0; ndet = 0
    while count < frames:
        a = _t.perf_counter()
        f = stream.read()
        b = _t.perf_counter()
        if f is None:
            stream.close(); stream = make_stream(mode, video); stream.open(); continue
        r = model.predict(f.array, device=device, imgsz=imgsz, verbose=False)
        c = _t.perf_counter()
        dec_ms.append((b - a) * 1000); inf_ms.append((c - b) * 1000)
        ndet += int(r[0].boxes.shape[0]) if r and r[0].boxes is not None else 0
        count += 1
    wall = _t.perf_counter() - t0
    cpu = cmon.stop(); gpu = gmon.stop()
    stream.close()
    return {
        "mode": mode, "frames": count, "e2e_fps": count / wall if wall else 0,
        "dec_ms": float(np.mean(dec_ms)) if dec_ms else 0,
        "inf_ms": float(np.mean(inf_ms)) if inf_ms else 0,
        "inf_fps": 1000.0 / np.mean(inf_ms) if inf_ms else 0,
        "cpu_sys": cpu.system_pct, "gpu": gpu.gpu_util_mean, "dec": gpu.dec_util_mean,
        "det_per_frame": ndet / count if count else 0,
    }


def record_annotated(mode, video, model, device, imgsz, out, seconds):
    import cv2, time as _t
    stream = make_stream(mode, video); stream.open()
    writer = None; t0 = _t.perf_counter(); times = []
    while _t.perf_counter() - t0 < seconds:
        f = stream.read()
        if f is None: break
        r = model.predict(f.array, device=device, imgsz=imgsz, verbose=False)
        annotated = r[0].plot()                      # desenha as caixas
        h, w = annotated.shape[:2]; s = min(1280 / w, 720 / h, 1.0)
        if s < 1.0: annotated = cv2.resize(annotated, (int(w * s), int(h * s)))
        times.append(_t.perf_counter()); times = times[-30:]
        fps = (len(times) - 1) / (times[-1] - times[0]) if len(times) > 1 else 0
        cv2.putText(annotated, f"YOLO11 + {mode}: {fps:.1f} FPS", (16, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(annotated, f"YOLO11 + {mode}: {fps:.1f} FPS", (16, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (80, 255, 120), 2, cv2.LINE_AA)
        if writer is None:
            writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), 25, (annotated.shape[1], annotated.shape[0]))
        writer.write(annotated)
    stream.close()
    if writer: writer.release(); print("gravado em", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--model", default="yolo11n.pt")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--device", default="0")
    ap.add_argument("--modes", default=None, help="csv de backends de decode")
    ap.add_argument("--record", default=None, help="grava MP4 anotado (1 backend)")
    a = ap.parse_args()

    from ultralytics import YOLO
    import torch
    dev = a.device if torch.cuda.is_available() else "cpu"
    print(f"torch {torch.__version__} | CUDA {torch.cuda.is_available()} "
          f"({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Modelo: {a.model} | imgsz {a.imgsz} | device {dev} | vídeo {a.video}")
    model = YOLO(a.model)

    if a.record:
        mode = (a.modes.split(",")[0] if a.modes else
                ("opencv-cuda" if cudacodec_available() else "gstreamer-gpu"))
        record_annotated(mode, a.video, model, dev, a.imgsz, a.record, seconds=8)
        return

    avail = list(ALL_MODES) if cudacodec_available() else list(MODES)
    modes = a.modes.split(",") if a.modes else [m for m in
            ("opencv-cuda", "gstreamer-gpu", "opencv-cpu") if m in avail]

    print(f"\n{'backend':<15}{'e2e fps':>9}{'decode ms':>11}{'infer ms':>10}"
          f"{'infer fps':>11}{'cpu%':>7}{'gpu%':>7}{'dec%':>7}{'det/frm':>9}")
    print("-" * 95)
    for mode in modes:
        try:
            r = run_mode(mode, a.video, model, dev, a.imgsz, a.frames, a.warmup)
            print(f"{r['mode']:<15}{r['e2e_fps']:>9.1f}{r['dec_ms']:>11.2f}{r['inf_ms']:>10.2f}"
                  f"{r['inf_fps']:>11.1f}{r['cpu_sys']:>7.0f}{r['gpu']:>7.0f}{r['dec']:>7.0f}"
                  f"{r['det_per_frame']:>9.1f}")
        except Exception as e:
            print(f"{mode:<15}  FALHOU: {type(e).__name__}: {e}")
    print("-" * 95)
    print("e2e = decode+inferência | infer fps = só a inferência YOLO | dec% = uso NVDEC")


if __name__ == "__main__":
    main()

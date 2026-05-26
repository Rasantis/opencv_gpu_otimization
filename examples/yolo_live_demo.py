#!/usr/bin/env python3
"""Demo AO VIVO: gpuvideo (decode) + YOLO11 (detecção) com troca de backend.

Janela com o vídeo + caixas do YOLO11 e um HUD (FPS, ms de inferência, nº de
detecções, NVDEC/GPU/CPU). Troque o backend de decode em tempo real e veja o
FPS/CPU mudarem — com a inferência rodando na GPU.

    .venv-yolo/bin/python examples/yolo_live_demo.py video.mp4
    .venv-yolo/bin/python examples/yolo_live_demo.py video.mp4 --record saida.mp4

Teclas:
    1-5  troca o backend de decode      y  liga/desliga o YOLO
    m    alterna modelo (n <-> s)        espaço pausa     q / ESC  sai
"""
from __future__ import annotations
import os, sys, time, argparse, glob

try:
    import gpuvideo  # noqa: F401
except ModuleNotFoundError:
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import cv2
import numpy as np
from gpuvideo import make_stream, ALL_MODES, MODES
from gpuvideo.monitor import GpuMonitor
from gpuvideo.cudacodec import cudacodec_available

KEY_TO_MODE = {ord("1"): "opencv-cpu", ord("2"): "opencv-gpu", ord("3"): "gstreamer-cpu",
               ord("4"): "gstreamer-gpu", ord("5"): "opencv-cuda"}
MODE_COLOR = {"opencv-cpu": (120, 120, 240), "opencv-gpu": (240, 180, 80),
              "gstreamer-cpu": (120, 200, 240), "gstreamer-gpu": (240, 120, 200),
              "opencv-cuda": (120, 240, 120)}


def shadow(img, txt, org, scale, color, thick=2):
    cv2.putText(img, txt, (org[0] + 2, org[1] + 2), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def bar(img, x, y, w, h, frac, color, label):
    frac = max(0.0, min(1.0, frac / 100.0))
    cv2.rectangle(img, (x, y), (x + w, y + h), (60, 60, 60), -1)
    cv2.rectangle(img, (x, y), (x + int(w * frac), y + h), color, -1)
    shadow(img, label, (x + 6, y + h - 6), 0.45, (255, 255, 255), 1)


def panel(img, x, y, w, h, alpha=0.55):
    sub = img[y:y + h, x:x + w].copy()
    cv2.rectangle(sub, (0, 0), (w, h), (20, 20, 20), -1)
    img[y:y + h, x:x + w] = cv2.addWeighted(sub, alpha, img[y:y + h, x:x + w], 1 - alpha, 0)


class CpuLive:
    def __init__(self): self.p = self._r(); self.t = time.perf_counter(); self.v = 0.0
    @staticmethod
    def _r():
        with open("/proc/stat") as f: x = list(map(int, f.readline().split()[1:]))
        return sum(x), x[3] + (x[4] if len(x) > 4 else 0)
    def __call__(self):
        if time.perf_counter() - self.t < 0.25: return self.v
        t, i = self._r(); pt, pi = self.p; d = t - pt
        self.v = 100.0 * (1 - (i - pi) / d) if d else self.v
        self.p = (t, i); self.t = time.perf_counter(); return self.v


def fit(img, mw=1280, mh=720):
    h, w = img.shape[:2]; s = min(mw / w, mh / h, 1.0)
    return cv2.resize(img, (int(w * s), int(h * s))) if s < 1.0 else img


def draw_hud(img, mode, model_name, fps, infms, ndet, yolo_on, gpu, dec, cpu, paused):
    h, w = img.shape[:2]
    panel(img, 10, 10, 450, 212)
    shadow(img, f"{fps:5.1f} FPS", (24, 70), 1.7, (80, 255, 120), 3)
    shadow(img, mode, (24, 104), 0.85, MODE_COLOR.get(mode, (255, 255, 255)), 2)
    yl = f"YOLO {model_name}  {infms:.1f} ms  {ndet} obj" if yolo_on else "YOLO OFF (só decode)"
    shadow(img, yl + ("  PAUSADO" if paused else ""), (24, 130), 0.55, (60, 220, 255), 1)
    bar(img, 24, 144, 420, 18, dec, (80, 200, 255), f"NVDEC {dec:3.0f}%")
    bar(img, 24, 166, 420, 18, gpu, (120, 240, 120), f"GPU   {gpu:3.0f}%")
    bar(img, 24, 188, 420, 18, cpu, (120, 120, 255), f"CPU   {cpu:3.0f}%")
    panel(img, 10, h - 36, w - 20, 26)
    shadow(img, "1-5 backend | y liga/desliga YOLO | m modelo n/s | espaco pausa | q sair",
           (24, h - 17), 0.5, (220, 220, 220), 1)


def pick_video(arg):
    if arg and os.path.exists(arg): return arg
    for c in glob.glob("*.mp4") + glob.glob("*.mkv") + glob.glob("*.webm"): return c
    from gpuvideo.make_test_video import make_test_video
    return make_test_video("demo_1080p.mp4", resolution="1080p", frames=300, pattern="ball")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", nargs="?", default=None)
    ap.add_argument("--model", default="yolo11n.pt")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--mode", default=None)
    ap.add_argument("--record", default=None)
    ap.add_argument("--seconds", type=float, default=10.0)
    a = ap.parse_args()

    from ultralytics import YOLO
    import torch
    dev = 0 if torch.cuda.is_available() else "cpu"
    print(f"torch {torch.__version__} | CUDA {torch.cuda.is_available()} | device {dev}")

    models = {}
    def get_model(name):
        if name not in models:
            print(f"carregando {name}...")
            models[name] = YOLO(name)
        return models[name]

    model_name = a.model
    model = get_model(model_name)
    video = pick_video(a.video)
    mode = a.mode or ("opencv-cuda" if cudacodec_available() else "gstreamer-gpu")
    print(f"Vídeo: {video} | backend: {mode}")

    gpu = GpuMonitor(150).start(); cpu = CpuLive()
    stream = make_stream(mode, video); stream.open()
    use_window = not a.record and (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    if use_window:
        cv2.namedWindow("gpuvideo + YOLO11", cv2.WINDOW_AUTOSIZE)
    writer = None; t_start = time.perf_counter()

    times = []; yolo_on = True; paused = False; infms = 0.0; ndet = 0; msg = ""
    try:
        while True:
            if not paused:
                f = stream.read()
                if f is None:
                    stream.close(); stream = make_stream(mode, video); stream.open(); continue
                arr = f.array
                if arr.ndim == 2: arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
                disp = fit(arr)
                if yolo_on:
                    t0 = time.perf_counter()
                    res = model.predict(disp, device=dev, imgsz=a.imgsz, verbose=False)
                    infms = (time.perf_counter() - t0) * 1000
                    ndet = int(res[0].boxes.shape[0]) if res[0].boxes is not None else 0
                    disp = res[0].plot()
                now = time.perf_counter(); times.append(now); times = times[-30:]
                fps = (len(times) - 1) / (times[-1] - times[0]) if len(times) > 1 else 0.0

            g = gpu.samples[-1] if gpu.samples else (0, 0, 0, 0, 0)
            if len(gpu.samples) > 200: del gpu.samples[:-100]
            draw_hud(disp, mode, model_name, fps, infms, ndet, yolo_on, g[0], g[1], cpu(), paused)
            if msg: shadow(disp, msg, (24, 246), 0.6, (60, 220, 255), 2)

            if use_window:
                cv2.imshow("gpuvideo + YOLO11", disp)
                k = cv2.waitKey(1) & 0xFF
            else:
                if writer is None:
                    h, w = disp.shape[:2]
                    writer = cv2.VideoWriter(a.record, cv2.VideoWriter_fourcc(*"mp4v"), 25, (w, h))
                writer.write(disp); k = 255
                if time.perf_counter() - t_start >= a.seconds: break

            if k in (ord("q"), 27): break
            elif k == ord(" "): paused = not paused
            elif k == ord("y"): yolo_on = not yolo_on; times = []
            elif k == ord("m"):
                model_name = "yolo11s.pt" if model_name == "yolo11n.pt" else "yolo11n.pt"
                model = get_model(model_name); times = []; msg = ""
            elif k in KEY_TO_MODE:
                nm = KEY_TO_MODE[k]
                if nm == "opencv-cuda" and not cudacodec_available():
                    msg = "opencv-cuda indisponivel"
                else:
                    try:
                        stream.close(); stream = make_stream(nm, video); stream.open()
                        mode = nm; times = []; msg = ""
                    except Exception as e:
                        msg = f"{nm} falhou: {type(e).__name__}"
    finally:
        gpu.stop()
        try: stream.close()
        except Exception: pass
        if writer is not None: writer.release(); print("gravado em", a.record)
        if use_window: cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

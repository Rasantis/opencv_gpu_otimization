#!/usr/bin/env python3
"""Demo VISUAL ao vivo do gpuvideo.

Mostra o vídeo decodificado numa janela com um HUD: FPS ao vivo, backend atual,
resolução, e barras de uso de NVDEC / GPU / CPU. Dá pra **trocar de backend em
tempo real** e ver o FPS e o uso de CPU mudarem na hora.

    python3 examples/visual_demo.py [video.mp4]
    python3 examples/visual_demo.py video.mp4 --record saida.mp4   # grava em vez de janela

Teclas:
    1 opencv-cpu   2 opencv-gpu   3 gstreamer-cpu   4 gstreamer-gpu   5 opencv-cuda
    e  alterna bordas (Canny)      g  alterna cinza
    espaço  pausa                  q / ESC  sai
"""
from __future__ import annotations
import os, sys, time, argparse, glob

# Permite rodar do repo sem instalar (acha o pacote em ../src).
try:
    import gpuvideo  # noqa: F401
except ModuleNotFoundError:
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import cv2
import numpy as np
from gpuvideo import make_stream, ALL_MODES, MODES
from gpuvideo.monitor import GpuMonitor
from gpuvideo.cudacodec import cudacodec_available

KEY_TO_MODE = {ord("1"): "opencv-cpu", ord("2"): "opencv-gpu",
               ord("3"): "gstreamer-cpu", ord("4"): "gstreamer-gpu",
               ord("5"): "opencv-cuda"}
MODE_COLOR = {"opencv-cpu": (120, 120, 240), "opencv-gpu": (240, 180, 80),
              "gstreamer-cpu": (120, 200, 240), "gstreamer-gpu": (240, 120, 200),
              "opencv-cuda": (120, 240, 120)}


# ---------------------------- monitor de CPU ao vivo ----------------------------
class CpuLive:
    def __init__(self):
        self.prev = self._read(); self.t = time.perf_counter(); self.pct = 0.0

    @staticmethod
    def _read():
        with open("/proc/stat") as f:
            v = list(map(int, f.readline().split()[1:]))
        idle = v[3] + (v[4] if len(v) > 4 else 0)
        return sum(v), idle

    def update(self):
        if time.perf_counter() - self.t < 0.25:
            return self.pct
        tot, idle = self._read(); ptot, pidle = self.prev
        dt = tot - ptot
        self.pct = 100.0 * (1 - (idle - pidle) / dt) if dt else self.pct
        self.prev = (tot, idle); self.t = time.perf_counter()
        return self.pct


# ---------------------------- desenho do HUD ----------------------------
def shadow_text(img, txt, org, scale, color, thick=2):
    cv2.putText(img, txt, (org[0] + 2, org[1] + 2), cv2.FONT_HERSHEY_SIMPLEX,
                scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def bar(img, x, y, w, h, frac, color, label):
    frac = max(0.0, min(1.0, frac / 100.0))
    cv2.rectangle(img, (x, y), (x + w, y + h), (60, 60, 60), -1)
    cv2.rectangle(img, (x, y), (x + int(w * frac), y + h), color, -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), (200, 200, 200), 1)
    shadow_text(img, label, (x + 6, y + h - 6), 0.5, (255, 255, 255), 1)


def panel(img, x, y, w, h, alpha=0.55):
    sub = img[y:y + h, x:x + w].copy()
    cv2.rectangle(sub, (0, 0), (w, h), (20, 20, 20), -1)
    img[y:y + h, x:x + w] = cv2.addWeighted(sub, alpha, img[y:y + h, x:x + w], 1 - alpha, 0)


def draw_hud(img, mode, fps, idx, total, dims, gpu, dec, enc, cpu, fx, paused):
    h, w = img.shape[:2]
    panel(img, 10, 10, 430, 200)
    col = MODE_COLOR.get(mode, (255, 255, 255))
    shadow_text(img, f"{fps:5.1f} FPS", (24, 70), 1.7, (80, 255, 120), 3)
    shadow_text(img, mode, (24, 105), 0.9, col, 2)
    extra = []
    if fx == "edges": extra.append("BORDAS")
    if fx == "gray": extra.append("CINZA")
    if paused: extra.append("PAUSADO")
    fr = f"frame {idx}" + (f"/{total}" if total > 0 else "")
    shadow_text(img, f"{dims[0]}x{dims[1]}  {fr}  " + " ".join(extra),
                (24, 132), 0.55, (230, 230, 230), 1)
    bar(img, 24, 146, 400, 18, dec, (80, 200, 255), f"NVDEC  {dec:3.0f}%")
    bar(img, 24, 168, 400, 18, gpu, (120, 240, 120), f"GPU    {gpu:3.0f}%")
    bar(img, 24, 190, 400, 18, cpu, (120, 120, 255), f"CPU    {cpu:3.0f}%")
    # rodapé com teclas
    panel(img, 10, h - 38, w - 20, 28)
    shadow_text(img, "1-5 backend  |  e bordas  g cinza  |  espaco pausa  |  q sair",
                (24, h - 18), 0.55, (220, 220, 220), 1)


# ---------------------------- util ----------------------------
def fit(img, max_w=1280, max_h=720):
    h, w = img.shape[:2]
    s = min(max_w / w, max_h / h, 1.0)
    return cv2.resize(img, (int(w * s), int(h * s))) if s < 1.0 else img


def apply_fx(arr, fx):
    if fx == "edges":
        g = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(cv2.Canny(g, 80, 160), cv2.COLOR_GRAY2BGR)
    if fx == "gray":
        return cv2.cvtColor(cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    return arr


def pick_video(arg):
    if arg and os.path.exists(arg):
        return arg
    for c in (glob.glob("*.mp4") + glob.glob("*.mkv") + glob.glob("*.webm")):
        return c
    print("Nenhum vídeo encontrado; gerando um de teste (NVENC)...")
    from gpuvideo.make_test_video import make_test_video
    return make_test_video("demo_1080p.mp4", resolution="1080p", frames=300, pattern="ball")


def open_mode(mode, video):
    s = make_stream(mode, video); s.open(); return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", nargs="?", default=None)
    ap.add_argument("--mode", default=None, help="backend inicial")
    ap.add_argument("--record", default=None, help="grava MP4 anotado em vez de abrir janela")
    a = ap.parse_args()

    video = pick_video(a.video)
    start = a.mode or ("opencv-cuda" if cudacodec_available() else "gstreamer-gpu")
    print(f"Vídeo: {video} | backend inicial: {start}")
    print("Backends:", ", ".join(ALL_MODES if cudacodec_available() else MODES))

    gmon = GpuMonitor(150).start()
    cpu = CpuLive()
    stream = open_mode(start, video)
    mode = start
    dims = (stream.width or 0, stream.height or 0)

    use_window = not a.record and (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    writer = None
    if use_window:
        cv2.namedWindow("gpuvideo — demo", cv2.WINDOW_AUTOSIZE)

    times = []  # timestamps p/ FPS
    idx = 0
    fx = "none"
    paused = False
    msg = ""
    try:
        while True:
            if not paused:
                frame = stream.read()
                if frame is None:
                    if use_window:               # janela: re-inicia (loop infinito)
                        stream.close(); stream = open_mode(mode, video); idx = 0
                        continue
                    break                        # record: para no fim do vídeo
                arr = frame.array
                if arr.ndim == 2:
                    arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
                dims = (frame.width, frame.height)
                idx += 1
                now = time.perf_counter(); times.append(now)
                times = times[-30:]
                fps = (len(times) - 1) / (times[-1] - times[0]) if len(times) > 1 else 0.0
                disp = fit(apply_fx(arr, fx))

            g = gmon.samples[-1] if gmon.samples else (0, 0, 0, 0, 0)
            if len(gmon.samples) > 200:
                del gmon.samples[:-100]
            draw_hud(disp, mode, fps, idx, stream.frame_count if stream.frame_count > 0 else 0,
                     dims, g[0], g[1], g[2], cpu.update(), fx, paused)
            if msg:
                shadow_text(disp, msg, (24, 232), 0.7, (60, 220, 255), 2)

            if use_window:
                cv2.imshow("gpuvideo — demo", disp)
                k = cv2.waitKey(1) & 0xFF
            else:
                if writer is None:
                    h, w = disp.shape[:2]
                    writer = cv2.VideoWriter(a.record, cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h))
                writer.write(disp); k = 255
                if idx >= (stream.frame_count if stream.frame_count > 0 else 300):
                    break

            if k in (ord("q"), 27):
                break
            elif k == ord(" "):
                paused = not paused
            elif k == ord("e"):
                fx = "none" if fx == "edges" else "edges"
            elif k == ord("g"):
                fx = "none" if fx == "gray" else "gray"
            elif k in KEY_TO_MODE:
                newm = KEY_TO_MODE[k]
                if newm == "opencv-cuda" and not cudacodec_available():
                    msg = "opencv-cuda indisponível (precisa de OpenCV-CUDA)"
                else:
                    try:
                        stream.close(); stream = open_mode(newm, video)
                        mode = newm; times = []; idx = 0; msg = ""
                    except Exception as e:
                        msg = f"{newm} falhou: {type(e).__name__}"
    finally:
        gmon.stop()
        try: stream.close()
        except Exception: pass
        if writer is not None:
            writer.release(); print("gravado em", a.record)
        if use_window:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Mosaico ao vivo: vários backends decodificando o MESMO vídeo lado a lado.

Cada tile roda um backend em sua própria thread, com FPS ao vivo — dá pra ver,
em paralelo, o opencv-cuda voando enquanto o opencv-cpu/gstreamer-cpu vão mais
devagar. Uma faixa no topo mostra o uso global de NVDEC / GPU / CPU.

    python3 examples/mosaic_demo.py [video.mp4]
    python3 examples/mosaic_demo.py video.mp4 --modes opencv-cpu,gstreamer-gpu,opencv-cuda
    python3 examples/mosaic_demo.py video.mp4 --record mosaico.mp4 --seconds 8

Teclas: q / ESC sai.
"""
from __future__ import annotations
import os, sys, time, math, argparse, glob, threading
from collections import deque

try:
    import gpuvideo  # noqa: F401
except ModuleNotFoundError:
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import cv2
import numpy as np
from gpuvideo import make_stream, ALL_MODES, MODES
from gpuvideo.monitor import GpuMonitor
from gpuvideo.cudacodec import cudacodec_available

TILE_W, TILE_H = 640, 360
MODE_COLOR = {"opencv-cpu": (120, 120, 240), "opencv-gpu": (240, 180, 80),
              "gstreamer-cpu": (120, 200, 240), "gstreamer-gpu": (240, 120, 200),
              "opencv-cuda": (120, 240, 120)}


def shadow(img, txt, org, scale, color, thick=2):
    cv2.putText(img, txt, (org[0] + 2, org[1] + 2), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def bar(img, x, y, w, h, frac, color):
    frac = max(0.0, min(1.0, frac / 100.0))
    cv2.rectangle(img, (x, y), (x + w, y + h), (60, 60, 60), -1)
    cv2.rectangle(img, (x, y), (x + int(w * frac), y + h), color, -1)


class Worker(threading.Thread):
    """Decodifica um backend em loop, guardando o último tile + FPS."""
    def __init__(self, mode, video):
        super().__init__(daemon=True)
        self.mode, self.video = mode, video
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.tile = np.zeros((TILE_H, TILE_W, 3), np.uint8)
        self.fps = 0.0
        self.count = 0
        self.error = None

    def run(self):
        try:
            stream = make_stream(self.mode, self.video)
            stream.open()
        except Exception as e:
            self.error = f"{type(e).__name__}"
            return
        times = deque(maxlen=30)
        while not self.stop.is_set():
            try:
                f = stream.read()
            except Exception as e:
                self.error = type(e).__name__; break
            if f is None:
                stream.close()
                try: stream = make_stream(self.mode, self.video); stream.open()
                except Exception as e: self.error = type(e).__name__; break
                continue
            arr = f.array
            if arr.ndim == 2:
                arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
            small = cv2.resize(arr, (TILE_W, TILE_H))
            times.append(time.perf_counter())
            fps = (len(times) - 1) / (times[-1] - times[0]) if len(times) > 1 else 0.0
            with self.lock:
                self.tile = small; self.fps = fps; self.count += 1
        try: stream.close()
        except Exception: pass

    def snapshot(self):
        with self.lock:
            return self.tile.copy(), self.fps, self.count


def render_tile(w):
    tile, fps, count = w.snapshot()
    img = tile.copy()
    col = MODE_COLOR.get(w.mode, (255, 255, 255))
    # faixa escura no topo do tile
    sub = img[0:64, 0:TILE_W].copy()
    cv2.rectangle(sub, (0, 0), (TILE_W, 64), (20, 20, 20), -1)
    img[0:64, 0:TILE_W] = cv2.addWeighted(sub, 0.55, img[0:64, 0:TILE_W], 0.45, 0)
    if w.error:
        shadow(img, f"{w.mode}: {w.error}", (12, 38), 0.7, (60, 60, 255), 2)
        return img
    shadow(img, f"{fps:5.1f} FPS", (12, 44), 1.2, (80, 255, 120), 3)
    shadow(img, w.mode, (210, 30), 0.7, col, 2)
    shadow(img, f"frame {count}", (210, 54), 0.5, (220, 220, 220), 1)
    cv2.rectangle(img, (0, 0), (TILE_W - 1, TILE_H - 1), col, 2)
    return img


def compose(workers, gpu, cpu):
    n = len(workers)
    cols = math.ceil(math.sqrt(n)); rows = math.ceil(n / cols)
    top = 54
    canvas = np.zeros((top + rows * TILE_H, cols * TILE_W, 3), np.uint8)
    shadow(canvas, "gpuvideo - mosaico de backends (mesmo video, lado a lado)", (12, 36), 0.7, (255, 255, 255), 2)
    g = gpu.samples[-1] if gpu.samples else (0, 0, 0, 0, 0)
    x = cols * TILE_W - 520
    shadow(canvas, "NVDEC", (x, 24), 0.5, (80, 200, 255), 1); bar(canvas, x + 60, 12, 100, 14, g[1], (80, 200, 255))
    shadow(canvas, "GPU", (x + 180, 24), 0.5, (120, 240, 120), 1); bar(canvas, x + 230, 12, 100, 14, g[0], (120, 240, 120))
    shadow(canvas, "CPU", (x + 350, 24), 0.5, (120, 120, 255), 1); bar(canvas, x + 400, 12, 100, 14, cpu(), (120, 120, 255))
    for i, w in enumerate(workers):
        r, c = divmod(i, cols)
        y0, x0 = top + r * TILE_H, c * TILE_W
        canvas[y0:y0 + TILE_H, x0:x0 + TILE_W] = render_tile(w)
    return canvas


class CpuLive:
    def __init__(self): self.prev = self._r(); self.t = time.perf_counter(); self.pct = 0.0
    @staticmethod
    def _r():
        with open("/proc/stat") as f: v = list(map(int, f.readline().split()[1:]))
        return sum(v), v[3] + (v[4] if len(v) > 4 else 0)
    def __call__(self):
        if time.perf_counter() - self.t < 0.25: return self.pct
        tot, idle = self._r(); pt, pi = self.prev; d = tot - pt
        self.pct = 100.0 * (1 - (idle - pi) / d) if d else self.pct
        self.prev = (tot, idle); self.t = time.perf_counter(); return self.pct


def pick_video(arg):
    if arg and os.path.exists(arg): return arg
    for c in glob.glob("*.mp4") + glob.glob("*.mkv") + glob.glob("*.webm"): return c
    from gpuvideo.make_test_video import make_test_video
    print("gerando vídeo de teste..."); return make_test_video("demo_1080p.mp4", resolution="1080p", frames=300, pattern="ball")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", nargs="?", default=None)
    ap.add_argument("--modes", default=None, help="csv; padrão = todos disponíveis")
    ap.add_argument("--record", default=None)
    ap.add_argument("--seconds", type=float, default=8.0, help="duração no modo --record")
    a = ap.parse_args()

    video = pick_video(a.video)
    avail = list(ALL_MODES) if cudacodec_available() else list(MODES)
    modes = a.modes.split(",") if a.modes else avail
    print(f"Vídeo: {video} | tiles: {modes}")

    gpu = GpuMonitor(150).start(); cpu = CpuLive()
    workers = [Worker(m, video) for m in modes]
    for w in workers: w.start()
    time.sleep(1.0)  # warmup pros tiles aparecerem

    use_window = not a.record and (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    writer = None; t0 = time.perf_counter(); frames = 0
    if use_window:
        cv2.namedWindow("gpuvideo — mosaico", cv2.WINDOW_AUTOSIZE)
    try:
        while True:
            canvas = compose(workers, gpu, cpu)
            if len(gpu.samples) > 200: del gpu.samples[:-100]
            if use_window:
                cv2.imshow("gpuvideo — mosaico", canvas)
                if (cv2.waitKey(20) & 0xFF) in (ord("q"), 27): break
            else:
                if writer is None:
                    h, w = canvas.shape[:2]
                    writer = cv2.VideoWriter(a.record, cv2.VideoWriter_fourcc(*"mp4v"), 25, (w, h))
                writer.write(canvas); frames += 1
                time.sleep(0.04)
                if time.perf_counter() - t0 >= a.seconds: break
    finally:
        for w in workers: w.stop.set()
        for w in workers: w.join(timeout=2)
        gpu.stop()
        if writer is not None: writer.release(); print("gravado em", a.record)
        if use_window: cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

"""Comparação lado a lado: TensorRT (.engine) vs PyTorch (.pt), ao vivo.

Justiça: os modelos NÃO rodam ao mesmo tempo (senão dividem a GPU capada a 30W +
o GIL do Python e empatam). Uma única thread ALTERNA os dois em rajadas — cada um
roda sozinho com a GPU inteira — e cada painel mostra seu fps REAL. Os frames são
decodificados uma vez (em memória), então a diferença medida é só a INFERÊNCIA.

    .venv-trt/bin/python examples/engine_vs_pt.py [segundos]

Tecla 'q'/ESC fecha. Precisa do venv com TensorRT (.venv-trt).
"""
import sys
import threading
import time

import cv2
import numpy as np
from ultralytics import YOLO

VIDEO = "15974169_3840_2160_25fps.mp4"
N = 150          # frames pré-carregados
DW = 640         # largura de cada painel
BURST = 30       # frames por rajada de cada modelo
SECONDS = float(sys.argv[1]) if len(sys.argv) > 1 else 0   # 0 = sem limite

# --- pré-carrega os frames (decode uma vez, fora da conta) ---
cap = cv2.VideoCapture(VIDEO, cv2.CAP_FFMPEG)
frames = []
while len(frames) < N:
    ok, f = cap.read()
    if not ok:
        break
    h, w = f.shape[:2]
    frames.append(cv2.resize(f, (DW, int(h * DW / w))))
cap.release()
print(f"{len(frames)} frames pré-carregados ({DW}px)")

MODELS = [("TensorRT  .engine", "yolo11n.engine"), ("PyTorch  .pt", "yolo11n.pt")]
latest, fps = {}, {}
running = True


def worker():
    """Uma thread só: alterna os modelos em rajadas (cada um sozinho na GPU)."""
    yolos = [(label, YOLO(path)) for label, path in MODELS]
    for _, m in yolos:                                      # warmup
        for f in frames[:8]:
            m.predict(f, imgsz=640, device=0, half=True, verbose=False)
    i = 0
    while running:
        for label, m in yolos:
            t = time.perf_counter()
            for _ in range(BURST):
                r = m.predict(frames[i % len(frames)], imgsz=640, device=0,
                              half=True, verbose=False)[0]
                i += 1
            dt = time.perf_counter() - t
            latest[label], fps[label] = r.plot(), BURST / dt if dt else 0
            if not running:
                break


threading.Thread(target=worker, daemon=True).start()

t0 = time.time()
blank = np.zeros((int(DW * 9 / 16), DW, 3), np.uint8)
cv2.putText(blank, "aquecendo...", (DW // 3, DW // 4), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 180, 0), 2)
while True:
    panels = []
    for label, _ in MODELS:
        img = latest.get(label, blank).copy()
        cv2.rectangle(img, (0, 0), (img.shape[1], 34), (0, 0, 0), -1)
        cv2.putText(img, f"{label}: {fps.get(label, 0):4.0f} fps", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        panels.append(img)
    h = min(p.shape[0] for p in panels)
    panels = [cv2.resize(p, (int(p.shape[1] * h / p.shape[0]), h)) for p in panels]
    cv2.imshow("engine (esq) vs .pt (dir) — gpuvideo", cv2.hconcat(panels))
    if (cv2.waitKey(30) & 0xFF) in (ord("q"), 27):
        break
    if SECONDS and time.time() - t0 > SECONDS:
        break
running = False
cv2.destroyAllWindows()
print(f"fps -> {', '.join(f'{l}: {fps.get(l,0):.0f}' for l,_ in MODELS)}")

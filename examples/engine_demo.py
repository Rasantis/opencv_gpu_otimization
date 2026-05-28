"""Demo do gpuvideo rodando com o engine TensorRT (.engine), no venv 3.12.

Rode com o venv que tem o TensorRT:
    .venv-trt/bin/python examples/engine_demo.py
Tecla 'q' ou ESC fecha. Backend opencv (cv2.VideoCapture) porque o venv 3.12 não
tem a build CUDA do OpenCV (cudacodec) nem o GStreamer — a inferência é o engine.
"""
import sys
import time

sys.path.insert(0, "src")
import cv2  # noqa: E402
from gpuvideo.analytics import VideoAnalytics, LineCounter  # noqa: E402

MODEL = sys.argv[1] if len(sys.argv) > 1 else "yolo11n.engine"
SECONDS = float(sys.argv[2]) if len(sys.argv) > 2 else 0  # 0 = sem limite

va = VideoAnalytics(
    "15974169_3840_2160_25fps.mp4", model=MODEL, classes=["person"],
    backend="opencv", engine="cpu",  # decode FFmpeg na CPU (venv 3.12 sem cudacodec)
    proc_max_side=960, annotate=True, trails=True, loop=True,
)
va.add(LineCounter([(0, 0.55), (1, 0.55)], name="fluxo"))

t0 = time.time()
win = f"gpuvideo + {MODEL}"
n = 0
for frame, tracks, events in va.run(should_stop=lambda: SECONDS and time.time() - t0 > SECONDS):
    if frame is not None:
        cv2.imshow(win, frame)
        n += 1
        if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
            break
cv2.destroyAllWindows()
print(f"frames exibidos: {n} | {n / (time.time() - t0):.1f} fps")

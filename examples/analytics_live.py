#!/usr/bin/env python3
"""Analytics ao vivo: YOLO11 + ByteTrack + soluções (contagem / dwell / heatmap / invasão).

    .venv-yolo/bin/python examples/analytics_live.py [video.mp4|rtsp://...|webcam]
    .venv-yolo/bin/python examples/analytics_live.py video.mp4 --record out.mp4

Mostra: caixas + IDs de track, linha de contagem (IN/OUT), zona de invasão (alerta),
mapa de calor, e um HUD com FPS + últimos eventos. Teclas: q/ESC sai.
"""
from __future__ import annotations
import os, sys, time, argparse, glob, collections

try:
    import gpuvideo  # noqa
except ModuleNotFoundError:
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import cv2
from gpuvideo.analytics import (VideoAnalytics, LineCounter, IntrusionZone,
                                Heatmap, DwellZone)


def pick_source(arg):
    if arg == "webcam":
        return "/dev/video0"
    if arg and (os.path.exists(arg) or "://" in arg or arg.startswith("/dev/")):
        return arg
    for c in glob.glob("*.mp4") + glob.glob("*.mkv"):
        return c
    return "/dev/video0"


def shadow(img, txt, org, scale, color, th=2):
    cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), th + 2, cv2.LINE_AA)
    cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, th, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source", nargs="?", default=None)
    ap.add_argument("--model", default="yolo11n.pt")
    ap.add_argument("--record", default=None)
    ap.add_argument("--seconds", type=float, default=20.0)
    a = ap.parse_args()

    src = pick_source(a.source)
    print(f"Fonte: {src} | modelo: {a.model}")

    # Soluções (coords normalizadas 0-1): linha no meio, zona de invasão à esquerda, heatmap.
    alerts = []
    va = (VideoAnalytics(src, model=a.model, classes=["person"])
          .add(Heatmap())
          .add(LineCounter([(0.0, 0.55), (1.0, 0.55)], name="fluxo"))
          .add(DwellZone([(0.62, 0.35), (0.98, 0.35), (0.98, 0.95), (0.62, 0.95)],
                         name="permanencia", alert_s=5))
          .add(IntrusionZone([(0.02, 0.25), (0.30, 0.25), (0.30, 0.95), (0.02, 0.95)],
                             name="restrito", on_alert=lambda e: alerts.append(e))))

    use_window = not a.record and (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    if use_window:
        cv2.namedWindow("gpuvideo analytics", cv2.WINDOW_AUTOSIZE)
    writer = None
    times = collections.deque(maxlen=30)
    log = collections.deque(maxlen=6)
    t0 = time.perf_counter()

    for frame, tracks, events in va.run():
        now = time.perf_counter(); times.append(now)
        fps = (len(times) - 1) / (times[-1] - times[0]) if len(times) > 1 else 0.0
        for e in events:
            log.appendleft(str(e)); print(e, flush=True)

        # HUD
        h, w = frame.shape[:2]
        sub = frame[0:96, 0:w].copy(); cv2.rectangle(sub, (0, 0), (w, 96), (15, 15, 15), -1)
        frame[0:96, 0:w] = cv2.addWeighted(sub, 0.5, frame[0:96, 0:w], 0.5, 0)
        shadow(frame, f"{fps:4.1f} FPS", (16, 40), 1.1, (80, 255, 120), 3)
        shadow(frame, f"tracks ativos: {len(tracks)}", (16, 74), 0.6, (230, 230, 230), 1)
        for i, line in enumerate(log):
            shadow(frame, line[:70], (16, 130 + i * 22), 0.5, (60, 220, 255), 1)

        disp = frame if w <= 1280 else cv2.resize(frame, (1280, int(h * 1280 / w)))
        if use_window:
            cv2.imshow("gpuvideo analytics", disp)
            if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                break
        else:
            if writer is None:
                writer = cv2.VideoWriter(a.record, cv2.VideoWriter_fourcc(*"mp4v"),
                                         25, (disp.shape[1], disp.shape[0]))
            writer.write(disp)
            if time.perf_counter() - t0 >= a.seconds:
                break

    if writer is not None:
        writer.release(); print("gravado em", a.record)
    if use_window:
        cv2.destroyAllWindows()
    print(f"\nResumo: {len(alerts)} alerta(s) de invasão.")


if __name__ == "__main__":
    main()

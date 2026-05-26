#!/usr/bin/env python3
"""Analytics ao vivo — UMA solução por vídeo, com FPS medido.

    # produção (event-only, FPS cheio, sem desenhar):
    .venv-yolo/bin/python examples/analytics_live.py video.mp4 --solution counting --no-annotate

    # ao vivo com janela (desenha; processa em 960px p/ não derrubar FPS):
    .venv-yolo/bin/python examples/analytics_live.py video.mp4 --solution intrusion

Soluções: counting | dwell | heatmap | intrusion   (uma por vez)
Teclas (janela): q/ESC sai.
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


def make_solution(kind, alerts):
    if kind == "counting":
        return LineCounter([(0.0, 0.55), (1.0, 0.55)], name="fluxo")
    if kind == "dwell":
        return DwellZone([(0.3, 0.3), (0.95, 0.3), (0.95, 0.95), (0.3, 0.95)],
                         name="permanencia", alert_s=5)
    if kind == "heatmap":
        return Heatmap()
    if kind == "intrusion":
        return IntrusionZone([(0.02, 0.25), (0.35, 0.25), (0.35, 0.95), (0.02, 0.95)],
                             name="restrito", on_alert=lambda e: alerts.append(e))
    raise SystemExit(f"solução desconhecida: {kind}")


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
    ap.add_argument("--solution", default="counting",
                    choices=["counting", "dwell", "heatmap", "intrusion"])
    ap.add_argument("--model", default="yolo11n.pt")
    ap.add_argument("--proc", type=int, default=960, help="lado maior p/ processar (acelera)")
    ap.add_argument("--no-annotate", action="store_true", help="event-only (produção)")
    ap.add_argument("--record", default=None)
    ap.add_argument("--seconds", type=float, default=20.0)
    a = ap.parse_args()

    src = pick_source(a.source)
    alerts = []
    annotate = not a.no_annotate          # event-only desliga o desenho (FPS cheio)
    va = VideoAnalytics(src, model=a.model, classes=["person"],
                        annotate=annotate, proc_max_side=a.proc)
    va.add(make_solution(a.solution, alerts))
    print(f"Fonte: {src} | solução: {a.solution} | annotate={annotate} | proc={a.proc}px")

    use_window = annotate and not a.record and (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    if use_window:
        cv2.namedWindow("gpuvideo analytics", cv2.WINDOW_AUTOSIZE)
    writer = None
    times = collections.deque(maxlen=60)
    log = collections.deque(maxlen=6)
    t0 = time.perf_counter()
    n = 0

    for frame, tracks, events in va.run():
        now = time.perf_counter(); times.append(now); n += 1
        fps = (len(times) - 1) / (times[-1] - times[0]) if len(times) > 1 else 0.0
        for e in events:
            log.appendleft(str(e)); print(e, flush=True)

        if frame is not None:  # modo annotate
            h, w = frame.shape[:2]
            sub = frame[0:88, 0:w].copy(); cv2.rectangle(sub, (0, 0), (w, 88), (15, 15, 15), -1)
            frame[0:88, 0:w] = cv2.addWeighted(sub, 0.5, frame[0:88, 0:w], 0.5, 0)
            shadow(frame, f"{fps:4.1f} FPS  [{a.solution}]", (16, 38), 1.0, (80, 255, 120), 2)
            shadow(frame, f"tracks: {len(tracks)}", (16, 70), 0.55, (230, 230, 230), 1)
            for i, line in enumerate(log):
                shadow(frame, line[:64], (16, 116 + i * 20), 0.45, (60, 220, 255), 1)
            if use_window:
                cv2.imshow("gpuvideo analytics", frame)
                if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                    break
            elif a.record:
                if writer is None:
                    writer = cv2.VideoWriter(a.record, cv2.VideoWriter_fourcc(*"mp4v"),
                                             25, (w, h))
                writer.write(frame)
                if time.perf_counter() - t0 >= a.seconds:
                    break
        else:  # event-only: imprime FPS periodicamente
            if n % 30 == 0:
                print(f"  ... {fps:.1f} fps | {n} frames", flush=True)
            if time.perf_counter() - t0 >= a.seconds:
                break

    dt = time.perf_counter() - t0
    if writer is not None:
        writer.release(); print("gravado em", a.record)
    if use_window:
        cv2.destroyAllWindows()
    print(f"\nMédia: {n/dt:.1f} fps em {dt:.1f}s | solução={a.solution} | "
          f"{len(alerts)} alerta(s)")


if __name__ == "__main__":
    main()

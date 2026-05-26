#!/usr/bin/env python3
"""Exemplos de uso do gpuvideo - facilidade de importacao e escala.

Rode:  python3 examples.py
"""
from gpuvideo import VideoStream, GstStream, CvStream, MultiStream

VIDEO = "heavy_1080p.mp4"


def ex1_basico():
    """O jeito mais simples: GPU (NVDEC) por padrao."""
    print("\n[1] uso basico - VideoStream (GPU por padrao)")
    with VideoStream(VIDEO) as stream:
        for frame in stream:
            # frame.array e' um numpy BGR pronto pra OpenCV/inferencia
            if frame.index == 0:
                print(f"    primeiro frame: {frame}")
            if frame.index >= 100:
                break
    print("    ok")


def ex2_escolhendo_backend():
    """Trocar de backend/engine e' um parametro."""
    print("\n[2] escolhendo backend e engine")
    configs = [
        ("gstreamer", "gpu"),   # NVDEC
        ("gstreamer", "cpu"),   # libav
        ("opencv", "gpu"),      # cv2 + NVDEC via GStreamer
        ("opencv", "cpu"),      # cv2 + FFmpeg
    ]
    for backend, engine in configs:
        with VideoStream(VIDEO, backend=backend, engine=engine) as s:
            n = sum(1 for _ in zip(range(50), s))
        print(f"    {backend:10}/{engine:3}: leu {n} frames ({s.width}x{s.height})")


def ex3_processamento():
    """Captura + processamento (ex.: deteccao de bordas)."""
    print("\n[3] captura + processamento por frame")
    import cv2
    with GstStream(VIDEO, engine="gpu") as s:
        for frame in s:
            gray = cv2.cvtColor(frame.array, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 80, 160)  # noqa: F841
            if frame.index >= 30:
                break
    print("    ok (Canny em 30 frames)")


def ex4_escala():
    """Varias streams em paralelo (teste de escala)."""
    print("\n[4] escala - 4 streams simultaneos via GPU")
    res = MultiStream.replicate(VIDEO, n=4, mode="gstreamer-gpu").run(max_frames=80)
    print("   ", res)


def ex5_pipeline_custom():
    """Voce pode passar um pipeline GStreamer proprio."""
    print("\n[5] pipeline GStreamer customizado")
    pipe = (f'filesrc location="{VIDEO}" ! qtdemux ! h264parse ! '
            f'nvh264dec ! cudadownload ! videoconvert ! '
            f'video/x-raw,format=BGR ! appsink name=sink sync=false')
    with GstStream(VIDEO, pipeline=pipe) as s:
        n = sum(1 for _ in zip(range(20), s))
    print(f"    leu {n} frames com pipeline manual")


if __name__ == "__main__":
    import os
    if not os.path.exists(VIDEO):
        print(f"Gere o video de teste antes: python3 -m gpuvideo gen {VIDEO} --res 1080p")
        raise SystemExit(1)
    ex1_basico()
    ex2_escolhendo_backend()
    ex3_processamento()
    ex4_escala()
    ex5_pipeline_custom()
    print("\nTodos os exemplos rodaram.")

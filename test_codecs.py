#!/usr/bin/env python3
"""Verificação end-to-end de suporte a codecs/containers.

Gera um clipe curto por codec (via GStreamer) e testa a leitura por todos os
backends do gpuvideo: GStreamer GPU/CPU, OpenCV CPU/GPU e cudacodec (NVDEC nativo).
Imprime uma matriz de suporte. Não exige rede; usa videotestsrc + encoders locais.
"""
from __future__ import annotations
import os, subprocess, sys

W, H = 640, 480

# codec -> (encoder gstreamer, parser p/ mux, mux, extensao)
GEN = {
    "h264":  ("nvh264enc",        "h264parse",       "mp4mux",       "mp4"),
    "h265":  ("nvh265enc",        "h265parse",       "mp4mux",       "mp4"),
    "vp8":   ("vp8enc",           "",                "webmmux",      "webm"),
    "vp9":   ("vp9enc",           "",                "webmmux",      "webm"),
    "av1":   ("av1enc cpu-used=8","av1parse",        "matroskamux",  "mkv"),
    "mpeg2": ("avenc_mpeg2video", "mpegvideoparse",  "mpegtsmux",    "ts"),
    "mjpeg": ("jpegenc",          "jpegparse",       "qtmux",        "mov"),
}
# av1 é lento de codificar -> resolução/menos frames
SMALL = {"av1": (320, 240, 10)}


def gen(codec):
    enc, parse, mux, ext = GEN[codec]
    w, h, nf = SMALL.get(codec, (W, H, 30))
    path = f"codectest_{codec}.{ext}"
    if os.path.exists(path):
        return path
    chain = ["gst-launch-1.0", "-q", "-e", "videotestsrc", f"num-buffers={nf}",
             "!", f"video/x-raw,format=I420,width={w},height={h},framerate=25/1", "!"]
    chain += enc.split() + ["!"]
    if parse:
        chain += [parse, "!"]
    chain += [mux, "!", "filesink", f"location={path}"]
    r = subprocess.run(chain, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    if r.returncode != 0 or not os.path.exists(path) or os.path.getsize(path) == 0:
        return None
    return path


def read_n(make, n=10):
    try:
        s = make(); s.open()
        c = 0
        for _ in range(n):
            f = s.read()
            if f is None:
                break
            c += 1
        s.close()
        return c
    except Exception as e:
        return f"ERRO:{type(e).__name__}"


def main():
    from gpuvideo import GstStream, CvStream
    from gpuvideo.pipelines import discover_codec
    from gpuvideo.cudacodec import cudacodec_available, CudaCodecStream
    has_cuda = cudacodec_available()
    print(f"cudacodec disponível: {has_cuda}\n")

    cols = ["detect", "gst-gpu", "gst-cpu", "cv-cpu", "cv-gpu", "cudacodec"]
    print(f"{'codec':<8}" + "".join(f"{c:>11}" for c in cols))
    print("-" * (8 + 11 * len(cols)))
    for codec in GEN:
        path = gen(codec)
        if not path:
            print(f"{codec:<8}{'(falha ao gerar clipe)':>33}")
            continue
        det = discover_codec(path) or "-"
        r_ggpu = read_n(lambda: GstStream(path, engine="gpu"))
        r_gcpu = read_n(lambda: GstStream(path, engine="cpu"))
        r_ccpu = read_n(lambda: CvStream(path, mode="cpu"))
        r_cgpu = read_n(lambda: CvStream(path, mode="gpu"))
        r_cuda = read_n(lambda: CudaCodecStream(path)) if has_cuda else "n/a"
        def fmt(x): return f"{x}f" if isinstance(x, int) else str(x)
        vals = [det, fmt(r_ggpu), fmt(r_gcpu), fmt(r_ccpu), fmt(r_cgpu), fmt(r_cuda)]
        print(f"{codec:<8}" + "".join(f"{v:>11}" for v in vals))
    print("\nf = frames lidos OK | '-' indeterminado | ERRO:X = falhou")


if __name__ == "__main__":
    main()

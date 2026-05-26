#!/usr/bin/env python3
"""Benchmark do caminho NVDEC nativo do OpenCV (cv2.cudacodec).

Requer a build CUDA do OpenCV. O script tenta localizar e carregar o cv2
compilado (em /home/rafael/opencv_build/build) antes de importar.

Compara, no mesmo vídeo:
  - os 4 modos base (opencv-cpu/gpu, gstreamer-cpu/gpu)
  - opencv-cuda           : NVDEC -> BGR (na GPU) -> download frame inteiro
  - opencv-cuda (gpu-op)   : NVDEC -> resize+gray NA GPU -> download só o resultado
"""
from __future__ import annotations

import os
import sys
import time

# Para o modo opencv-cuda é preciso um OpenCV compilado com CUDA + cudacodec
# (veja docs/BUILD_CUDA.md). Se você usa o cv2 só do build local (sem instalar),
# aponte GPUVIDEO_CV2_PATH para o diretório que contém o cv2.*.so.
_extra = os.environ.get("GPUVIDEO_CV2_PATH")
if _extra and os.path.isdir(_extra):
    sys.path.insert(0, _extra)

import cv2  # noqa: E402
import numpy as np  # noqa: E402

print(f"cv2 {cv2.__version__} | cudacodec={hasattr(cv2,'cudacodec')} | "
      f"cuda devices={cv2.cuda.getCudaEnabledDeviceCount()}")

from gpuvideo.benchmark import Benchmark, OPS  # noqa: E402
from gpuvideo.cudacodec import CudaCodecStream, gpu_op_light  # noqa: E402
from gpuvideo import ALL_MODES  # noqa: E402

VIDEO = sys.argv[1] if len(sys.argv) > 1 else "heavy_1080p.mp4"
FRAMES = int(sys.argv[2]) if len(sys.argv) > 2 else 250


def bench_gpu_resident(video, frames, warmup=20):
    """NVDEC + processamento 100% na GPU, baixando só o resultado pequeno."""
    s = CudaCodecStream(video, gpu_op=gpu_op_light, color="BGR", download=True)
    s.open()
    for _ in range(warmup):
        if s.read() is None:
            break
    t0 = time.perf_counter()
    n = 0
    while n < frames:
        f = s.read()
        if f is None:
            break
        n += 1
    wall = time.perf_counter() - t0
    s.close()
    return n, wall, (n / wall if wall else 0)


if __name__ == "__main__":
    if not hasattr(cv2, "cudacodec"):
        print("\nERRO: este cv2 não tem cudacodec. Build CUDA ainda não pronta?")
        sys.exit(1)

    print(f"\n=== Comparativo COMPLETO ({VIDEO}, {FRAMES} frames, op=light) ===")
    b = Benchmark(VIDEO, frames=FRAMES, warmup=20, op="light", modes=ALL_MODES)
    b.compare()
    b.report()
    b.to_json("results/capture_with_cuda.json")

    print("\n=== BÔNUS: NVDEC + processamento 100% na GPU (download só do resultado) ===")
    n, wall, fps = bench_gpu_resident(VIDEO, FRAMES)
    print(f"opencv-cuda (gpu-op): {fps:.1f} fps ({n} frames em {wall:.2f}s)")
    print("   -> decode + resize + grayscale sem trazer o frame BGR inteiro pra CPU")

"""gpuvideo - captura e processamento de video acelerados por GPU.

API de importacao minima:

    from gpuvideo import VideoStream

    with VideoStream("video.mp4") as stream:      # GPU (NVDEC) por padrao
        for frame in stream:
            cv2.imshow("x", frame.array)

Backends:
    - GstStream  : GStreamer nativo (NVDEC) -> appsink -> numpy
    - CvStream   : OpenCV (FFmpeg na CPU, ou GStreamer NVDEC na GPU)

Escala:
    from gpuvideo import MultiStream

Benchmark:
    from gpuvideo import Benchmark
"""
from __future__ import annotations

from .frame import Frame
from .base import BaseStream
from .gstreamer import GstStream
from .opencv import CvStream

__all__ = [
    "Frame", "BaseStream", "GstStream", "CvStream",
    "VideoStream", "make_stream", "MODES", "ALL_MODES",
    "MultiStream", "Benchmark", "BenchmarkResult",
]

__version__ = "0.1.0"

# Modos do comparativo: framework x engine.
MODES = ("opencv-cpu", "opencv-gpu", "gstreamer-cpu", "gstreamer-gpu")
# opencv-cuda exige uma build CUDA do OpenCV (cv2.cudacodec / NVDEC nativo).
ALL_MODES = MODES + ("opencv-cuda",)


def make_stream(mode: str, source, **kwargs) -> BaseStream:
    """Cria o stream do modo pedido ('opencv-cpu', 'gstreamer-gpu', ...)."""
    mode = mode.lower()
    if mode == "opencv-cpu":
        return CvStream(source, mode="cpu", **kwargs)
    if mode == "opencv-gpu":
        return CvStream(source, mode="gpu", **kwargs)
    if mode == "gstreamer-cpu":
        return GstStream(source, engine="cpu", **kwargs)
    if mode == "gstreamer-gpu":
        return GstStream(source, engine="gpu", **kwargs)
    if mode in ("opencv-cuda", "opencv-cuda-native"):
        from .cudacodec import CudaCodecStream
        return CudaCodecStream(source, **kwargs)
    raise ValueError(f"modo desconhecido: {mode!r}. Use um de {ALL_MODES}")


def VideoStream(source, *, backend: str = "auto", engine: str = "gpu",
                **kwargs) -> BaseStream:
    """Fabrica de stream com API amigavel.

    backend: "auto" | "gstreamer" | "opencv"
    engine : "gpu" | "cpu"
    """
    backend = backend.lower()
    if backend in ("auto", "gstreamer", "gst"):
        return GstStream(source, engine=engine, **kwargs)
    if backend in ("opencv", "cv2"):
        return CvStream(source, mode=engine, **kwargs)
    raise ValueError(f"backend desconhecido: {backend!r}")


# Imports tardios para nao puxar dependencias pesadas no import base.
def __getattr__(name):  # PEP 562
    if name == "MultiStream":
        from .multistream import MultiStream
        return MultiStream
    if name == "Benchmark":
        from .benchmark import Benchmark
        return Benchmark
    if name == "BenchmarkResult":
        from .benchmark import BenchmarkResult
        return BenchmarkResult
    raise AttributeError(name)

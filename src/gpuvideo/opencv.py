"""Backend OpenCV.

Dois modos:
  - "cpu": cv2.VideoCapture + FFmpeg (decode na CPU). E' o "OpenCV nativo"
           que a maioria usa fora da caixa.
  - "gpu": cv2.VideoCapture consumindo um pipeline GStreamer NVDEC via
           CAP_GSTREAMER -- assim o OpenCV usa a GPU mesmo sem build CUDA.

Obs.: o pacote python3-opencv do Ubuntu nao traz o modulo cv2.cuda
(cudacodec/NVDEC). Para decode NVDEC "nativo" dentro do cv2 seria preciso
compilar o OpenCV do fonte com CUDA. O modo "gpu" aqui entrega o mesmo
ganho de GPU sem essa compilacao.
"""
from __future__ import annotations

from typing import Optional

from .base import BaseStream
from .frame import Frame
from . import pipelines


class CvStream(BaseStream):
    backend_name = "opencv"

    def __init__(self, source, *, mode: str = "cpu",
                 codec: Optional[str] = None, output_format: str = "BGR",
                 convert_threads: int = 4, stream_id: str = "0",
                 pipeline: Optional[str] = None) -> None:
        super().__init__(source, stream_id=stream_id)
        if mode not in ("cpu", "gpu"):
            raise ValueError("mode deve ser 'cpu' ou 'gpu'")
        self.mode = mode
        self._codec = codec
        self._output_format = output_format
        self._convert_threads = convert_threads
        self._explicit_pipeline = pipeline
        self._cap = None
        self._gst_str: Optional[str] = None

    def open(self) -> "CvStream":
        import cv2
        self._cv2 = cv2
        if self.mode == "cpu":
            src = self.source if isinstance(self.source, int) else str(self.source)
            self._cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
        else:  # gpu via GStreamer
            self._gst_str = self._explicit_pipeline or pipelines.build_pipeline(
                self.source, engine="gpu", output_format=self._output_format,
                codec=self._codec, sync=False, max_buffers=2, drop=False,
                convert_threads=self._convert_threads, appsink_name="opencvsink",
                for_opencv=True,
            )
            self._cap = cv2.VideoCapture(self._gst_str, cv2.CAP_GSTREAMER)

        if not self._cap.isOpened():
            raise RuntimeError(
                f"OpenCV nao abriu a fonte (mode={self.mode}). "
                + (f"Pipeline: {self._gst_str}" if self._gst_str else str(self.source))
            )

        self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = float(self._cap.get(cv2.CAP_PROP_FPS) or 0.0)
        fc = self._cap.get(cv2.CAP_PROP_FRAME_COUNT)
        self.frame_count = int(fc) if fc and fc > 0 else -1
        self._opened = True
        self._index = 0
        return self

    def read(self) -> Optional[Frame]:
        if not self._opened:
            self.open()
        ok, array = self._cap.read()
        if not ok or array is None:
            return None
        if not self.width:
            self.height, self.width = array.shape[:2]
        frame = Frame(
            array=array, index=self._index,
            width=self.width or array.shape[1],
            height=self.height or array.shape[0],
            pts_ns=None, capture_monotonic=self._now(), stream_id=self.stream_id,
        )
        self._index += 1
        return frame

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
        self._cap = None
        self._opened = False

    @property
    def pipeline_string(self) -> Optional[str]:
        return self._gst_str

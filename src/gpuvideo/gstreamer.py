"""Backend GStreamer nativo: NVDEC (GPU) -> appsink -> numpy.

Este e' o caminho de menor overhead: o frame sai do decoder de hardware,
e' lido direto do ``appsink`` e copiado para um ``numpy.ndarray`` proprio.
"""
from __future__ import annotations

import threading
from typing import Optional

import numpy as np

from .base import BaseStream
from .frame import Frame
from . import pipelines

_GST_READY = False
_GST_LOCK = threading.Lock()


def _ensure_gst():
    global _GST_READY
    with _GST_LOCK:
        if _GST_READY:
            return
        import gi
        gi.require_version("Gst", "1.0")
        gi.require_version("GstApp", "1.0")
        gi.require_version("GstVideo", "1.0")
        # Importar GstApp/GstVideo (nao so' require_version) e' o que vincula
        # metodos como AppSink.try_pull_sample e VideoInfo.new_from_caps.
        from gi.repository import Gst, GstApp, GstVideo  # noqa: F401
        if not Gst.is_initialized():
            Gst.init(None)
        _GST_READY = True


class GstStream(BaseStream):
    """Captura via GStreamer com decode na GPU (NVDEC) por padrao.

    Parameters
    ----------
    source : str | int
        Arquivo, rtsp://, http://, indice de camera, ou "test".
    engine : {"gpu", "cpu"}
        "gpu" usa NVDEC; "cpu" usa libav (avdec_*), util pra comparacao.
    pipeline : str, opcional
        Pipeline GStreamer pronto (sobrescreve a construcao automatica).
        Deve terminar em ``appsink name=sink``.
    """

    backend_name = "gstreamer"

    def __init__(self, source, *, engine: str = "gpu",
                 pipeline: Optional[str] = None, codec: Optional[str] = None,
                 sync: bool = False, max_buffers: int = 4, drop: bool = False,
                 output_format: str = "BGR", convert_threads: int = 4,
                 read_timeout_s: float = 5.0, stream_id: str = "0") -> None:
        super().__init__(source, stream_id=stream_id)
        _ensure_gst()
        self.engine = engine
        self.output_format = output_format
        self._read_timeout_ns = int(read_timeout_s * 1e9)
        self._channels = 1 if output_format in ("GRAY8",) else 3
        self._pipeline_str = pipeline or pipelines.build_pipeline(
            source, engine=engine, output_format=output_format, codec=codec,
            sync=sync, max_buffers=max_buffers, drop=drop,
            convert_threads=convert_threads, appsink_name="sink",
        )
        self._pipeline = None
        self._appsink = None
        self._bus = None
        self._vinfo = None  # cache de VideoInfo (stride/dims)

    # ------------------------------------------------------------------
    def open(self) -> "GstStream":
        import gi
        from gi.repository import Gst
        self._Gst = Gst
        self._pipeline = Gst.parse_launch(self._pipeline_str)
        self._appsink = self._pipeline.get_by_name("sink")
        if self._appsink is None:
            raise RuntimeError("Pipeline nao contem um appsink chamado 'sink'.")
        self._bus = self._pipeline.get_bus()

        ret = self._pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            self._dump_error()
            raise RuntimeError(f"Falha ao iniciar pipeline:\n{self._pipeline_str}")
        # Espera o pipeline chegar em PLAYING (preroll).
        ret, _, _ = self._pipeline.get_state(self._read_timeout_ns)
        if ret == Gst.StateChangeReturn.FAILURE:
            self._dump_error()
            raise RuntimeError(f"Pipeline nao pre-rolou:\n{self._pipeline_str}")
        self._opened = True
        self._index = 0
        return self

    # ------------------------------------------------------------------
    def _setup_vinfo(self, caps):
        import gi
        from gi.repository import GstVideo
        self._vinfo = GstVideo.VideoInfo.new_from_caps(caps)
        st = caps.get_structure(0)
        ok_w, w = st.get_int("width")
        ok_h, h = st.get_int("height")
        self.width = w if ok_w else self._vinfo.width
        self.height = h if ok_h else self._vinfo.height
        ok_fr, num, den = st.get_fraction("framerate")
        if ok_fr and den:
            self.fps = num / den

    def _extract(self, sample) -> np.ndarray:
        Gst = self._Gst
        buf = sample.get_buffer()
        caps = sample.get_caps()
        if self._vinfo is None:
            self._setup_vinfo(caps)

        h, w, c = self.height, self.width, self._channels
        ok, mapinfo = buf.map(Gst.MapFlags.READ)
        if not ok:
            raise RuntimeError("buffer.map() falhou")
        try:
            raw = np.frombuffer(mapinfo.data, dtype=np.uint8)
            stride = self._vinfo.stride[0] if self._vinfo else w * c
            expected_tight = w * c
            if stride == expected_tight and raw.size >= h * w * c:
                arr = raw[: h * w * c].reshape(h, w, c).copy()
            else:
                # Linha tem padding: recorta usando o stride real.
                arr = raw[: h * stride].reshape(h, stride)[:, : w * c]
                arr = arr.reshape(h, w, c).copy()
        finally:
            buf.unmap(mapinfo)
        if c == 1:
            arr = arr[:, :, 0]
        return arr

    # ------------------------------------------------------------------
    def read(self) -> Optional[Frame]:
        if not self._opened:
            self.open()
        Gst = self._Gst

        # Erros do bus tem prioridade.
        msg = self._bus.pop_filtered(Gst.MessageType.ERROR)
        if msg is not None:
            err, dbg = msg.parse_error()
            raise RuntimeError(f"GStreamer: {err.message} | {dbg}")

        sample = self._appsink.try_pull_sample(self._read_timeout_ns)
        if sample is None:
            # Fim do stream ou timeout.
            return None

        arr = self._extract(sample)
        buf = sample.get_buffer()
        pts = buf.pts if buf.pts != Gst.CLOCK_TIME_NONE else None
        frame = Frame(
            array=arr, index=self._index, width=self.width, height=self.height,
            pts_ns=pts, capture_monotonic=self._now(), stream_id=self.stream_id,
        )
        self._index += 1
        return frame

    # ------------------------------------------------------------------
    def _dump_error(self):
        if self._bus is None:
            return
        from gi.repository import Gst
        msg = self._bus.pop_filtered(Gst.MessageType.ERROR)
        if msg:
            err, dbg = msg.parse_error()
            print(f"[GstStream] ERRO: {err.message}\n{dbg}")

    def close(self) -> None:
        if self._pipeline is not None:
            self._pipeline.set_state(self._Gst.State.NULL)
        self._pipeline = None
        self._appsink = None
        self._bus = None
        self._opened = False

    @property
    def pipeline_string(self) -> str:
        return self._pipeline_str

"""Construcao de pipelines GStreamer (GPU/NVDEC e CPU/libav).

Tudo aqui e' so' montagem de *string* de pipeline -- nao instancia nada.
Assim os backends GStreamer e OpenCV (via CAP_GSTREAMER) compartilham
exatamente a mesma logica de pipeline, mantendo o comparativo justo.

GPU neste host (RTX 3050, GStreamer 1.28):
    ... ! nvh264dec ! cudadownload ! videoconvert ! BGR
    (decode NVDEC na GPU; conversao NV12->BGR na CPU, pois este build
     do GStreamer nao traz `cudaconvert`/`cudascale`).
"""
from __future__ import annotations

import os
import re
from typing import Optional

# Demuxer por extensao de container.
_DEMUX_BY_EXT = {
    ".mp4": "qtdemux", ".mov": "qtdemux", ".m4v": "qtdemux", ".3gp": "qtdemux",
    ".mkv": "matroskademux", ".webm": "matroskademux",
    ".avi": "avidemux", ".ts": "tsdemux", ".mts": "tsdemux", ".flv": "flvdemux",
}

# codec -> (parser, decoder_gpu, decoder_cpu)
_CODEC = {
    "h264": ("h264parse", "nvh264dec", "avdec_h264"),
    "h265": ("h265parse", "nvh265dec", "avdec_h265"),
    "hevc": ("h265parse", "nvh265dec", "avdec_h265"),
    "vp8":  ("",          "nvvp8dec", "avdec_vp8"),
    "vp9":  ("vp9parse",  "nvvp9dec", "avdec_vp9"),
    "av1":  ("av1parse",  "nvav1dec", "av1dec"),
    "mpeg2": ("mpegvideoparse", "nvmpeg2videodec", "avdec_mpeg2video"),
    "jpeg": ("jpegparse", "nvjpegdec", "jpegdec"),
}

_RTSP_RE = re.compile(r"^rtsp://", re.I)
_HTTP_RE = re.compile(r"^https?://", re.I)


class SourceKind:
    FILE = "file"
    RTSP = "rtsp"
    HTTP = "http"
    CAMERA = "camera"
    TEST = "test"


def classify_source(source) -> str:
    """Descobre o tipo da fonte a partir do valor passado pelo usuario."""
    if isinstance(source, int):
        return SourceKind.CAMERA
    s = str(source)
    if s in ("test", "videotestsrc"):
        return SourceKind.TEST
    if _RTSP_RE.match(s):
        return SourceKind.RTSP
    if _HTTP_RE.match(s):
        return SourceKind.HTTP
    if s.startswith("/dev/video") or s.isdigit():
        return SourceKind.CAMERA
    return SourceKind.FILE


def discover_codec(path: str) -> Optional[str]:
    """Usa o GstDiscoverer para descobrir o codec de video de um arquivo.

    Retorna uma chave de ``_CODEC`` (ex.: 'h264') ou ``None`` se nao detectar.
    """
    try:
        import gi
        gi.require_version("Gst", "1.0")
        gi.require_version("GstPbutils", "1.0")
        from gi.repository import Gst, GstPbutils
        if not Gst.is_initialized():
            Gst.init(None)
        disc = GstPbutils.Discoverer.new(5 * Gst.SECOND)
        uri = path if "://" in path else "file://" + os.path.abspath(path)
        info = disc.discover_uri(uri)
        for stream in info.get_video_streams():
            caps = stream.get_caps()
            name = caps.get_structure(0).get_name() if caps else ""
            # ex.: "video/x-h264" -> "h264" ; "image/jpeg" -> "jpeg"
            short = name.split("/")[-1].replace("x-", "")
            # Normalizacoes de nomes de caps -> chave de _CODEC.
            alias = {"hevc": "h265", "mpeg": "mpeg2", "mpeg2video": "mpeg2",
                     "mjpeg": "jpeg", "motionjpeg": "jpeg"}
            short = alias.get(short, short)
            if short in _CODEC:
                return short
    except Exception:
        pass
    return None


def _appsink(name: str, *, sync: bool, max_buffers: int, drop: bool,
             for_opencv: bool) -> str:
    # O backend OpenCV nao precisa de nome no appsink (ele anexa o proprio),
    # mas e' inofensivo manter. drop/sync controlam realtime vs vazao maxima.
    parts = [
        "appsink",
        f"name={name}",
        "emit-signals=false",
        f"max-buffers={max_buffers}",
        f"drop={'true' if drop else 'false'}",
        f"sync={'true' if sync else 'false'}",
    ]
    return " ".join(parts)


def _decode_chain(codec: str, engine: str) -> str:
    parse, gpu_dec, cpu_dec = _CODEC[codec]
    decoder = gpu_dec if engine == "gpu" else cpu_dec
    chain = []
    if parse:
        chain.append(parse)
    chain.append(decoder)
    if engine == "gpu":
        # NVDEC entrega CUDAMemory/NV12 -> traz pra memoria de sistema.
        chain.append("cudadownload")
    return " ! ".join(chain)


def build_pipeline(
    source,
    *,
    engine: str = "gpu",            # "gpu" (NVDEC) ou "cpu" (libav)
    output_format: str = "BGR",      # formato entregue (BGR p/ OpenCV/numpy)
    codec: Optional[str] = None,     # forca o codec; None = autodetecta
    appsink_name: str = "sink",
    sync: bool = False,              # False = decodifica o mais rapido possivel
    max_buffers: int = 4,
    drop: bool = False,
    convert_threads: int = 4,
    rtsp_latency_ms: int = 100,
    rtsp_protocols: str = "tcp",
    for_opencv: bool = False,
) -> str:
    """Monta a string de pipeline GStreamer terminando em ``appsink``."""
    kind = classify_source(source)
    sink = _appsink(appsink_name, sync=sync, max_buffers=max_buffers,
                    drop=drop, for_opencv=for_opencv)
    convert = (f"videoconvert n-threads={convert_threads} ! "
               f"video/x-raw,format={output_format}")

    if kind == SourceKind.TEST:
        # Fonte sintetica (sem decode); util pra sanity check.
        return (f"videotestsrc is-live=false ! "
                f"video/x-raw,width=1920,height=1080,framerate=30/1 ! "
                f"{convert} ! {sink}")

    if kind == SourceKind.CAMERA:
        dev = source if str(source).startswith("/dev/") else f"/dev/video{source}"
        return f"v4l2src device={dev} ! {convert} ! {sink}"

    if kind == SourceKind.RTSP:
        c = codec or "h264"
        parse, gpu_dec, cpu_dec = _CODEC[c]
        depay = "rtph264depay" if c == "h264" else "rtph265depay"
        dec = _decode_chain(c, engine)
        # protocols=tcp: robusto p/ cameras IP e evita problemas de UDP/IPv6.
        return (f"rtspsrc location={source} latency={rtsp_latency_ms} "
                f"protocols={rtsp_protocols} ! "
                f"{depay} ! {dec} ! {convert} ! {sink}")

    # ---- arquivo (e http como uri) ----
    if kind == SourceKind.HTTP:
        src_el = f"souphttpsrc location={source}"
        ext = os.path.splitext(str(source).split('?')[0])[1].lower()
    else:
        path = os.path.abspath(str(source))
        src_el = f'filesrc location="{path}"'
        ext = os.path.splitext(path)[1].lower()

    demux = _DEMUX_BY_EXT.get(ext, "qtdemux")
    c = codec or (discover_codec(str(source)) if kind == SourceKind.FILE else None) or "h264"
    if c not in _CODEC:
        c = "h264"
    dec = _decode_chain(c, engine)
    return f"{src_el} ! {demux} ! {dec} ! {convert} ! {sink}"

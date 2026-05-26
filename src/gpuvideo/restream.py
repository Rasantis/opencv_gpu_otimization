"""Restreaming de baixa latência: ingest -> [YOLO] -> NVENC -> RTMP/SRT.

Pensado para alimentar um servidor de borda (ex.: MediaMTX) que faz o fan-out
em WebRTC/LL-HLS para o front. Dois modos:

  - transcode (padrão): 100% GPU, sem Python por frame. Menor latência.
        ingest -> nvh264dec -> nvh264enc(low-latency) -> RTMP/SRT
  - infer: decodifica, roda YOLO11, desenha as caixas e recodifica.
        ingest -> frames -> YOLO -> appsrc -> nvenc -> RTMP/SRT
"""
from __future__ import annotations

import os
from typing import Optional

from .pipelines import (classify_source, SourceKind, _CODEC, discover_codec,
                        _DEMUX_BY_EXT)
from .gstreamer import _ensure_gst, GstStream


# --------------------------- pedaços de pipeline ---------------------------
def _nvenc(codec: str, bitrate_kbps: int, gop: int, low_latency: bool) -> str:
    enc = "nvh264enc" if codec == "h264" else "nvh265enc"
    if low_latency:
        return (f"{enc} preset=low-latency-hq rc-mode=cbr bitrate={bitrate_kbps} "
                f"gop-size={gop} bframes=0")
    return f"{enc} bitrate={bitrate_kbps} gop-size={gop}"


def _sink(protocol: str, url: str, codec: str) -> str:
    parse = "h264parse" if codec == "h264" else "h265parse"
    if protocol == "rtmp":   # RTMP carrega só H.264
        return f"{parse} ! flvmux streamable=true ! rtmp2sink location={url}"
    if protocol == "srt":    # SRT carrega MPEG-TS (menor latência que RTMP)
        return f"{parse} ! mpegtsmux ! srtsink uri={url} wait-for-connection=false"
    raise ValueError(f"protocolo desconhecido: {protocol}")


def _ingest_decoded(source) -> str:
    """ingest -> raw decodificado (pronto p/ reencode ou appsink)."""
    kind = classify_source(source)
    if kind == SourceKind.RTSP:
        return (f"rtspsrc location={source} protocols=tcp latency=100 ! "
                f"rtph264depay ! h264parse ! nvh264dec")
    if kind == SourceKind.RTMP:
        return f"rtmp2src location={source} ! flvdemux ! h264parse ! nvh264dec"
    if kind == SourceKind.TEST:
        return ("videotestsrc is-live=true ! "
                "video/x-raw,width=1280,height=720,framerate=30/1 ! timeoverlay")
    if kind == SourceKind.HTTP:
        # uridecodebin resolve container/codec; saída raw pronta p/ nvenc.
        return f"uridecodebin uri={source} ! videoconvert"
    # arquivo
    ext = os.path.splitext(str(source))[1].lower()
    demux = _DEMUX_BY_EXT.get(ext, "qtdemux")
    c = discover_codec(str(source)) or "h264"
    parse, gpu_dec, _ = _CODEC[c]
    p = (parse + " ! ") if parse else ""
    return f'filesrc location="{os.path.abspath(str(source))}" ! {demux} ! {p}{gpu_dec}'


# --------------------------------- Restreamer ---------------------------------
class Restreamer:
    def __init__(self, source, out_url: str, *, codec_out: str = "h264",
                 bitrate_kbps: int = 4000, gop: int = 30, low_latency: bool = True,
                 protocol: str = "rtmp", infer: bool = False,
                 model: str = "yolo11n.pt", imgsz: int = 640, device: str = "0"):
        _ensure_gst()
        self.source = source
        self.out_url = out_url
        self.codec_out = codec_out
        self.bitrate = bitrate_kbps
        self.gop = gop
        self.low_latency = low_latency
        self.protocol = protocol
        self.infer = infer
        self.model_name = model
        self.imgsz = imgsz
        self.device = device
        self._pipeline = None

    # ---- modo transcode: 100% GStreamer/GPU ----
    def _transcode_pipeline(self) -> str:
        return (f"{_ingest_decoded(self.source)} ! "
                f"{_nvenc(self.codec_out, self.bitrate, self.gop, self.low_latency)} ! "
                f"{_sink(self.protocol, self.out_url, self.codec_out)}")

    def run_transcode(self):
        import gi
        from gi.repository import Gst, GLib
        pstr = self._transcode_pipeline()
        print("[restream] pipeline:\n ", pstr)
        self._pipeline = Gst.parse_launch(pstr)
        self._pipeline.set_state(Gst.State.PLAYING)
        bus = self._pipeline.get_bus()
        loop = GLib.MainLoop()

        def on_msg(_b, msg):
            t = msg.type
            if t == Gst.MessageType.EOS:
                print("[restream] EOS"); loop.quit()
            elif t == Gst.MessageType.ERROR:
                err, dbg = msg.parse_error()
                print(f"[restream] ERRO: {err.message}\n{dbg}"); loop.quit()
        bus.add_signal_watch(); bus.connect("message", on_msg)
        try:
            loop.run()
        except KeyboardInterrupt:
            pass
        finally:
            self._pipeline.set_state(Gst.State.NULL)

    # ---- modo infer: decode -> YOLO -> appsrc -> nvenc -> sink ----
    def run_infer(self):
        import gi, time
        import numpy as np
        from gi.repository import Gst
        from ultralytics import YOLO

        src = GstStream(self.source, engine="gpu")
        src.open()
        # 1º frame para saber dimensões
        first = src.read()
        if first is None:
            raise RuntimeError("fonte não entregou frames")
        h, w = first.array.shape[:2]
        fps = int(src.fps or 30)
        model = YOLO(self.model_name)

        out = (f"appsrc name=src is-live=true do-timestamp=true format=time "
               f"caps=video/x-raw,format=BGR,width={w},height={h},framerate={fps}/1 ! "
               f"queue max-size-buffers=4 leaky=downstream ! videoconvert ! "
               f"{_nvenc(self.codec_out, self.bitrate, self.gop, self.low_latency)} ! "
               f"{_sink(self.protocol, self.out_url, self.codec_out)}")
        print("[restream/infer] saída:\n ", out)
        pipe = Gst.parse_launch(out)
        appsrc = pipe.get_by_name("src")
        pipe.set_state(Gst.State.PLAYING)

        def push(arr):
            buf = Gst.Buffer.new_allocate(None, arr.nbytes, None)
            buf.fill(0, arr.tobytes())
            appsrc.emit("push-buffer", buf)

        frame = first
        try:
            while frame is not None:
                res = model.predict(frame.array, device=self.device,
                                    imgsz=self.imgsz, verbose=False)
                push(np.ascontiguousarray(res[0].plot()))
                frame = src.read()
        except KeyboardInterrupt:
            pass
        finally:
            appsrc.emit("end-of-stream")
            pipe.set_state(Gst.State.NULL)
            src.close()

    def run(self):
        if self.infer:
            self.run_infer()
        else:
            self.run_transcode()

"""Benchmark de transcode (decode -> encode).

Este e' o cenario onde a GPU domina: NVDEC -> NVENC roda 100% em engines
dedicadas de video, enquanto a CPU precisa de libav (decode) + x264 (encode).
Aqui usamos "o poder maximo da GPU" de verdade (decoder + encoder).

Mede via gst-launch (pipeline puro, sem overhead de Python) e amostra
o uso de NVDEC/NVENC com o GpuMonitor.
"""
from __future__ import annotations

import shutil
import subprocess
import time
from dataclasses import dataclass, field
from typing import List, Optional

from .monitor import GpuMonitor


@dataclass
class TranscodeResult:
    label: str
    engine: str            # "gpu" | "cpu"
    frames: int
    wall_s: float
    fps: float
    dec_util_mean: float = 0.0
    enc_util_mean: float = 0.0
    gpu_util_mean: float = 0.0
    pipeline: str = ""
    ok: bool = True
    error: str = ""


def _count_frames(path: str) -> int:
    try:
        import gi
        gi.require_version("Gst", "1.0")
        gi.require_version("GstPbutils", "1.0")
        from gi.repository import Gst, GstPbutils
        if not Gst.is_initialized():
            Gst.init(None)
        disc = GstPbutils.Discoverer.new(5 * Gst.SECOND)
        import os
        uri = "file://" + os.path.abspath(path)
        info = disc.discover_uri(uri)
        for s in info.get_video_streams():
            dur = info.get_duration() / Gst.SECOND
            num, den = s.get_framerate_num(), s.get_framerate_denom()
            if dur and num and den:
                return int(round(dur * num / den))
    except Exception:
        pass
    return -1


def _run_pipeline_timed(pipeline_args: List[str], gpu_index: int = 0):
    mon = GpuMonitor(120, gpu_index).start()
    t0 = time.perf_counter()
    proc = subprocess.run(["gst-launch-1.0", "-q", *pipeline_args],
                          stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                          text=True)
    wall = time.perf_counter() - t0
    summary = mon.stop()
    return wall, summary, proc.returncode, proc.stderr


def transcode_benchmark(source: str, *, out_codec: str = "h264",
                        bitrate_kbps: int = 8000, x264_preset: str = "medium",
                        gpu_index: int = 0, verbose: bool = True
                        ) -> List[TranscodeResult]:
    """Compara transcode GPU (NVDEC->NVENC) vs CPU (libav->x264)."""
    if shutil.which("gst-launch-1.0") is None:
        raise RuntimeError("gst-launch-1.0 nao encontrado")
    frames = _count_frames(source)
    base = ["filesrc", f"location={source}", "!", "qtdemux", "!", "h264parse"]
    gpu_enc = "nvh264enc" if out_codec == "h264" else "nvh265enc"

    gpu_pipe = base + ["!", "nvh264dec", "!", gpu_enc, f"bitrate={bitrate_kbps}",
                       "!", "h264parse", "!", "fakesink"]
    cpu_pipe = base + ["!", "avdec_h264", "!", "videoconvert", "!", "x264enc",
                       f"bitrate={bitrate_kbps}", f"speed-preset={x264_preset}",
                       "!", "fakesink"]

    results = []
    for label, engine, pipe in (("gpu nvdec->nvenc", "gpu", gpu_pipe),
                                ("cpu libav->x264", "cpu", cpu_pipe)):
        if verbose:
            print(f"  -> transcode {label} ...", flush=True)
        wall, summ, rc, err = _run_pipeline_timed(pipe, gpu_index)
        r = TranscodeResult(
            label=label, engine=engine,
            frames=frames if frames > 0 else 0,
            wall_s=wall, fps=(frames / wall if frames > 0 and wall else 0.0),
            dec_util_mean=summ.dec_util_mean, enc_util_mean=summ.enc_util_mean,
            gpu_util_mean=summ.gpu_util_mean,
            pipeline=" ".join(pipe), ok=(rc == 0),
            error="" if rc == 0 else (err or "").strip()[-200:],
        )
        results.append(r)
        if verbose:
            print(f"     {r.fps:7.1f} fps em {r.wall_s:.2f}s | "
                  f"dec {r.dec_util_mean:.0f}% enc {r.enc_util_mean:.0f}%"
                  + ("" if r.ok else f" | FALHOU: {r.error}"), flush=True)

    if verbose and len(results) == 2 and results[1].fps:
        spd = results[0].fps / results[1].fps if results[1].fps else 0
        print(f"\n  GPU e' {spd:.2f}x o CPU (x264 {x264_preset}) neste transcode.")
    return results

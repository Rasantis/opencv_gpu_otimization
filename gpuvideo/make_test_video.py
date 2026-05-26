"""Gera videos de teste codificando na GPU (NVENC)."""
from __future__ import annotations

import shutil
import subprocess

_RES = {
    "720p": (1280, 720),
    "1080p": (1920, 1080),
    "1440p": (2560, 1440),
    "4k": (3840, 2160),
}

_ENC = {"h264": ("nvh264enc", "h264parse"), "h265": ("nvh265enc", "h265parse")}


def make_test_video(path: str, *, resolution: str = "1080p", frames: int = 600,
                    fps: int = 30, codec: str = "h264", bitrate_kbps: int = 8000,
                    pattern: str = "smpte") -> str:
    """Cria um .mp4 codificado via NVENC. Retorna o caminho."""
    if shutil.which("gst-launch-1.0") is None:
        raise RuntimeError("gst-launch-1.0 nao encontrado")
    w, h = _RES.get(resolution, (1920, 1080))
    enc, parse = _ENC.get(codec, _ENC["h264"])
    pipeline = [
        "gst-launch-1.0", "-e",
        "videotestsrc", f"num-buffers={frames}", f"pattern={pattern}", "!",
        f"video/x-raw,width={w},height={h},framerate={fps}/1", "!",
        "timeoverlay", "!",
        enc, f"bitrate={bitrate_kbps}", "!",
        parse, "!", "mp4mux", "!", f"filesink", f"location={path}",
    ]
    subprocess.run(pipeline, check=True, stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL)
    return path


if __name__ == "__main__":  # pragma: no cover
    import argparse
    ap = argparse.ArgumentParser(description="Gera video de teste via NVENC")
    ap.add_argument("output")
    ap.add_argument("--res", default="1080p", choices=list(_RES))
    ap.add_argument("--frames", type=int, default=600)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--codec", default="h264", choices=list(_ENC))
    a = ap.parse_args()
    out = make_test_video(a.output, resolution=a.res, frames=a.frames,
                          fps=a.fps, codec=a.codec)
    print("gerado:", out)

"""CLI do gpuvideo.

Exemplos:
    python -m gpuvideo gen test_1080p.mp4 --res 1080p --frames 600
    python -m gpuvideo bench test_1080p.mp4 --frames 300 --op light
    python -m gpuvideo scale test_1080p.mp4 --n 8 --mode gstreamer-gpu --frames 200
"""
from __future__ import annotations

import argparse
import sys

from . import MODES


def _cmd_gen(a):
    from .make_test_video import make_test_video
    out = make_test_video(a.output, resolution=a.res, frames=a.frames,
                          fps=a.fps, codec=a.codec)
    print("video gerado:", out)


def _cmd_bench(a):
    from .benchmark import Benchmark
    modes = a.modes.split(",") if a.modes else list(MODES)
    bench = Benchmark(a.source, frames=a.frames, warmup=a.warmup, op=a.op,
                      modes=modes, gpu_index=a.gpu)
    print(f"Fonte: {a.source} | modos: {modes}")
    bench.compare()
    bench.report()
    if a.json:
        bench.to_json(a.json)
        print("json salvo em", a.json)
    if a.csv:
        bench.to_csv(a.csv)
        print("csv salvo em", a.csv)


def _cmd_transcode(a):
    from .transcode import transcode_benchmark
    print(f"Transcode: {a.source} | x264 preset={a.preset} | bitrate={a.bitrate}k")
    transcode_benchmark(a.source, bitrate_kbps=a.bitrate, x264_preset=a.preset,
                        gpu_index=a.gpu)


def _cmd_scale(a):
    from .multistream import MultiStream
    ms = MultiStream.replicate(a.source, a.n, mode=a.mode)
    print(f"Escala: {a.n}x [{a.mode}] -> {a.source}")
    res = ms.run(max_frames=a.frames)
    print(res)
    for i, fps in enumerate(res.per_stream_fps):
        print(f"   stream {i}: {fps:6.1f} fps")
    for e in res.errors:
        print("   ERRO:", e)


def main(argv=None):
    p = argparse.ArgumentParser(prog="gpuvideo")
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("gen", help="gera video de teste via NVENC")
    g.add_argument("output")
    g.add_argument("--res", default="1080p")
    g.add_argument("--frames", type=int, default=600)
    g.add_argument("--fps", type=int, default=30)
    g.add_argument("--codec", default="h264")
    g.set_defaults(func=_cmd_gen)

    b = sub.add_parser("bench", help="benchmark comparativo")
    b.add_argument("source")
    b.add_argument("--frames", type=int, default=300)
    b.add_argument("--warmup", type=int, default=20)
    b.add_argument("--op", default="light", choices=["none", "light", "heavy"])
    b.add_argument("--modes", default="", help="csv (ex.: opencv-cpu,gstreamer-gpu)")
    b.add_argument("--gpu", type=int, default=0)
    b.add_argument("--json", default="")
    b.add_argument("--csv", default="")
    b.set_defaults(func=_cmd_bench)

    t = sub.add_parser("transcode", help="benchmark de transcode GPU vs CPU")
    t.add_argument("source")
    t.add_argument("--bitrate", type=int, default=8000)
    t.add_argument("--preset", default="medium")
    t.add_argument("--gpu", type=int, default=0)
    t.set_defaults(func=_cmd_transcode)

    s = sub.add_parser("scale", help="teste de escala (N streams)")
    s.add_argument("source")
    s.add_argument("--n", type=int, default=4)
    s.add_argument("--mode", default="gstreamer-gpu", choices=list(MODES))
    s.add_argument("--frames", type=int, default=200)
    s.set_defaults(func=_cmd_scale)

    args = p.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())

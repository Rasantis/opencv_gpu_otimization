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


def _cmd_doctor(a):
    from .env import doctor
    return doctor()


def _cmd_analytics(a):
    from .pipeline import run_cameras
    run_cameras(a.config, seconds=a.seconds, force_display=a.display)


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


def _cmd_restream(a):
    from .restream import Restreamer
    r = Restreamer(a.source, a.out, codec_out=a.codec, bitrate_kbps=a.bitrate,
                   gop=a.gop, low_latency=not a.no_low_latency, protocol=a.protocol,
                   infer=a.infer, model=a.model, imgsz=a.imgsz, device=a.device)
    print(f"Restream: {a.source} -> {a.out} | {a.protocol}/{a.codec} "
          f"{a.bitrate}k | infer={a.infer}")
    r.run()


def _cmd_export(a):
    from .optimize import export_engine
    path = export_engine(a.model, fp16=not a.int8, int8=a.int8, imgsz=a.imgsz,
                         batch=a.batch, workspace=a.workspace, data=a.data or None,
                         device=a.device)
    print("OK ->", path)


def _cmd_optbench(a):
    from .optimize import benchmark_precision
    print(f"Benchmark de precisão: {a.model} em {a.source}")
    benchmark_precision(a.model, a.source, frames=a.frames, imgsz=a.imgsz,
                        device=a.device)


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

    d = sub.add_parser("doctor", help="checa o ambiente (deps, GStreamer, GPU, YOLO)")
    d.set_defaults(func=_cmd_doctor)

    an = sub.add_parser("analytics", help="roda múltiplas câmeras de um YAML (1 solução cada)")
    an.add_argument("config", help="arquivo YAML de câmeras (ver examples/cameras.yaml)")
    an.add_argument("--seconds", type=float, default=None, help="duração (padrão: até o fim)")
    an.add_argument("--display", action="store_true", help="força janela p/ todas as câmeras")
    an.set_defaults(func=_cmd_analytics)

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

    rs = sub.add_parser("restream", help="restream baixa-latência (-> RTMP/SRT, ex.: MediaMTX)")
    rs.add_argument("source", help="rtsp://, http://, arquivo, indice de camera ou 'test'")
    rs.add_argument("out", help="URL de saida (ex.: rtmp://mediamtx:1935/cam1)")
    rs.add_argument("--codec", default="h264", choices=["h264", "h265"])
    rs.add_argument("--bitrate", type=int, default=4000)
    rs.add_argument("--gop", type=int, default=30)
    rs.add_argument("--protocol", default="rtmp", choices=["rtmp", "srt"])
    rs.add_argument("--no-low-latency", action="store_true")
    rs.add_argument("--infer", action="store_true", help="desenha deteccoes YOLO11")
    rs.add_argument("--model", default="yolo11n.pt")
    rs.add_argument("--imgsz", type=int, default=640)
    rs.add_argument("--device", default="0")
    rs.set_defaults(func=_cmd_restream)

    ex = sub.add_parser("export", help="exporta YOLO -> engine TensorRT (FP16/INT8, deploy)")
    ex.add_argument("model", help="modelo .pt (ex.: yolo11n.pt)")
    ex.add_argument("--int8", action="store_true", help="quantiza INT8 (precisa --data)")
    ex.add_argument("--data", default="", help="YAML de calibração p/ INT8 (ex.: coco128.yaml)")
    ex.add_argument("--imgsz", type=int, default=640)
    ex.add_argument("--batch", type=int, default=1)
    ex.add_argument("--workspace", type=int, default=4, help="GiB de workspace do builder")
    ex.add_argument("--device", default="0")
    ex.set_defaults(func=_cmd_export)

    ob = sub.add_parser("optbench", help="compara FPS de inferência FP32 vs FP16")
    ob.add_argument("model", help="modelo (.pt ou .engine)")
    ob.add_argument("source", help="fonte de vídeo p/ alimentar a inferência")
    ob.add_argument("--frames", type=int, default=150)
    ob.add_argument("--imgsz", type=int, default=640)
    ob.add_argument("--device", default="0")
    ob.set_defaults(func=_cmd_optbench)

    s = sub.add_parser("scale", help="teste de escala (N streams)")
    s.add_argument("source")
    s.add_argument("--n", type=int, default=4)
    s.add_argument("--mode", default="gstreamer-gpu", choices=list(MODES))
    s.add_argument("--frames", type=int, default=200)
    s.set_defaults(func=_cmd_scale)

    args = p.parse_args(argv)
    rc = args.func(args)
    return rc if isinstance(rc, int) else 0


if __name__ == "__main__":
    sys.exit(main())

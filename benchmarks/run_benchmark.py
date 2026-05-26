#!/usr/bin/env python3
"""Suite completa de benchmark: captura, transcode e escala.

Gera os videos de teste se nao existirem e roda os tres cenarios,
salvando os resultados em ./results/.

    python3 run_benchmark.py            # suite padrao
    python3 run_benchmark.py --quick    # versao rapida
"""
from __future__ import annotations

import argparse
import json
import os

from gpuvideo.make_test_video import make_test_video
from gpuvideo.benchmark import Benchmark
from gpuvideo.transcode import transcode_benchmark
from gpuvideo.multistream import MultiStream

# Raiz do repo (este script vive em benchmarks/); vídeos e results/ ficam lá.
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(HERE, "results")


def ensure_video(path, **kw):
    if not os.path.exists(path):
        print(f"  gerando {path} (NVENC)...", flush=True)
        make_test_video(path, **kw)
    return path


def banner(txt):
    print("\n" + "#" * 70 + f"\n# {txt}\n" + "#" * 70)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    os.makedirs(RESULTS, exist_ok=True)

    frames = 150 if a.quick else 250
    heavy = ensure_video(os.path.join(HERE, "heavy_1080p.mp4"),
                         resolution="1080p", frames=300, pattern="snow",
                         bitrate_kbps=25000)
    uhd = ensure_video(os.path.join(HERE, "bench_4k.mp4"),
                       resolution="4k", frames=300)

    summary = {}

    banner("1) CAPTURA -> numpy (1080p, conteudo pesado, op=light)")
    b1 = Benchmark(heavy, frames=frames, warmup=20, op="light").compare()
    b1.report()
    b1.to_json(os.path.join(RESULTS, "capture_1080p.json"))
    summary["capture_1080p"] = [r.to_dict() for r in b1.results]

    banner("2) CAPTURA -> numpy (4K, op=none p/ isolar decode)")
    b2 = Benchmark(uhd, frames=min(frames, 150), warmup=10, op="none").compare()
    b2.report()
    b2.to_json(os.path.join(RESULTS, "capture_4k.json"))
    summary["capture_4k"] = [r.to_dict() for r in b2.results]

    banner("3) TRANSCODE (decode->encode H.264): GPU NVDEC+NVENC vs CPU x264")
    tr = transcode_benchmark(heavy, bitrate_kbps=8000, x264_preset="medium")
    summary["transcode"] = [r.__dict__ for r in tr]

    banner("4) ESCALA (N streams 1080p simultaneos)")
    for n in ([2, 4] if a.quick else [2, 4, 8]):
        for mode in ("opencv-cpu", "gstreamer-gpu"):
            res = MultiStream.replicate(heavy, n, mode=mode).run(max_frames=120)
            print(" ", res)
            summary.setdefault("scale", []).append(res.__dict__)

    with open(os.path.join(RESULTS, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nResultados salvos em {RESULTS}/")


if __name__ == "__main__":
    main()

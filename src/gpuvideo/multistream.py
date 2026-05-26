"""MultiStream - executa varias streams em paralelo (teste de escala).

Cada stream roda em sua propria thread (decode em C/GStreamer libera o GIL),
medindo vazao agregada e por-stream. Util pra responder "quantas cameras
1080p essa GPU aguenta neste backend?".

    from gpuvideo import MultiStream
    res = MultiStream.replicate("video.mp4", n=8, mode="gstreamer-gpu").run()
    print(res.aggregate_fps, res.per_stream_fps)
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

from . import make_stream
from .frame import Frame


@dataclass
class ScaleResult:
    mode: str
    n_streams: int
    total_frames: int
    wall_s: float
    aggregate_fps: float
    per_stream_fps: List[float] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        ok = "OK" if not self.errors else f"{len(self.errors)} ERRO(S)"
        return (f"[{self.mode}] {self.n_streams} streams | "
                f"{self.total_frames} frames em {self.wall_s:.2f}s | "
                f"agregado {self.aggregate_fps:.1f} fps | "
                f"~{(self.aggregate_fps / max(self.n_streams,1)):.1f} fps/stream | {ok}")


class MultiStream:
    def __init__(self, sources: Sequence, *, mode: str = "gstreamer-gpu",
                 stream_kwargs: Optional[dict] = None):
        self.sources = list(sources)
        self.mode = mode
        self.stream_kwargs = stream_kwargs or {}

    @classmethod
    def replicate(cls, source, n: int, *, mode: str = "gstreamer-gpu",
                  stream_kwargs: Optional[dict] = None) -> "MultiStream":
        return cls([source] * n, mode=mode, stream_kwargs=stream_kwargs)

    def run(self, *, max_frames: Optional[int] = None,
            op: Optional[Callable[[Frame], None]] = None,
            warmup: int = 5) -> ScaleResult:
        n = len(self.sources)
        counts = [0] * n
        durations = [0.0] * n
        errors: List[str] = []
        barrier = threading.Barrier(n)

        def worker(i: int, src):
            nonlocal errors
            try:
                stream = make_stream(self.mode, src, stream_id=str(i),
                                     **self.stream_kwargs)
                stream.open()
                # Warmup pra estabilizar antes de cronometrar.
                for _ in range(warmup):
                    if stream.read() is None:
                        break
                barrier.wait()  # todas comecam a medir juntas
                t0 = time.perf_counter()
                c = 0
                while True:
                    frame = stream.read()
                    if frame is None:
                        break
                    if op is not None:
                        op(frame)
                    c += 1
                    if max_frames is not None and c >= max_frames:
                        break
                durations[i] = time.perf_counter() - t0
                counts[i] = c
                stream.close()
            except Exception as e:  # noqa: BLE001
                errors.append(f"stream {i}: {type(e).__name__}: {e}")
                try:
                    barrier.wait(timeout=0.1)
                except Exception:
                    pass

        threads = [threading.Thread(target=worker, args=(i, s), daemon=True)
                   for i, s in enumerate(self.sources)]
        wall0 = time.perf_counter()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        wall = time.perf_counter() - wall0

        total = sum(counts)
        per_stream = [c / d if d else 0.0 for c, d in zip(counts, durations)]
        agg = total / max((max(durations) if durations else wall), 1e-9)
        return ScaleResult(
            mode=self.mode, n_streams=n, total_frames=total, wall_s=wall,
            aggregate_fps=agg, per_stream_fps=per_stream, errors=errors,
        )

"""Interface comum a todos os backends de captura.

Qualquer backend (OpenCV, GStreamer, ...) implementa ``open``/``read``/``close``
e ganha de graca o protocolo de context manager e de iterador:

    with SomeStream("video.mp4") as s:
        for frame in s:
            usar(frame.array)
"""
from __future__ import annotations

import abc
import time
from typing import Optional

from .frame import Frame


class BaseStream(abc.ABC):
    """Contrato minimo de uma fonte de video."""

    backend_name: str = "base"

    def __init__(self, source, *, stream_id: str = "0") -> None:
        self.source = source
        self.stream_id = stream_id
        # Preenchidos em open()/primeiro frame quando conhecidos.
        self.width: int = 0
        self.height: int = 0
        self.fps: float = 0.0
        self.frame_count: int = -1   # -1 = desconhecido (live/stream)
        self._index: int = 0
        self._opened: bool = False

    # ----- API que cada backend implementa -----
    @abc.abstractmethod
    def open(self) -> "BaseStream":
        ...

    @abc.abstractmethod
    def read(self) -> Optional[Frame]:
        """Retorna o proximo Frame, ou ``None`` no fim do stream (EOS)."""
        ...

    @abc.abstractmethod
    def close(self) -> None:
        ...

    # ----- conveniencias compartilhadas -----
    @property
    def is_open(self) -> bool:
        return self._opened

    def __enter__(self) -> "BaseStream":
        if not self._opened:
            self.open()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __iter__(self):
        if not self._opened:
            self.open()
        return self

    def __next__(self) -> Frame:
        frame = self.read()
        if frame is None:
            raise StopIteration
        return frame

    def _now(self) -> float:
        return time.perf_counter()

    def __repr__(self) -> str:  # pragma: no cover - cosmetico
        return (
            f"<{type(self).__name__} backend={self.backend_name} "
            f"source={self.source!r} {self.width}x{self.height}@{self.fps:.1f} "
            f"open={self._opened}>"
        )

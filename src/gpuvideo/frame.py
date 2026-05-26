"""Estrutura de um frame entregue ao usuário."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(slots=True)
class Frame:
    """Um frame decodificado, pronto para uso.

    O atributo ``array`` e' sempre um ``numpy.ndarray`` proprio (copia),
    seguro para usar depois que o proximo frame for lido.
    """

    array: np.ndarray            # HxWxC (BGR por padrao), uint8
    index: int                   # numero do frame na stream (0-based)
    width: int
    height: int
    pts_ns: Optional[int] = None         # presentation timestamp (ns), se houver
    capture_monotonic: float = 0.0       # time.perf_counter() na entrega ao app
    stream_id: str = "0"

    @property
    def shape(self) -> tuple:
        return self.array.shape

    def __repr__(self) -> str:  # pragma: no cover - cosmetico
        return (
            f"Frame(#{self.index} {self.width}x{self.height} "
            f"{self.array.dtype} shape={self.array.shape} stream={self.stream_id!r})"
        )

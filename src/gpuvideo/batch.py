"""Inferência compartilhada em batch para múltiplas câmeras.

Por que: um modelo YOLO em batch=1 é *overhead-bound* (pré/pós-processo, NMS,
lançamento de kernel dominam) e subutiliza a GPU. Empilhar frames de várias
câmeras num único forward amortiza esse overhead e satura a GPU — é como
DeepStream/Triton operam em escala.

Desenho:
  - UM modelo por grupo de câmeras com a mesma assinatura (modelo/imgsz/device/
    classes). Cada câmera chama `infer(frame)` (bloqueante).
  - Uma thread "batcher" drena a fila, acumula até `max_batch` frames (ou espera
    `max_wait_ms`), roda UM `model.predict(lote)` e devolve cada Result à câmera.
  - A DETECÇÃO é batched (a parte cara na GPU). O TRACKING continua por câmera
    (estado por stream) via `make_tracker()` — ver analytics.VideoAnalytics.

    det = BatchInference("yolo11n.pt", classes=["person"], max_batch=8)
    res = det.infer(frame)          # Result da ultralytics (sem IDs)
    ...
    det.close()
"""
from __future__ import annotations

import queue
import threading
import time
from typing import List, Optional, Sequence


class _Slot:
    """Caixa de resposta de um frame: a câmera espera no event, o batcher preenche."""
    __slots__ = ("event", "result", "error")

    def __init__(self):
        self.event = threading.Event()
        self.result = None
        self.error = None


def make_tracker(name: str = "bytetrack.yaml", frame_rate: int = 30):
    """Cria um BYTETracker/BOTSORT por câmera, igual ao que o ultralytics faz
    internamente em on_predict_start (estado de tracking é por stream)."""
    from ultralytics.trackers import BYTETracker, BOTSORT
    from ultralytics.utils import IterableSimpleNamespace, YAML
    from ultralytics.utils.checks import check_yaml
    cfg = IterableSimpleNamespace(**YAML.load(check_yaml(name)))
    if cfg.tracker_type not in ("bytetrack", "botsort"):
        raise ValueError(f"tracker não suportado: {cfg.tracker_type}")
    Tracker = BOTSORT if cfg.tracker_type == "botsort" else BYTETracker
    try:
        return Tracker(cfg, frame_rate=frame_rate)
    except TypeError:
        return Tracker(cfg)            # versões sem frame_rate na assinatura


class BatchInference:
    """Servidor de inferência compartilhado (um modelo, forward em batch)."""

    def __init__(self, model: str = "yolo11n.pt", device: str = "0",
                 imgsz: int = 640, half="auto", classes: Optional[Sequence[str]] = None,
                 max_batch: int = 8, max_wait_ms: float = 8.0, tensor_mode: bool = False):
        from ultralytics import YOLO
        self.model = YOLO(model)
        self.device = device
        self.imgsz = imgsz
        # tensor_mode: câmeras submetem tensores CUDA já pré-processados (keep-on-GPU);
        # o batcher faz torch.cat (agrupando por shape) -> 1 forward, sem PCIe.
        self.tensor_mode = tensor_mode
        if half == "auto":
            half = str(device).strip().lower() not in ("cpu", "-1")
        self.half = bool(half)
        self.names = self.model.names
        self.class_ids = None
        if classes:
            inv = {v: k for k, v in self.model.names.items()}
            self.class_ids = [inv[c] for c in classes if c in inv]
        self.max_batch = max(1, int(max_batch))
        self.max_wait = max(0.0, max_wait_ms / 1000.0)
        self._q: "queue.Queue" = queue.Queue()
        self._running = True
        self._n_batches = 0
        self._n_frames = 0
        self._thread = threading.Thread(target=self._loop, name="batcher", daemon=True)
        self._thread.start()

    def infer(self, frame):
        """Enfileira o frame e bloqueia até o resultado do batch. Devolve Result."""
        slot = _Slot()
        self._q.put((frame, slot))
        slot.event.wait()
        if slot.error is not None:
            raise slot.error
        return slot.result

    @property
    def avg_batch(self) -> float:
        return self._n_frames / self._n_batches if self._n_batches else 0.0

    def _drain(self):
        """Pega o 1º item (bloqueante curto) e acumula a janela de batch."""
        try:
            first = self._q.get(timeout=0.1)
        except queue.Empty:
            return None
        batch = [first]
        deadline = time.time() + self.max_wait
        while len(batch) < self.max_batch:
            timeout = deadline - time.time()
            if timeout <= 0:
                break
            try:
                batch.append(self._q.get(timeout=timeout))
            except queue.Empty:
                break
        return batch

    def _forward(self, items):
        """Roda 1 forward p/ um conjunto de itens já homogêneo e devolve os Results."""
        if self.tensor_mode:
            import torch
            x = torch.cat([it[0] for it in items], 0)      # (N,3,H,W) na GPU
        else:
            x = [it[0] for it in items]                    # lista de numpy
        return self.model.predict(x, imgsz=self.imgsz, device=self.device,
                                  half=self.half, classes=self.class_ids, verbose=False)

    def _loop(self):
        while self._running:
            batch = self._drain()
            if not batch:
                continue
            # tensor_mode: agrupa por shape (câmeras de mesmo aspecto batcham juntas,
            # sem distorção); numpy: um grupo só (o predict lida com tamanhos variados).
            groups = {}
            if self.tensor_mode:
                for it in batch:
                    groups.setdefault(tuple(it[0].shape), []).append(it)
            else:
                groups[None] = batch
            for items in groups.values():
                try:
                    results = self._forward(items)
                    for (f, slot), res in zip(items, results):
                        slot.result = res
                        slot.event.set()
                    self._n_batches += 1
                    self._n_frames += len(items)
                except Exception as e:  # noqa: BLE001 — não derruba o batcher
                    for f, slot in items:
                        slot.error = e
                        slot.event.set()

    def close(self):
        self._running = False
        if self._thread.is_alive():
            self._thread.join(timeout=2)

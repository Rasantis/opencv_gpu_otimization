"""Analytics de vídeo: tracking (ByteTrack) + soluções plugáveis.

Soluções construídas sobre rastreamento de IDs (YOLO11 + ByteTrack):
contagem por linha, tempo de permanência (dwell), mapa de calor e invasão.
Todas usam coordenadas NORMALIZADAS (0-1) → independentes de resolução.

    from gpuvideo.analytics import VideoAnalytics, LineCounter, IntrusionZone, Heatmap, DwellZone

    va = (VideoAnalytics("rtsp://cam", model="yolo11n.pt", classes=["person"])
          .add(LineCounter([(0,0.6),(1,0.6)], name="porta"))
          .add(IntrusionZone([(0.05,0.1),(0.4,0.1),(0.4,0.6),(0.05,0.6)], name="restrito"))
          .add(Heatmap()))
    for frame, tracks, events in va.run():
        for e in events: print(e)        # frame = numpy anotado pronto pra exibir/restream
"""
from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np


_PALETTE = [(0, 255, 255), (255, 0, 255), (0, 255, 0), (255, 128, 0),
            (0, 128, 255), (255, 255, 0), (128, 0, 255), (0, 255, 128),
            (255, 0, 128), (60, 220, 255)]


def _color_for_id(i: int):
    """Cor vívida e distinta por track_id (BGR)."""
    return _PALETTE[i % len(_PALETTE)]


class _LazyCv2:
    """Carrega o cv2 só no 1º uso (desenho) — a lógica das soluções é numpy puro,
    então o módulo importa e roda sem cv2 (testes, headless, event-only)."""
    _mod = None

    def __getattr__(self, name):
        if _LazyCv2._mod is None:
            from .env import require_cv2
            _LazyCv2._mod = require_cv2()
        return getattr(_LazyCv2._mod, name)


cv2 = _LazyCv2()

Point = Tuple[float, float]


# --------------------------------------------------------------------------
@dataclass
class Track:
    id: int
    cls: int
    name: str
    bbox: Tuple[float, float, float, float]  # x1,y1,x2,y2 (px)
    conf: float

    @property
    def center(self):  # centróide (px)
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    @property
    def foot(self):  # ponto no chão (px) — base do bbox
        x1, _, x2, y2 = self.bbox
        return ((x1 + x2) / 2, y2)


@dataclass
class Event:
    type: str                # count_in | count_out | intrusion | dwell_exit | dwell_alert
    solution: str
    t: float
    track_id: int = -1
    label: str = ""
    data: dict = field(default_factory=dict)

    def __str__(self):
        extra = f" {self.data}" if self.data else ""
        return f"[{self.type}] {self.solution} track={self.track_id} {self.label}{extra}"


# ---- geometria (em coords normalizadas) ----
def _point_in_poly(pt: Point, poly: Sequence[Point]) -> bool:
    x, y = pt
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi):
            inside = not inside
        j = i
    return inside


def _norm_poly_px(poly, w, h):
    return np.array([(int(px * w), int(py * h)) for px, py in poly], np.int32)


# --------------------------------------------------------------------------
class Solution:
    """Base: recebe os tracks do frame e devolve eventos; desenha sua camada."""
    name = "solution"

    def process(self, tracks: List[Track], w: int, h: int, t: float) -> List[Event]:
        return []

    def draw(self, img) -> None:
        pass


def _label(img, txt, org, color):
    cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)


class LineCounter(Solution):
    """Conta tracks que cruzam uma linha (direção → in/out)."""

    def __init__(self, line: Sequence[Point], name="linha", classes: Optional[Sequence[str]] = None):
        self.p1, self.p2 = line
        self.name = name
        self.classes = set(classes) if classes else None
        self._side = {}
        self.count_in = 0
        self.count_out = 0

    def _cross(self, pt):
        (ax, ay), (bx, by) = self.p1, self.p2
        return (bx - ax) * (pt[1] - ay) - (by - ay) * (pt[0] - ax)

    def process(self, tracks, w, h, t):
        events = []
        for tk in tracks:
            if self.classes and tk.name not in self.classes:
                continue
            pt = (tk.foot[0] / w, tk.foot[1] / h)
            s = self._cross(pt)
            prev = self._side.get(tk.id)
            if prev is not None and prev * s < 0:
                if s > 0:
                    self.count_in += 1
                    events.append(Event("count_in", self.name, t, tk.id, tk.name))
                else:
                    self.count_out += 1
                    events.append(Event("count_out", self.name, t, tk.id, tk.name))
            self._side[tk.id] = s
        return events

    def draw(self, img):
        h, w = img.shape[:2]
        p1 = (int(self.p1[0] * w), int(self.p1[1] * h))
        p2 = (int(self.p2[0] * w), int(self.p2[1] * h))
        cv2.line(img, p1, p2, (0, 230, 230), 3, cv2.LINE_AA)
        _label(img, f"{self.name}  IN {self.count_in}  OUT {self.count_out}",
               (p1[0], max(20, p1[1] - 10)), (0, 230, 230))


class DwellZone(Solution):
    """Tempo de permanência: tempo de cada track dentro de uma zona."""

    def __init__(self, zone: Sequence[Point], name="zona", classes=None, alert_s: Optional[float] = None):
        self.poly = zone
        self.name = name
        self.classes = set(classes) if classes else None
        self.alert_s = alert_s
        self._enter = {}
        self._alerted = set()
        self._inside_px = []  # (foot_px, dwell) p/ desenhar

    def process(self, tracks, w, h, t):
        events = []
        inside = set()
        self._inside_px = []
        for tk in tracks:
            if self.classes and tk.name not in self.classes:
                continue
            if _point_in_poly((tk.foot[0] / w, tk.foot[1] / h), self.poly):
                inside.add(tk.id)
                self._enter.setdefault(tk.id, t)
                dwell = t - self._enter[tk.id]
                self._inside_px.append((tk.foot, dwell))
                if self.alert_s and dwell >= self.alert_s and tk.id not in self._alerted:
                    self._alerted.add(tk.id)
                    events.append(Event("dwell_alert", self.name, t, tk.id, tk.name,
                                        {"dwell_s": round(dwell, 1)}))
        for tid in list(self._enter):
            if tid not in inside:
                dwell = t - self._enter.pop(tid)
                self._alerted.discard(tid)
                events.append(Event("dwell_exit", self.name, t, tid, data={"dwell_s": round(dwell, 1)}))
        return events

    def draw(self, img):
        h, w = img.shape[:2]
        cv2.polylines(img, [_norm_poly_px(self.poly, w, h)], True, (255, 180, 60), 2, cv2.LINE_AA)
        p0 = _norm_poly_px(self.poly, w, h)[0]
        _label(img, f"{self.name}  dentro:{len(self._inside_px)}", (p0[0], max(20, p0[1] - 8)), (255, 180, 60))
        for (fx, fy), dwell in self._inside_px:
            _label(img, f"{dwell:.0f}s", (int(fx), int(fy)), (255, 180, 60))


class Heatmap(Solution):
    """Mapa de calor acumulado das posições dos tracks (com decaimento)."""

    def __init__(self, name="heatmap", classes=None, decay=0.97, alpha=0.45, radius=10):
        self.name = name
        self.classes = set(classes) if classes else None
        self.decay = decay
        self.alpha = alpha
        self.radius = radius
        self._acc = None

    def process(self, tracks, w, h, t):
        gh, gw = h // 4, w // 4
        if self._acc is None or self._acc.shape != (gh, gw):
            self._acc = np.zeros((gh, gw), np.float32)
        self._acc *= self.decay
        for tk in tracks:
            if self.classes and tk.name not in self.classes:
                continue
            cx, cy = tk.center
            x, y = int(cx / 4), int(cy / 4)
            if 0 <= y < gh and 0 <= x < gw:
                cv2.circle(self._acc, (x, y), self.radius, 1.0, -1)
        return []

    def draw(self, img):
        if self._acc is None:
            return
        m = self._acc.max()
        if m < 1e-6:
            return
        hm = cv2.resize((np.clip(self._acc / m, 0, 1) * 255).astype(np.uint8),
                        (img.shape[1], img.shape[0]))
        cm = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
        mask = hm > 25
        img[mask] = (img[mask] * (1 - self.alpha) + cm[mask] * self.alpha).astype(np.uint8)


class IntrusionZone(Solution):
    """Detecção de invasão: alerta quando um track entra numa zona proibida."""

    def __init__(self, zone: Sequence[Point], name="invasao", classes=("person",),
                 on_alert: Optional[Callable[[Event], None]] = None):
        self.poly = zone
        self.name = name
        self.classes = set(classes) if classes else None
        self.on_alert = on_alert
        self._active = set()
        self.breached = False

    def process(self, tracks, w, h, t):
        events = []
        inside = set()
        for tk in tracks:
            if self.classes and tk.name not in self.classes:
                continue
            if _point_in_poly((tk.foot[0] / w, tk.foot[1] / h), self.poly):
                inside.add(tk.id)
                if tk.id not in self._active:
                    self._active.add(tk.id)
                    ev = Event("intrusion", self.name, t, tk.id, tk.name)
                    events.append(ev)
                    if self.on_alert:
                        try:
                            self.on_alert(ev)
                        except Exception:
                            pass
        self._active &= inside
        self.breached = bool(inside)
        return events

    def draw(self, img):
        h, w = img.shape[:2]
        poly = _norm_poly_px(self.poly, w, h)
        color = (0, 0, 255) if self.breached else (0, 140, 255)
        overlay = img.copy()
        cv2.fillPoly(overlay, [poly], color)
        cv2.addWeighted(overlay, 0.18, img, 0.82, 0, img)
        cv2.polylines(img, [poly], True, color, 2, cv2.LINE_AA)
        txt = f"{self.name}: INVASAO!" if self.breached else self.name
        _label(img, txt, (poly[0][0], max(20, poly[0][1] - 8)), color)


# --------------------------------------------------------------------------
class VideoAnalytics:
    """Decodifica na GPU, roda YOLO11+ByteTrack e aplica as soluções."""

    def __init__(self, source, model="yolo11n.pt", classes: Optional[Sequence[str]] = None,
                 imgsz=640, device="0", tracker="bytetrack.yaml",
                 backend="gstreamer", engine="gpu", solutions=None,
                 annotate=True, proc_max_side: Optional[int] = None,
                 decoder="auto", trails=True, trail_len=30,
                 reconnect=True, loop=False, half="auto", detector=None,
                 infer_fps=None, keep_on_gpu=False):
        self.source = source
        self.model_name = model
        self.classes = list(classes) if classes else None
        self.imgsz = imgsz
        self.device = device
        self.tracker = tracker
        self.backend = backend
        self.engine = engine
        self.solutions: List[Solution] = list(solutions) if solutions else []
        # annotate=False -> só eventos (produção/bus), sem desenhar = FPS cheio.
        # proc_max_side -> redimensiona o frame antes de processar (coords são
        #   normalizadas, então não muda o resultado das soluções; só acelera).
        self.annotate = annotate
        self.proc_max_side = proc_max_side
        # decoder: "auto" usa cudacodec (NVDEC nativo + resize na GPU, baixa só o
        # frame pequeno -> evita contenção PCIe/CPU do download 4K); senão GStreamer.
        self.decoder = decoder
        # rastro (trail) do ByteTrack: histórico de posições por track_id.
        self.trails = trails
        self.trail_len = trail_len
        self._trail = defaultdict(lambda: deque(maxlen=trail_len))
        self._trail_age = {}
        # robustez: reconectar fontes ao vivo que caem; loop p/ arquivos.
        self.reconnect = reconnect
        self.loop = loop
        # FP16: ~1.3-2x de throughput na GPU, custo ~zero de acurácia. "auto" =
        # liga em CUDA (device numérico ou "cuda"); CPU sempre FP32.
        if half == "auto":
            half = str(device).strip().lower() not in ("cpu", "-1")
        self.half = bool(half)
        # detector compartilhado (batch.BatchInference): se informado, a DETECÇÃO
        # roda em batch nesse servidor e o tracking fica aqui (por câmera).
        self.detector = detector
        # desacopla decode de inferência: decodifica a fps cheio (vídeo suave) mas
        # infere a infer_fps (ex.: 5). Contagem/dwell/heatmap não precisam de 30fps
        # -> corte direto de custo. Frames pulados reusam os últimos tracks.
        self.infer_fps = infer_fps
        # keep-on-GPU: NVDEC->GpuMat->tensor torch (D2D, sem PCIe)->inferência.
        # Só as caixas descem; o frame nunca toca a CPU (em modo event-only).
        # Squash p/ imgsz² preserva coords normalizadas das soluções. Standalone
        # (detector=None); cudacodec só (arquivo/RTSP suportado pelo NVDEC).
        self.keep_on_gpu = keep_on_gpu
        # contadores p/ observabilidade (o runner faz a ponte p/ Prometheus).
        self.stats = {"inferences": 0, "infer_ms": 0.0, "reconnects": 0}

    def add(self, solution: Solution) -> "VideoAnalytics":
        self.solutions.append(solution)
        return self

    def _update_trails(self, tracks):
        seen = set()
        for tk in tracks:
            self._trail[tk.id].append(tuple(map(int, tk.center)))
            self._trail_age[tk.id] = 0
            seen.add(tk.id)
        for tid in list(self._trail_age):
            if tid not in seen:
                self._trail_age[tid] += 1
                if self._trail_age[tid] > self.trail_len:
                    self._trail.pop(tid, None)
                    self._trail_age.pop(tid, None)

    def _draw_trails(self, img):
        for tid, pts in self._trail.items():
            if len(pts) < 2 or self._trail_age.get(tid, 99) > 2:
                continue
            col = _color_for_id(tid)
            cv2.polylines(img, [np.array(pts, np.int32).reshape(-1, 1, 2)],
                          False, col, 3, cv2.LINE_AA)
            cv2.circle(img, pts[-1], 5, col, -1)
            cv2.circle(img, pts[-1], 5, (255, 255, 255), 1, cv2.LINE_AA)

    def _extract(self, r, names) -> List[Track]:
        tracks = []
        b = r.boxes
        if b is None or b.id is None:
            return tracks
        xyxy = b.xyxy.cpu().numpy()
        ids = b.id.cpu().numpy().astype(int)
        cls = b.cls.cpu().numpy().astype(int)
        conf = b.conf.cpu().numpy()
        for i in range(len(ids)):
            tracks.append(Track(int(ids[i]), int(cls[i]), names[int(cls[i])],
                                tuple(map(float, xyxy[i])), float(conf[i])))
        return tracks

    def _extract_from_array(self, arr, names) -> List[Track]:
        """tracks do BYTETracker.update: linhas [x1,y1,x2,y2,id,conf,cls,idx]."""
        tracks = []
        for row in arr:
            x1, y1, x2, y2, tid, conf, cls = row[:7]
            cls = int(cls)
            tracks.append(Track(int(tid), cls, names[cls],
                                (float(x1), float(y1), float(x2), float(y2)), float(conf)))
        return tracks

    def _draw_tracks(self, frame, tracks):
        """Desenha bbox + id por track (modo batched, em que o Result não traz IDs)."""
        for tk in tracks:
            x1, y1, x2, y2 = map(int, tk.bbox)
            c = _color_for_id(tk.id)
            cv2.rectangle(frame, (x1, y1), (x2, y2), c, 2)
            cv2.putText(frame, f"{tk.name} {tk.id}", (x1, max(0, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 2)

    def run(self, should_stop: Optional[Callable[[], bool]] = None):
        """Generator: (frame_anotado, tracks, events) por frame.

        Resiliente: fontes ao vivo que caem são reconectadas (backoff); arquivos
        terminam (ou repetem com loop=True). `should_stop()` permite parada limpa.
        """
        from . import VideoStream
        from .cudacodec import cudacodec_available, CudaCodecStream
        from .pipelines import classify_source, SourceKind
        model = tracker = class_ids = None
        kog = self.keep_on_gpu and self.detector is None
        if self.detector is None:
            from ultralytics import YOLO
            model = YOLO(self.model_name)
            names = model.names
            if self.classes:
                inv = {v: k for k, v in model.names.items()}
                class_ids = [inv[c] for c in self.classes if c in inv]
            if kog:                          # keep-on-GPU usa predict(tensor)+tracker local
                from .batch import make_tracker
                tracker = make_tracker(self.tracker)
        else:
            # detector compartilhado: detecção em batch lá fora, tracking aqui.
            from .batch import make_tracker
            tracker = make_tracker(self.tracker)
            names = self.detector.names

        proc = self.proc_max_side
        src = str(self.source)
        live = classify_source(self.source) in (SourceKind.RTSP, SourceKind.RTMP,
                                                SourceKind.HTTP, SourceKind.CAMERA)
        cuda_ok = cudacodec_available() and "://" not in src and not src.startswith("/dev/")
        use_cuda = self.decoder in ("auto", "cuda") and cuda_ok
        if kog and not cuda_ok:                  # pediu keep-on-GPU mas sem NVDEC -> cai p/ padrão
            print(f"[{src}] keep_on_gpu requer cudacodec; usando caminho padrão.", flush=True)
            kog = False
        if kog:
            use_cuda = True
        imgsz = self.imgsz

        def _make():
            if kog:                              # NVDEC -> GpuMat RGB (fica na GPU)
                def _gpu_prep(g, _cv2):
                    rgb = _cv2.cuda.cvtColor(g, _cv2.COLOR_BGRA2RGB) if g.channels() == 4 \
                        else _cv2.cuda.cvtColor(g, _cv2.COLOR_BGR2RGB)
                    ow, oh = g.size()           # preserva aspecto; lados múltiplos de 32
                    th = max(32, int(round(imgsz * oh / ow / 32)) * 32)
                    return _cv2.cuda.resize(rgb, (imgsz, th))
                return CudaCodecStream(self.source, gpu_op=_gpu_prep, color="BGRA",
                                       as_gpumat=True)
            if use_cuda:
                def _gpu_resize(g, _cv2):
                    if not proc:
                        return g
                    w, h = g.size()
                    if max(w, h) <= proc:
                        return g
                    sc = proc / max(w, h)
                    return _cv2.cuda.resize(g, (int(w * sc), int(h * sc)))
                return CudaCodecStream(self.source, gpu_op=_gpu_resize, color="BGR")
            return VideoStream(self.source, backend=self.backend, engine=self.engine)

        def _stopped():
            return bool(should_stop and should_stop())

        def _sleep(sec):
            end = time.time() + sec
            while time.time() < end and not _stopped():
                time.sleep(0.2)

        def _open_retry():
            backoff = 1.0
            while not _stopped():
                try:
                    s = _make()
                    s.open()
                    return s
                except Exception as e:  # noqa: BLE001
                    if not (self.reconnect and live):
                        raise
                    print(f"[{src}] open falhou ({type(e).__name__}); retry em {backoff:.0f}s",
                          flush=True)
                    _sleep(backoff)
                    backoff = min(backoff * 2, 10)
            return None

        # estado do desacople decode/inferência
        infer_period = (1.0 / self.infer_fps) if self.infer_fps else 0.0
        last_infer = 0.0
        last_tracks: List[Track] = []

        stream = _open_retry()
        backoff = 1.0
        try:
            while stream is not None and not _stopped():
                try:
                    frame = stream.read()
                except Exception:  # noqa: BLE001
                    frame = None
                if frame is None:
                    if not live and not self.loop:
                        break                       # arquivo terminou
                    try:
                        stream.close()
                    except Exception:
                        pass
                    if live and self.reconnect:
                        self.stats["reconnects"] += 1
                        print(f"[{src}] stream caiu; reconectando em {backoff:.0f}s", flush=True)
                        _sleep(backoff)
                        backoff = min(backoff * 2, 10)
                    elif not live and self.loop:
                        pass                        # re-abre o arquivo (loop)
                    else:
                        break
                    stream = _open_retry()
                    continue
                backoff = 1.0
                t = time.time()
                if kog:                             # keep-on-GPU: frame fica na GPU
                    w, h = frame.gpu.size()         # (imgsz, th) já redimensionado na GPU
                else:
                    arr = frame.array
                    if arr.ndim == 2:
                        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
                    if proc and not use_cuda:       # CPU resize só se não foi na GPU
                        H, W = arr.shape[:2]
                        if max(H, W) > proc:
                            s = proc / max(H, W)
                            arr = cv2.resize(arr, (int(W * s), int(H * s)))
                    h, w = arr.shape[:2]
                do_infer = infer_period <= 0 or (t - last_infer) >= infer_period
                if do_infer:
                    last_infer = t
                    _t_inf = time.perf_counter()
                    if kog:                             # GpuMat -> tensor (D2D) -> predict
                        from .gpu_bridge import preprocess
                        tens = preprocess(frame.gpu, half=self.half)
                        r = model.predict(tens, imgsz=self.imgsz, classes=class_ids,
                                          device=self.device, half=self.half, verbose=False)[0]
                        tracks = self._extract_from_array(tracker.update(r.boxes.cpu().numpy(), None), names)
                        base = None
                        if self.annotate:              # só aqui o frame (pequeno) desce
                            base = cv2.cvtColor(frame.gpu.download(), cv2.COLOR_RGB2BGR)
                            self._draw_tracks(base, tracks)
                    elif self.detector is None:         # standalone: model.track
                        r = model.track(arr, persist=True, tracker=self.tracker,
                                        classes=class_ids, imgsz=self.imgsz,
                                        device=self.device, half=self.half, verbose=False)[0]
                        tracks = self._extract(r, model.names)
                        base = r.plot() if self.annotate else None
                    else:                               # batched: detecção compartilhada + tracking local
                        res = self.detector.infer(arr)
                        upd = tracker.update(res.boxes.cpu().numpy(), res.orig_img)
                        tracks = self._extract_from_array(upd, names)
                        base = arr.copy() if self.annotate else None
                        if base is not None:
                            self._draw_tracks(base, tracks)
                    self.stats["infer_ms"] = (time.perf_counter() - _t_inf) * 1000
                    self.stats["inferences"] += 1
                    last_tracks = tracks
                    if self.trails:
                        self._update_trails(tracks)
                    events = []
                    for sol in self.solutions:
                        events.extend(sol.process(tracks, w, h, t))
                else:
                    # frame pulado: analytics só na taxa de inferência; reusa tracks.
                    tracks, events = last_tracks, []
                    if not self.annotate:
                        continue                        # event-only: nada a entregar
                    base = (cv2.cvtColor(frame.gpu.download(), cv2.COLOR_RGB2BGR)
                            if kog else arr.copy())
                    self._draw_tracks(base, tracks)
                annotated = None
                if self.annotate:
                    annotated = base
                    if self.trails:
                        self._draw_trails(annotated)
                    for sol in self.solutions:
                        sol.draw(annotated)
                yield annotated, tracks, events
        finally:
            if stream is not None:
                try:
                    stream.close()
                except Exception:
                    pass

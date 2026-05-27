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
                 decoder="auto", trails=True, trail_len=30):
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

    def run(self):
        """Generator: (frame_anotado, tracks, events) por frame."""
        from ultralytics import YOLO
        from . import VideoStream
        from .cudacodec import cudacodec_available, CudaCodecStream
        model = YOLO(self.model_name)
        class_ids = None
        if self.classes:
            inv = {v: k for k, v in model.names.items()}
            class_ids = [inv[c] for c in self.classes if c in inv]

        # Decoder: cudacodec (resize na GPU, baixa frame pequeno) p/ arquivos.
        proc = self.proc_max_side
        use_cuda = (self.decoder in ("auto", "cuda") and cudacodec_available()
                    and "://" not in str(self.source) and not str(self.source).startswith("/dev/"))
        if use_cuda:
            def _gpu_resize(g, _cv2):
                if not proc:
                    return g
                w, h = g.size()
                if max(w, h) <= proc:
                    return g
                sc = proc / max(w, h)
                return _cv2.cuda.resize(g, (int(w * sc), int(h * sc)))
            stream = CudaCodecStream(self.source, gpu_op=_gpu_resize, color="BGR")
        else:
            stream = VideoStream(self.source, backend=self.backend, engine=self.engine)
        stream.open()
        try:
            for frame in stream:
                arr = frame.array
                if arr.ndim == 2:
                    arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
                if proc and not use_cuda:  # CPU resize só se não foi na GPU
                    H, W = arr.shape[:2]
                    if max(H, W) > proc:
                        s = proc / max(H, W)
                        arr = cv2.resize(arr, (int(W * s), int(H * s)))
                h, w = arr.shape[:2]
                t = time.time()
                res = model.track(arr, persist=True, tracker=self.tracker,
                                  classes=class_ids, imgsz=self.imgsz,
                                  device=self.device, verbose=False)
                r = res[0]
                tracks = self._extract(r, model.names)
                if self.trails:
                    self._update_trails(tracks)
                events = []
                for sol in self.solutions:
                    events.extend(sol.process(tracks, w, h, t))
                # annotate só quando alguém vai ver (custa caro em alta res)
                annotated = None
                if self.annotate:
                    annotated = r.plot()
                    if self.trails:
                        self._draw_trails(annotated)
                    for sol in self.solutions:
                        sol.draw(annotated)
                yield annotated, tracks, events
        finally:
            stream.close()

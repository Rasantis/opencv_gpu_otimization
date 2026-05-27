"""Config declarativa por câmera (YAML) + runner multi-câmera.

Cada câmera = uma entrada no YAML com sua fonte, modelo e UMA solução. O runner
sobe uma thread por câmera (decode na GPU + YOLO + solução) e agrega os eventos.

    gpuvideo analytics examples/cameras.yaml --seconds 20

YAML:
    defaults: {model: yolo11n.pt, classes: [person], proc_max_side: 960, annotate: false}
    cameras:
      - id: entrada
        source: rtsp://cam1/stream
        solution: {type: counting, line: [[0,0.55],[1,0.55]], name: fluxo}
      - id: restrito
        source: rtsp://cam2/stream
        model: yolo11s.pt
        solution: {type: intrusion, zone: [[0,0.25],[0.4,0.25],[0.4,1],[0,1]]}
"""
from __future__ import annotations

import os
import threading
import time
from typing import Callable, Dict, List, Optional

from .analytics import (VideoAnalytics, LineCounter, DwellZone, Heatmap, IntrusionZone)


def build_solution(spec: dict):
    """dict do YAML -> instância de Solution."""
    t = spec["type"]
    name = spec.get("name", t)
    classes = spec.get("classes")
    if t in ("counting", "line_counting", "line"):
        return LineCounter(spec["line"], name=name, classes=classes)
    if t in ("dwell", "dwell_time", "permanencia"):
        return DwellZone(spec["zone"], name=name, classes=classes, alert_s=spec.get("alert_s"))
    if t in ("heatmap", "mapa_calor"):
        return Heatmap(name=name, classes=classes)
    if t in ("intrusion", "invasao", "intrusion_zone"):
        return IntrusionZone(spec["zone"], name=name, classes=classes or ["person"])
    raise ValueError(f"tipo de solução desconhecido: {t!r}")


def load_config(path: str) -> List[dict]:
    """Lê o YAML e devolve a lista de câmeras (defaults aplicados)."""
    try:
        import yaml
    except ModuleNotFoundError as e:
        raise RuntimeError("PyYAML não instalado: pip install pyyaml") from e
    with open(path) as f:
        cfg = yaml.safe_load(f)
    defaults = cfg.get("defaults", {})
    cams = []
    for c in cfg.get("cameras", []):
        merged = {**defaults, **c}
        if "id" not in merged or "source" not in merged:
            raise ValueError(f"câmera sem 'id' ou 'source': {c}")
        cams.append(merged)
    return cams


def _truthy(v):
    return bool(v) and v not in ("false", "no", "0", 0)


def build_analytics(cam: dict, force_display=False, detector=None) -> VideoAnalytics:
    # precisa anotar o frame se for exibir, gravar ou streamar essa câmera
    visual = (force_display or _truthy(cam.get("display")) or
              cam.get("record") or cam.get("stream"))
    va = VideoAnalytics(
        cam["source"], model=cam.get("model", "yolo11n.pt"),
        classes=cam.get("classes"), imgsz=int(cam.get("imgsz", 640)),
        device=str(cam.get("device", "0")),
        annotate=bool(visual) or _truthy(cam.get("annotate")),
        proc_max_side=cam.get("proc_max_side", 960),
        decoder=cam.get("decoder", "auto"),
        trails=bool(cam.get("trails", True)),
        reconnect=bool(cam.get("reconnect", True)),
        loop=bool(cam.get("loop", False)),
        half=cam.get("half", "auto"),
        detector=detector,
    )
    if cam.get("solution"):
        va.add(build_solution(cam["solution"]))
    for s in cam.get("solutions", []):     # múltiplas, se quiser
        va.add(build_solution(s))
    return va


def build_detectors(cams: List[dict]) -> Dict:
    """Para as câmeras com batch ligado, cria UM BatchInference por assinatura de
    inferência (modelo/imgsz/device/classes). Anota cam['_detector'] e devolve os
    detectores (p/ fechar no fim). Câmeras com mesma assinatura compartilham GPU."""
    from .batch import BatchInference
    detectors: Dict[tuple, object] = {}
    for c in cams:
        if not _truthy(c.get("batch")):
            continue
        key = (c.get("model", "yolo11n.pt"), int(c.get("imgsz", 640)),
               str(c.get("device", "0")), tuple(c.get("classes") or []))
        det = detectors.get(key)
        if det is None:
            det = detectors[key] = BatchInference(
                model=key[0], imgsz=key[1], device=key[2],
                classes=list(key[3]) or None, half=c.get("half", "auto"),
                max_batch=int(c.get("batch_size", 8)),
                max_wait_ms=float(c.get("batch_wait_ms", 8)))
        c["_detector"] = det
    if detectors:
        print(f"Batching: {len(detectors)} modelo(s) compartilhado(s) p/ "
              f"{sum(1 for c in cams if c.get('_detector'))} câmera(s)")
    return detectors


def run_cameras(config_path: str, on_event: Optional[Callable] = None,
                seconds: Optional[float] = None, force_display=False,
                base_dir: Optional[str] = None) -> Dict:
    """Roda as câmeras do YAML em paralelo. Cada câmera pode (via YAML):
       display: true        -> janela em tempo real
       record:  saida.mp4    -> grava o vídeo anotado
       stream:  rtmp://...   -> restreama (WebRTC/HLS via MediaMTX)
       events:  {format: csv|xlsx|jsonl, path: ...}  -> salva os eventos
    """
    from .sinks import make_sink
    cams = load_config(config_path)
    base = base_dir or os.path.dirname(os.path.abspath(config_path)) or "."
    print(f"Subindo {len(cams)} câmera(s): " +
          ", ".join(f"{c['id']}({(c.get('solution') or {}).get('type','?')})" for c in cams))
    detectors = build_detectors(cams)
    stats: Dict[str, dict] = {}
    latest: Dict[str, object] = {}        # último frame p/ exibir (GUI no main thread)
    lock = threading.Lock()
    stop = threading.Event()
    display_cams = [c["id"] for c in cams if force_display or _truthy(c.get("display"))]

    def worker(cam):
        cid = cam["id"]
        n = ev = 0
        sink = make_sink(cam.get("events"), cid, base)
        streamer = writer = None
        show = force_display or _truthy(cam.get("display"))
        t0 = time.perf_counter()
        try:
            va = build_analytics(cam, force_display, detector=cam.get("_detector"))
            if cam.get("stream"):
                from .restream import FrameStreamer
                url = cam["stream"] if isinstance(cam["stream"], str) else f"rtmp://localhost:1935/{cid}"
                streamer = FrameStreamer(url, fps=int(cam.get("fps", 25)))
            for frame, _tracks, events in va.run(should_stop=stop.is_set):
                for e in events:
                    ev += 1
                    sink.write(cid, e)
                    if on_event:
                        on_event(cid, e)
                if frame is not None:
                    if cam.get("record"):
                        import cv2
                        if writer is None:
                            h, w = frame.shape[:2]
                            writer = cv2.VideoWriter(os.path.join(base, str(cam["record"])),
                                                     cv2.VideoWriter_fourcc(*"mp4v"), 25, (w, h))
                        writer.write(frame)
                    if streamer is not None:
                        streamer.push(frame)
                    if show:
                        with lock:
                            latest[cid] = frame
                n += 1
                if stop.is_set() or (seconds and time.perf_counter() - t0 > seconds):
                    break
        except Exception as ex:  # noqa: BLE001
            print(f"[{cid}] ERRO: {type(ex).__name__}: {ex}", flush=True)
        finally:
            sink.close()
            if streamer:
                streamer.close()
            if writer:
                writer.release()
        dt = time.perf_counter() - t0
        stats[cid] = {"frames": n, "events": ev, "fps": n / dt if dt else 0}

    # Ctrl+C -> parada limpa (só funciona no main thread).
    import signal
    try:
        signal.signal(signal.SIGINT, lambda *_: stop.set())
    except (ValueError, OSError):
        pass

    threads = [threading.Thread(target=worker, args=(c,), daemon=True) for c in cams]
    for th in threads:
        th.start()

    # Exibição: TODA a GUI no main thread (cv2 não é thread-safe).
    if display_cams and (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        import cv2
        try:
            while any(t.is_alive() for t in threads) and not stop.is_set():
                for cid in display_cams:
                    with lock:
                        f = latest.get(cid)
                    if f is not None:
                        cv2.imshow(f"camera: {cid}", f)
                if (cv2.waitKey(15) & 0xFF) in (ord("q"), 27):
                    stop.set()
        finally:
            cv2.destroyAllWindows()

    for th in threads:
        th.join()
    for det in detectors.values():
        try:
            print(f"  [batch] batch médio efetivo: {det.avg_batch:.1f} frames/forward")
            det.close()
        except Exception:
            pass

    print("\n=== resumo por câmera ===")
    for cid, s in stats.items():
        print(f"  {cid:<14} {s['fps']:6.1f} fps | {s['frames']:5d} frames | {s['events']} eventos")
    return stats

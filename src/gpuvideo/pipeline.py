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


def build_analytics(cam: dict) -> VideoAnalytics:
    va = VideoAnalytics(
        cam["source"], model=cam.get("model", "yolo11n.pt"),
        classes=cam.get("classes"), imgsz=int(cam.get("imgsz", 640)),
        device=str(cam.get("device", "0")),
        annotate=bool(cam.get("annotate", False)),
        proc_max_side=cam.get("proc_max_side", 960),
        decoder=cam.get("decoder", "auto"),
    )
    if cam.get("solution"):
        va.add(build_solution(cam["solution"]))
    for s in cam.get("solutions", []):     # múltiplas, se quiser
        va.add(build_solution(s))
    return va


def run_cameras(config_path: str, on_event: Optional[Callable] = None,
                seconds: Optional[float] = None, verbose: bool = True) -> Dict:
    """Roda todas as câmeras do YAML em paralelo (1 thread cada)."""
    cams = load_config(config_path)
    print(f"Subindo {len(cams)} câmera(s): " +
          ", ".join(f"{c['id']}({(c.get('solution') or {}).get('type','?')})" for c in cams))
    stats: Dict[str, dict] = {}
    stop_at = time.time() + seconds if seconds else None

    def worker(cam):
        cid = cam["id"]
        n = ev = 0
        t0 = time.perf_counter()
        try:
            va = build_analytics(cam)
            for _frame, _tracks, events in va.run():
                for e in events:
                    ev += 1
                    if on_event:
                        on_event(cid, e)
                    elif verbose:
                        print(f"[{cid}] {e}", flush=True)
                n += 1
                if stop_at and time.time() > stop_at:
                    break
        except Exception as ex:  # noqa: BLE001
            print(f"[{cid}] ERRO: {type(ex).__name__}: {ex}", flush=True)
        dt = time.perf_counter() - t0
        stats[cid] = {"frames": n, "events": ev, "fps": n / dt if dt else 0}

    threads = [threading.Thread(target=worker, args=(c,), daemon=True) for c in cams]
    for th in threads:
        th.start()
    for th in threads:
        th.join()

    print("\n=== resumo por câmera ===")
    for cid, s in stats.items():
        print(f"  {cid:<14} {s['fps']:6.1f} fps | {s['frames']:5d} frames | {s['events']} eventos")
    return stats

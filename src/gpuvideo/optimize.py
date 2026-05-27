"""Otimização de inferência: export TensorRT (FP16/INT8) e benchmark de precisão.

Por que importa em escala: a inferência é o gargalo de custo por câmera. FP16
~dobra o throughput na GPU com perda de acurácia desprezível; um engine TensorRT
(FP16/INT8) vai além fundindo kernels e especializando p/ a GPU alvo.

    # FP16 (rápido, sem deps extras) já é o padrão em CUDA no VideoAnalytics.
    # Engine TensorRT (deploy): precisa do pacote `tensorrt` instalado.
    gpuvideo export yolo11n.pt --fp16            # -> yolo11n.engine
    gpuvideo export yolo11n.pt --int8 --data coco128.yaml

O VideoAnalytics carrega o .engine de forma transparente: basta apontar
`model: yolo11n.engine` no YAML (ultralytics resolve o backend pelo sufixo).
"""
from __future__ import annotations

import time
from typing import List, Optional


def export_engine(model: str, fp16: bool = True, int8: bool = False,
                  imgsz: int = 640, batch: int = 1, workspace: int = 4,
                  data: Optional[str] = None, device: str = "0") -> str:
    """Exporta um modelo YOLO p/ engine TensorRT. Devolve o caminho do .engine.

    int8 exige `data` (YAML de calibração); cai p/ fp16 se não informado.
    """
    try:
        from ultralytics import YOLO
    except ModuleNotFoundError as e:
        raise RuntimeError("ultralytics não instalado: pip install 'gpuvideo[yolo]'") from e
    try:
        import tensorrt  # noqa: F401
    except ModuleNotFoundError as e:
        raise RuntimeError(
            "pacote 'tensorrt' não encontrado — necessário p/ exportar engine.\n"
            "  Instale no ambiente de deploy (imagem com TensorRT) e rode de novo.\n"
            "  Enquanto isso, FP16 já está ligado por padrão em CUDA (sem TensorRT)."
        ) from e
    if int8 and not data:
        print("[export] INT8 sem --data (calibração); usando FP16.", flush=True)
        int8, fp16 = False, True
    kw = dict(format="engine", imgsz=imgsz, half=fp16 and not int8,
              int8=int8, batch=batch, workspace=workspace, device=device, verbose=False)
    if data:
        kw["data"] = data
    path = YOLO(model).export(**kw)
    print(f"[export] engine salvo em: {path}", flush=True)
    return str(path)


def benchmark_precision(model: str, source, frames: int = 150, imgsz: int = 640,
                        device: str = "0", warmup: int = 20,
                        precisions: Optional[List[str]] = None) -> dict:
    """Mede FPS de inferência por precisão (fp32/fp16) na mesma fonte.

    Retorna {precisao: fps}. Útil p/ justificar FP16 em produção.
    """
    from ultralytics import YOLO
    from . import VideoStream
    precisions = precisions or ["fp32", "fp16"]
    # pré-carrega N frames p/ tirar I/O da conta (mede só a inferência).
    buf = []
    s = VideoStream(source, engine="gpu")
    s.open()
    try:
        for f in s:
            arr = f.array
            buf.append(arr)
            if len(buf) >= frames + warmup:
                break
    finally:
        s.close()
    if not buf:
        raise RuntimeError(f"sem frames de {source}")

    # reduz à escala de processamento (como o proc_max_side em produção): assim
    # o letterbox interno é trivial e cronometramos a INFERÊNCIA, não o resize 4K.
    from .env import require_cv2
    cv2 = require_cv2()
    rs = []
    for arr in buf:
        H, W = arr.shape[:2]
        if max(H, W) > imgsz:
            sc = imgsz / max(H, W)
            arr = cv2.resize(arr, (int(W * sc), int(H * sc)))
        rs.append(arr)
    buf = rs

    out = {}
    mdl = YOLO(model)
    for prec in precisions:
        half = prec == "fp16"
        for arr in buf[:warmup]:
            mdl.predict(arr, imgsz=imgsz, device=device, half=half, verbose=False)
        t0 = time.perf_counter()
        infer_ms = 0.0
        for arr in buf[warmup:]:
            r = mdl.predict(arr, imgsz=imgsz, device=device, half=half, verbose=False)
            infer_ms += r[0].speed.get("inference", 0.0)  # só o forward na GPU
        dt = time.perf_counter() - t0
        n = len(buf) - warmup
        wall = n / dt if dt else 0.0
        gpu = (n * 1000.0 / infer_ms) if infer_ms else 0.0
        out[prec] = {"wall_fps": wall, "gpu_fps": gpu}
        print(f"  {prec:>5}: forward {gpu:7.1f} fps | pipeline {wall:6.1f} fps "
              f"({n} frames)", flush=True)
    if "fp32" in out and "fp16" in out and out["fp32"]["gpu_fps"]:
        sp = out["fp16"]["gpu_fps"] / out["fp32"]["gpu_fps"]
        print(f"  -> FP16 é {sp:.2f}x o FP32 no forward da GPU", flush=True)
    return out

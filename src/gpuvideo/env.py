"""Checagem de ambiente e erros amigáveis para deps de sistema.

cv2 (OpenCV) e gi (PyGObject/GStreamer) NÃO vêm por pip — precisam de pacotes
do sistema. Estas funções dão um erro claro com o comando de instalação, e o
`check_environment()` (CLI: `gpuvideo doctor`) mostra o que está disponível.
"""
from __future__ import annotations

import shutil
import subprocess

APT_GSTREAMER = ("sudo apt install -y python3-gi gir1.2-gst-plugins-base-1.0 "
                 "gir1.2-gst-plugins-bad-1.0 gstreamer1.0-plugins-bad gstreamer1.0-libav")
APT_OPENCV = "sudo apt install -y python3-opencv"


class DependencyError(RuntimeError):
    pass


def require_gi():
    """Importa gi/Gst ou levanta erro acionável."""
    try:
        import gi
        return gi
    except ModuleNotFoundError as e:
        raise DependencyError(
            "GStreamer/PyGObject (módulo 'gi') não encontrado — não vem por pip.\n"
            f"  Instale: {APT_GSTREAMER}\n"
            "  (veja docs/INSTALL.md · cheque com: gpuvideo doctor)"
        ) from e


def require_cv2():
    """Importa cv2 ou levanta erro acionável."""
    try:
        import cv2
        return cv2
    except ModuleNotFoundError as e:
        raise DependencyError(
            "OpenCV (cv2) não encontrado — não vem por pip nesta configuração.\n"
            f"  Instale: {APT_OPENCV}\n"
            "  (GPU nativo/cudacodec: docs/BUILD_CUDA.md · cheque: gpuvideo doctor)"
        ) from e


def _gst_plugins(names):
    """Quais elementos GStreamer existem (sem precisar de gi)."""
    out = {}
    gi_ok = False
    try:
        import gi
        gi.require_version("Gst", "1.0")
        from gi.repository import Gst
        if not Gst.is_initialized():
            Gst.init(None)
        gi_ok = True
        for n in names:
            out[n] = Gst.ElementFactory.find(n) is not None
    except Exception:
        for n in names:
            out[n] = False
    return gi_ok, out


def check_environment() -> dict:
    """Levanta o estado do ambiente para o `gpuvideo doctor`."""
    r = {}

    try:
        import numpy
        r["numpy"] = {"ok": True, "version": numpy.__version__}
    except Exception as e:
        r["numpy"] = {"ok": False, "error": type(e).__name__}

    # OpenCV + flags do build
    try:
        import cv2
        bi = cv2.getBuildInformation()
        def flag(k):
            for ln in bi.splitlines():
                s = ln.strip()
                if s.startswith(k):
                    return "YES" in s
            return False
        cuda_devs = 0
        try:
            cuda_devs = cv2.cuda.getCudaEnabledDeviceCount()
        except Exception:
            pass
        r["opencv"] = {"ok": True, "version": cv2.__version__,
                       "ffmpeg": flag("FFMPEG"), "gstreamer": flag("GStreamer"),
                       "cuda_devices": cuda_devs, "cudacodec": hasattr(cv2, "cudacodec")}
    except Exception as e:
        r["opencv"] = {"ok": False, "error": type(e).__name__, "fix": APT_OPENCV}

    # GStreamer + plugins-chave
    gi_ok, plugins = _gst_plugins([
        "nvh264dec", "nvh264enc", "nvh265dec", "nvh265enc",
        "rtspsrc", "rtmp2sink", "appsink"])
    r["gstreamer"] = {"ok": gi_ok, "plugins": plugins,
                      **({} if gi_ok else {"fix": APT_GSTREAMER})}

    # GPU NVIDIA
    if shutil.which("nvidia-smi"):
        try:
            o = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version",
                                "--format=csv,noheader"], capture_output=True,
                               text=True, timeout=5).stdout.strip()
            r["gpu"] = {"ok": bool(o), "info": o}
        except Exception:
            r["gpu"] = {"ok": False}
    else:
        r["gpu"] = {"ok": False, "info": "nvidia-smi não encontrado"}

    # YOLO (opcional)
    try:
        import torch
        r["torch"] = {"ok": True, "version": torch.__version__,
                      "cuda": torch.cuda.is_available()}
    except Exception:
        r["torch"] = {"ok": False, "fix": "pip install gpuvideo[yolo]"}
    try:
        import ultralytics
        r["ultralytics"] = {"ok": True, "version": ultralytics.__version__}
    except Exception:
        r["ultralytics"] = {"ok": False, "fix": "pip install gpuvideo[yolo]"}

    return r


def doctor() -> int:
    """Imprime um relatório legível do ambiente. Retorna 0 se o básico está OK."""
    r = check_environment()
    tick = lambda b: "\033[32m✓\033[0m" if b else "\033[31m✗\033[0m"
    print("gpuvideo doctor — estado do ambiente\n" + "-" * 44)

    print(f"{tick(r['numpy']['ok'])} numpy            "
          f"{r['numpy'].get('version','')}")

    o = r["opencv"]
    if o["ok"]:
        print(f"{tick(True)} OpenCV (cv2)     {o['version']}  "
              f"[FFmpeg:{'sim' if o['ffmpeg'] else 'NAO'} "
              f"GStreamer:{'sim' if o['gstreamer'] else 'NAO'} "
              f"CUDA:{o['cuda_devices']} cudacodec:{'sim' if o['cudacodec'] else 'NAO'}]")
    else:
        print(f"{tick(False)} OpenCV (cv2)     ausente → {o.get('fix','')}")

    g = r["gstreamer"]
    if g["ok"]:
        nv = [k for k, v in g["plugins"].items() if k.startswith("nv") and v]
        miss = [k for k, v in g["plugins"].items() if not v]
        print(f"{tick(True)} GStreamer (gi)   NVDEC/NVENC: {', '.join(nv) or 'nenhum'}"
              + (f"  | faltando: {', '.join(miss)}" if miss else ""))
    else:
        print(f"{tick(False)} GStreamer (gi)   ausente → {g.get('fix','')}")

    print(f"{tick(r['gpu']['ok'])} GPU NVIDIA       {r['gpu'].get('info','')}")
    print(f"{tick(r['torch']['ok'])} torch (YOLO)     "
          + (f"{r['torch']['version']} cuda={r['torch']['cuda']}" if r['torch']['ok']
             else f"opcional → {r['torch'].get('fix','')}"))
    print(f"{tick(r['ultralytics']['ok'])} ultralytics      "
          + (r['ultralytics'].get('version', '') if r['ultralytics']['ok']
             else f"opcional → {r['ultralytics'].get('fix','')}"))

    base_ok = r["numpy"]["ok"] and (r["opencv"]["ok"] or r["gstreamer"]["ok"])
    print("-" * 44)
    print("Pronto pra capturar vídeo na GPU." if (r["gstreamer"]["ok"] or
          (r["opencv"]["ok"] and r["opencv"].get("cudacodec")))
          else "Instale as deps de sistema acima p/ habilitar a captura.")
    return 0 if base_ok else 1

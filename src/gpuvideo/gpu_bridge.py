"""Ponte zero-cópia cv2.cuda.GpuMat -> tensor torch CUDA (keep-on-GPU).

O frame decodificado pelo NVDEC vive na GPU como GpuMat. O caminho normal baixa
ele pra CPU (numpy) e o ultralytics re-sobe pra torch — um roundtrip PCIe por
frame. Aqui copiamos GpuMat -> tensor torch **device-to-device** (não toca o
PCIe), via `cudaMemcpy2D` da libcudart por ctypes (sem cupy/pycuda).

Detalhe de corretude: a cópia D2D pode ser assíncrona em relação ao host e o
torch opera no seu próprio stream; então emitimos `cudaMemcpy2DAsync` **no stream
atual do torch**. As ops cv2.cuda sem Stream são bloqueantes (a fonte já está
pronta quando chamamos), e a cópia + ops torch ficam no mesmo stream -> ordenado
sem sync de device por frame.
"""
from __future__ import annotations

import ctypes

_rt = None
_CUDA_MEMCPY_D2D = 3


def _cudart():
    global _rt
    if _rt is None:
        rt = ctypes.CDLL("libcudart.so")
        rt.cudaMemcpy2DAsync.argtypes = [
            ctypes.c_void_p, ctypes.c_size_t,      # dst, dpitch
            ctypes.c_void_p, ctypes.c_size_t,      # src, spitch
            ctypes.c_size_t, ctypes.c_size_t,      # width(bytes), height
            ctypes.c_int, ctypes.c_void_p,         # kind, stream
        ]
        rt.cudaMemcpy2DAsync.restype = ctypes.c_int
        _rt = rt
    return _rt


def gpumat_to_tensor(gmat):
    """GpuMat (uint8, HWC) -> tensor torch uint8 (H,W,C) na GPU. Cópia D2D."""
    import torch
    w, h = gmat.size()                  # GpuMat.size() = (width, height)
    ch = gmat.channels()
    t = torch.empty((h, w, ch), dtype=torch.uint8, device="cuda")
    wbytes = w * ch
    stream = torch.cuda.current_stream().cuda_stream
    err = _cudart().cudaMemcpy2DAsync(
        ctypes.c_void_p(t.data_ptr()), wbytes,
        ctypes.c_void_p(gmat.cudaPtr()), gmat.step,
        wbytes, h, _CUDA_MEMCPY_D2D, ctypes.c_void_p(stream))
    if err != 0:
        raise RuntimeError(f"cudaMemcpy2DAsync falhou (err={err})")
    return t


def preprocess(gmat, half: bool = True):
    """GpuMat RGB (uint8 HWC) -> tensor (1,3,H,W) normalizado [0,1], na GPU.

    Pronto p/ `model.predict(tensor)` do ultralytics (que NÃO faz letterbox nem
    /255 em tensores). Use uma GpuMat já no tamanho de inferência (ex.: 640x640).
    """
    t = gpumat_to_tensor(gmat).permute(2, 0, 1).unsqueeze(0)   # 1,3,H,W
    t = t.half() if half else t.float()
    return t.div_(255.0)

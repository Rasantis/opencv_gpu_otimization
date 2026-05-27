"""Backend OpenCV CUDA nativo (cv2.cudacodec) — NVDEC de verdade dentro do cv2.

Só funciona com um OpenCV compilado com CUDA + Video Codec SDK
(WITH_CUDA + WITH_NVCUVID). O pacote `python3-opencv` do Ubuntu NÃO tem;
veja BUILD_CUDA.md para compilar.

Diferença-chave para os outros caminhos GPU: o frame é decodificado pelo
NVDEC direto para um `GpuMat` (fica na GPU). A conversão NV12->BGR é feita
na GPU (NPP/CUDA), e o processamento (resize, grayscale, ...) também pode
ficar na GPU — baixando para a CPU só o resultado final. Isso elimina o
gargalo de transferência + `videoconvert` que limita os caminhos via
GStreamer neste host.
"""
from __future__ import annotations

from typing import Callable, Optional

from .base import BaseStream
from .frame import Frame


def cudacodec_available() -> bool:
    try:
        import cv2
        return hasattr(cv2, "cudacodec") and cv2.cuda.getCudaEnabledDeviceCount() > 0
    except Exception:
        return False


class CudaCodecStream(BaseStream):
    """NVDEC nativo via cv2.cudacodec.VideoReader.

    Parameters
    ----------
    gpu_op : callable(GpuMat) -> GpuMat, opcional
        Processamento executado NA GPU antes do download. Se devolver um
        GpuMat menor (ex.: 720p cinza), o download fica muito mais barato.
    color : {"BGR", "BGRA", "GRAY"}
        Formato de saída do decoder (conversão feita na GPU).
    download : bool
        Se False, NÃO baixa o frame (mede só decode+processamento na GPU);
        `frame.array` fica vazio. Útil para isolar o custo de transferência.
    """

    backend_name = "opencv-cuda"

    def __init__(self, source, *, gpu_op: Optional[Callable] = None,
                 color: str = "BGR", download: bool = True,
                 as_gpumat: bool = False, stream_id: str = "0") -> None:
        super().__init__(source, stream_id=stream_id)
        self.gpu_op = gpu_op
        self.color = color
        self.download = download
        # as_gpumat: entrega o GpuMat em frame.gpu (sem baixar) p/ keep-on-GPU.
        self.as_gpumat = as_gpumat
        self._reader = None
        self._cv2 = None

    def open(self) -> "CudaCodecStream":
        import cv2
        if not hasattr(cv2, "cudacodec"):
            raise RuntimeError(
                "cv2.cudacodec ausente: rode com a build CUDA do OpenCV "
                "(veja BUILD_CUDA.md / use PYTHONPATH).")
        self._cv2 = cv2
        self._reader = cv2.cudacodec.createVideoReader(str(self.source))
        # Define o formato de cor de saída (conversão na GPU).
        cf = {
            "BGR": getattr(cv2.cudacodec, "ColorFormat_BGR", None),
            "BGRA": getattr(cv2.cudacodec, "ColorFormat_BGRA", None),
            "GRAY": getattr(cv2.cudacodec, "ColorFormat_GRAY", None),
        }.get(self.color)
        if cf is not None:
            try:
                self._reader.set(cf)
            except Exception:
                pass
        try:
            fmt = self._reader.format()
            self.width, self.height = int(fmt.width), int(fmt.height)
        except Exception:
            pass
        self._opened = True
        self._index = 0
        return self

    def read(self) -> Optional[Frame]:
        if not self._opened:
            self.open()
        cv2 = self._cv2
        ok, gpu_frame = self._reader.nextFrame()
        if not ok or gpu_frame is None:
            return None
        # O cudacodec entrega BGRA (4 canais) por padrao. Para paridade com
        # os outros modos (BGR 3 canais), convertemos NA GPU.
        if self.color == "BGR" and gpu_frame.channels() == 4:
            gpu_frame = cv2.cuda.cvtColor(gpu_frame, cv2.COLOR_BGRA2BGR)
        elif self.color == "GRAY" and gpu_frame.channels() == 4:
            gpu_frame = cv2.cuda.cvtColor(gpu_frame, cv2.COLOR_BGRA2GRAY)
        if self.gpu_op is not None:
            gpu_frame = self.gpu_op(gpu_frame, cv2)
        if not self.width:
            self.width, self.height = gpu_frame.size()
        array = None if (self.as_gpumat or not self.download) else gpu_frame.download()
        frame = Frame(
            array=array, index=self._index,
            width=self.width, height=self.height,
            pts_ns=None, capture_monotonic=self._now(), stream_id=self.stream_id,
            gpu=gpu_frame if self.as_gpumat else None,
        )
        self._index += 1
        return frame

    def close(self) -> None:
        self._reader = None
        self._opened = False


# ---- operações GPU prontas (paridade com a op "light"/"heavy" do benchmark) ----
def _to_gray_gpu(small, cv2):
    ch = small.channels()
    if ch == 4:
        return cv2.cuda.cvtColor(small, cv2.COLOR_BGRA2GRAY)
    if ch == 3:
        return cv2.cuda.cvtColor(small, cv2.COLOR_BGR2GRAY)
    return small


def gpu_op_light(gpu_frame, cv2):
    """resize 720p + grayscale, tudo na GPU."""
    small = cv2.cuda.resize(gpu_frame, (1280, 720))
    return _to_gray_gpu(small, cv2)


def gpu_op_heavy(gpu_frame, cv2):
    """resize + blur gaussiano + grayscale, tudo na GPU."""
    small = cv2.cuda.resize(gpu_frame, (1280, 720))
    gray = _to_gray_gpu(small, cv2)
    flt = cv2.cuda.createGaussianFilter(gray.type(), gray.type(), (7, 7), 0)
    return flt.apply(gray)

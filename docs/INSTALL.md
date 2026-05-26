# Instalação

Testado em Ubuntu (resolute), Python 3.14, NVIDIA RTX 3050 6GB Laptop
(driver 595, CUDA 13.2), GStreamer 1.28.

## Dependências do sistema

```bash
# OpenCV para Python (com suporte a GStreamer e FFmpeg embutidos)
sudo apt install -y python3-opencv

# Bindings GObject do GStreamer (GstApp / GstVideo / GstPbutils)
sudo apt install -y gir1.2-gst-plugins-base-1.0 gir1.2-gst-plugins-bad-1.0

# Plugins NVIDIA (NVDEC/NVENC) - normalmente ja' vem em:
sudo apt install -y gstreamer1.0-plugins-bad gstreamer1.0-libav
```

`numpy` vem junto do `python3-opencv`. Não é preciso `pip` nem venv.

## Verificação rápida

```bash
# GPU e plugins NVIDIA
nvidia-smi
gst-inspect-1.0 nvh264dec nvh264enc

# Python
python3 -c "import cv2; print(cv2.__version__)"
python3 -c "import gi; gi.require_version('GstApp','1.0'); from gi.repository import GstApp; print('GstApp ok')"
```

## Por que não pip install opencv-python?

- Não há wheel de `opencv-python` para CPython 3.14 ainda, e o ambiente
  está sem `pip`/`ensurepip`.
- O wheel do PyPI **não** traz suporte a GStreamer (só FFmpeg), o que
  impediria o modo `opencv-gpu` (que consome um pipeline NVDEC via
  `CAP_GSTREAMER`).
- O pacote `python3-opencv` do Ubuntu é compilado **com GStreamer**, que
  é exatamente o que precisamos.

## Sobre cv2.cuda (NVDEC nativo no OpenCV)

O `python3-opencv` do Ubuntu **não** é compilado com CUDA, então
`cv2.cudacodec` / `cv2.cuda` não estão disponíveis. Para decode NVDEC
"nativo" dentro do próprio OpenCV seria preciso compilar o OpenCV do
fonte com `-DWITH_CUDA=ON -DWITH_NVCUVID=ON`. Este projeto contorna isso:
o modo `opencv-gpu` entrega o mesmo ganho de GPU consumindo um pipeline
GStreamer NVDEC, sem precisar recompilar nada.

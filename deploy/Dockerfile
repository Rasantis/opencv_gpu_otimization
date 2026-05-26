# Imagem de deploy do gpuvideo (restreaming acelerado por GPU).
#
# - GStreamer + plugin nvcodec (NVDEC/NVENC) -> usa o driver NVIDIA montado pelo
#   nvidia-container-toolkit em runtime (NVIDIA_DRIVER_CAPABILITIES inclui 'video').
# - WITH_YOLO=1 (padrão) adiciona torch+ultralytics p/ o modo --infer.
#   Para uma imagem enxuta só de transcode: --build-arg WITH_YOLO=0
#
# Build:  docker build -f deploy/Dockerfile -t gpuvideo .
# Run:    docker run --rm --gpus all gpuvideo restream rtsp://CAM rtmp://HOST:1935/cam1
FROM nvidia/cuda:12.6.3-runtime-ubuntu24.04

ARG WITH_YOLO=1
ENV DEBIAN_FRONTEND=noninteractive \
    NVIDIA_DRIVER_CAPABILITIES=all \
    NVIDIA_VISIBLE_DEVICES=all \
    PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1

# GStreamer (com nvcodec em plugins-bad) + bindings GObject + libs do OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv python3-pip python3-gi python3-numpy \
        gir1.2-gst-plugins-base-1.0 gir1.2-gst-plugins-bad-1.0 \
        gstreamer1.0-tools gstreamer1.0-plugins-base gstreamer1.0-plugins-good \
        gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav \
        libgl1 libglib2.0-0 ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# venv que enxerga o python3-gi do sistema (PyGObject não vem por pip)
RUN python3 -m venv --system-site-packages /opt/venv

WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir . \
 && if [ "$WITH_YOLO" = "1" ]; then pip install --no-cache-dir ultralytics; fi

# sanidade em build (não precisa de GPU): import e CLI
RUN python -c "import gpuvideo; print('gpuvideo', gpuvideo.__version__)" \
 && gpuvideo --help >/dev/null

ENTRYPOINT ["gpuvideo"]
CMD ["--help"]

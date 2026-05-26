#!/usr/bin/env bash
# Sobe o stack de restreaming LOCALMENTE (sem Docker) para ver no navegador:
#   câmera (sim) -> MediaMTX -> worker gpuvideo (NVDEC->[YOLO]->NVENC) -> /cam1 -> WebRTC/HLS
#
#   ./examples/local_stack.sh                       # câmera sintética contínua
#   ./examples/local_stack.sh /caminho/video.mp4    # usa um arquivo (re-encodado, ~loop)
#   SOURCE="rtsp://cam/stream" ./examples/local_stack.sh   # câmera REAL (RTSP)
#   INFER=1 ./examples/local_stack.sh               # com detecções YOLO11 (usa .venv-yolo)
#
# Abra:  web/index.html  (host=http://localhost, path=cam1)
#   WebRTC: http://localhost:8889/cam1   |   HLS: http://localhost:8888/cam1/index.m3u8
REPO="$(cd "$(dirname "$0")/.." && pwd)"
MTX_DIR="${MTX_DIR:-/tmp/mediamtx}"
PY="$REPO/.venv-yolo/bin/python"; [ "${INFER:-0}" = 1 ] || PY="python3"
[ -x "$PY" ] || PY="python3"
wait_log() { for _ in $(seq 1 "${3:-20}"); do grep -q "$2" "$1" 2>/dev/null && return 0; sleep 0.5; done; return 1; }

# 1) MediaMTX (baixa se não existir)
if [ ! -x "$MTX_DIR/mediamtx" ]; then
  echo ">> baixando MediaMTX em $MTX_DIR ..."; mkdir -p "$MTX_DIR"
  TAG=$(curl -s https://api.github.com/repos/bluenviron/mediamtx/releases/latest | grep -oP '"tag_name":\s*"\K[^"]+')
  curl -sL "https://github.com/bluenviron/mediamtx/releases/download/${TAG}/mediamtx_${TAG}_linux_amd64.tar.gz" | tar xz -C "$MTX_DIR"
fi
"$MTX_DIR/mediamtx" "$REPO/deploy/mediamtx.yml" >/tmp/mediamtx.log 2>&1 & MTX=$!
wait_log /tmp/mediamtx.log "RTMP. listener opened" 20 && echo ">> MediaMTX no ar (PID $MTX)" || echo "!! MediaMTX nao subiu"

# 2) câmera (origem). Re-encoda (como um encoder ao vivo) para /camera.
SRC="${SOURCE:-}"
if [ -n "$SRC" ]; then
  # câmera REAL: o worker lê direto da fonte; não simula nada.
  WORKER_IN="$SRC"
else
  FILE="${1:-}"
  if [ -n "$FILE" ]; then
    gst-launch-1.0 -q multifilesrc location="$FILE" loop=true ! qtdemux ! h264parse ! nvh264dec ! \
      nvh264enc bitrate=8000 gop-size=30 bframes=0 ! h264parse ! flvmux streamable=true ! \
      rtmp2sink location="rtmp://localhost:1935/camera" >/tmp/cam.log 2>&1 & CAM=$!
  else
    gst-launch-1.0 -q videotestsrc is-live=true pattern=ball ! \
      video/x-raw,width=1280,height=720,framerate=30/1 ! timeoverlay ! \
      nvh264enc preset=low-latency-hq rc-mode=cbr bitrate=4000 gop-size=30 bframes=0 ! \
      h264parse ! flvmux streamable=true ! rtmp2sink location="rtmp://localhost:1935/camera" >/tmp/cam.log 2>&1 & CAM=$!
  fi
  wait_log /tmp/mediamtx.log "publishing to path 'camera'" 20 \
    && echo ">> câmera (sim) publicando em rtmp://localhost:1935/camera (PID $CAM)" \
    || echo "!! câmera nao publicou (veja /tmp/cam.log)"
  WORKER_IN="rtsp://127.0.0.1:8554/camera"
fi

# 3) worker: ingest -> [YOLO] -> NVENC low-latency -> /cam1
INFER_FLAG=""; [ "${INFER:-0}" = 1 ] && INFER_FLAG="--infer"
echo ">> worker: $WORKER_IN -> rtmp://localhost:1935/cam1 $INFER_FLAG"
PYTHONPATH="$REPO/src" "$PY" -m gpuvideo restream "$WORKER_IN" "rtmp://localhost:1935/cam1" $INFER_FLAG & RS=$!

cat <<EOF

==================================================================
  STACK NO AR. Abra no navegador:
    WebRTC (sub-seg): http://localhost:8889/cam1
    LL-HLS:           http://localhost:8888/cam1/index.m3u8
    Player pronto:    file://$REPO/web/index.html  (path=cam1)
  Ctrl+C para parar.
==================================================================
EOF
trap "kill $RS $CAM $MTX 2>/dev/null" EXIT
wait $RS

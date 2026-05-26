# Deploy — restreaming de baixa latência (tempo real)

Arquitetura para ingerir uma transmissão (câmera/origem), processar na GPU
(decode NVDEC + YOLO opcional + encode NVENC) e **re-transmitir** para o front
com baixíssima latência, escalável e de baixo custo.

## Arquitetura

```
   [Câmera / origem]              [Worker gpuvideo (GPU)]                [Borda]            [Front]
   RTSP / RTMP / SRT  ─────►  NVDEC ─► (YOLO11 opcional) ─► NVENC  ──RTMP/SRT──►  MediaMTX  ──►  WebRTC  ─► <video>
   HTTP / arquivo            decode    inferência na GPU   low-latency           (fan-out)     (sub-seg)
                              └──────── tudo na GPU, CPU livre ────────┘          └─ LL-HLS ──►  hls.js (CDN)
```

- **Worker** (`gpuvideo restream`): 1 processo por stream, **stateless** → escala
  horizontal. Decode+encode ficam no NVDEC/NVENC (engines dedicadas) → uma GPU
  aguenta vários streams gastando pouca CPU.
- **MediaMTX** (1 binário Go): recebe do worker e faz o **fan-out** para N
  espectadores em WebRTC e LL-HLS. Os workers **não** lidam com viewers — quem
  escala a audiência é o MediaMTX/CDN.
- **Front**: `<video>` + WebRTC (sub-segundo) ou hls.js (LL-HLS, ~1-3s, CDN-friendly).

## Por que essas escolhas (otimização / latência / escala / custo)

| Objetivo | Decisão |
|---|---|
| **Baixíssima latência** | NVENC `preset=low-latency-hq rc-mode=cbr bframes=0`; WebRTC no front; RTSP/TCP no ingest |
| **Tempo real** | pipeline 100% GPU no transcode (sem roundtrip); 1 GPU transcodifica ~5x 1080p30 |
| **Escalável** | worker stateless (replica por câmera); MediaMTX faz fan-out p/ milhares de viewers |
| **Custo baixo** | NVDEC/NVENC liberam a CPU (ver YOLO: 9% CPU); menos vCPU/instâncias; MediaMTX é 1 binário |
| **Integração fácil** | front é uma tag `<video>` + WebRTC/HLS; sem SDK proprietário (ver `web/index.html`) |

## Testar localmente (sem Docker)

Sobe MediaMTX + uma câmera simulada + o worker, e mostra as URLs pro navegador:

```bash
./examples/local_stack.sh                      # câmera sintética contínua
./examples/local_stack.sh video.mp4            # usa um arquivo
SOURCE="rtsp://cam/stream" ./examples/local_stack.sh   # câmera REAL
INFER=1 ./examples/local_stack.sh              # com detecções YOLO11
```
Abra `web/index.html` (host=`http://localhost`, path=`cam1`) ou
`http://localhost:8889/cam1` (WebRTC).

Frame capturado de uma câmera (vídeo 4K) passando por **toda a cadeia de
restream** (câmera → MediaMTX → worker NVDEC→NVENC → MediaMTX → leitura):

![restream end-to-end](restream_demo.jpg)

> ⚠️ **Simular câmera com um arquivo**: re-encode o vídeo (decode→NVENC), como
> faz o `local_stack.sh` — um encoder ao vivo de verdade. Empurrar o H.264 do
> arquivo **por passthrough** (`h264parse ! flvmux`) pode chegar sem croma
> (verde) por uma incompatibilidade do stream pré-codificado no RTMP; câmeras
> reais (encoder ao vivo) não têm esse problema.

## Subir (Docker)

Pré-requisitos no host: driver NVIDIA + **NVIDIA Container Toolkit** (`nvidia-ctk`).

```bash
cd deploy
SOURCE="rtsp://usuario:senha@camera/stream" docker compose up --build
```

Pronto:
- **WebRTC** (menor latência): `http://HOST:8889/cam1`
- **LL-HLS** (escalável): `http://HOST:8888/cam1/index.m3u8`
- Player de exemplo: abra `web/index.html` (aponte o host) — ver [README](../README.md).

Imagem **enxuta só de transcode** (sem torch/YOLO): `--build-arg WITH_YOLO=0`.

## Modos do worker

```bash
# transcode puro (100% GPU, menor latência)
gpuvideo restream rtsp://CAM rtmp://127.0.0.1:1935/cam1 --bitrate 4000 --gop 30

# com detecção YOLO11 desenhada no stream
gpuvideo restream rtsp://CAM rtmp://127.0.0.1:1935/cam1 --infer --model yolo11n.pt

# SRT (menor latência que RTMP no hop worker->borda)
gpuvideo restream rtsp://CAM srt://127.0.0.1:8890?streamid=publish:cam1 --protocol srt
```

## Orçamento de latência (típico, 1080p)

| Etapa | Latência |
|---|---|
| Captura + rede (RTSP/TCP) | depende da origem |
| NVDEC decode | ~3-5 ms |
| YOLO11n (se `--infer`) | ~5 ms |
| NVENC encode (low-latency) | ~5-10 ms + ~1 GOP |
| Worker → MediaMTX (RTMP/SRT, mesma rede) | ~10-50 ms |
| **MediaMTX → front (WebRTC)** | **~100-300 ms** |
| MediaMTX → front (LL-HLS) | ~1-3 s |

→ **glass-to-glass ~0.3-0.6 s** via WebRTC. Reduza o `--gop` p/ join mais rápido
(custa bitrate); use `--protocol srt` e WebRTC pra cortar o resto.

## Escala

- **Mais câmeras**: replique o serviço `restreamer` no compose (um path por câmera).
  Uma RTX 3050 transcodifica vários 1080p30 simultâneos (NVENC é o limite).
- **Mais viewers**: o MediaMTX serve N leitores; para audiência grande, ponha um
  CDN na frente do LL-HLS, ou múltiplas instâncias MediaMTX atrás de um LB.
- **Multi-GPU / multi-host**: workers são stateless; orquestre com Kubernetes
  (1 pod por stream, `nvidia.com/gpu` request) apontando todos pro mesmo MediaMTX.

## Notas de produção

- `network_mode: host` simplifica o ICE do WebRTC. Atrás de NAT, configure
  `webrtcAdditionalHosts` (IP público) e STUN/TURN no `mediamtx.yml`.
- O NVDEC/NVENC no container vêm do **driver montado** pelo nvidia-container-toolkit
  (precisa `NVIDIA_DRIVER_CAPABILITIES` incluir `video` — já setado na imagem).
- Use tags de versão fixas (`gpuvideo @ ...@v0.1.0`, `mediamtx:<versão>`) em produção.

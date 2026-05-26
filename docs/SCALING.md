# Roadmap de escala — multi-câmera, multi-modelo, tempo real (visão de CTO)

Onde estamos: 1 worker = 1 câmera = 1 decode (NVDEC) + 1 inferência. Funciona e
é ótimo na borda/protótipo. Onde queremos chegar: **centenas de câmeras**, cada
uma com **modelo/tamanho/solução próprios** (contagem, permanência, mapa de
calor, invasão), em **cloud com pods que escalam**, em tempo real e barato.

Este doc é o plano para chegar lá, ancorado no que medimos (NVDEC é o limite por
GPU; baixar o frame inteiro é o gargalo; NVENC/decode liberam CPU).

---

## 1. A mudança arquitetural central

Hoje cada câmera carrega seu próprio modelo → desperdício de VRAM e de GPU.
A arquitetura que escala separa **3 planos**:

```
 Câmeras ─► [DECODE workers]  ─frames(GPU)─►  [INFERÊNCIA batched]  ─dets─►  [ANALYTICS/solução]  ─►  eventos
  RTSP        1 NVDEC/câmera     fila/NVMM       Triton/TensorRT,            tracking + counting/         NATS/Kafka
  RTMP        (stateless)        shared mem      batch de N câmeras          dwell/heatmap/intrusion      → DB/alerta/dash
                                                 (1 instância p/ modelo)     (stateful por câmera, leve)
                         └────────────── tudo o que dá fica na GPU; só metadados descem ──────────────┘
```

- **Decode != Inferência**: um pool de workers de decode (NVDEC) alimenta um
  pool de **servidores de inferência compartilhados**. Não se carrega 1 modelo
  por câmera; carrega-se 1 instância por (modelo, GPU) e faz-se **batch** dos
  frames de várias câmeras.
- **Batch entre câmeras** é o maior ganho: 16 câmeras → 1 chamada de GPU em vez
  de 16. É o que o NVIDIA DeepStream (`nvstreammux`) e o Triton (dynamic
  batching) fazem nativamente.
- **Solução = tracking + lógica leve**: contagem, permanência, heatmap e invasão
  são todas construídas sobre **rastreamento de IDs** (ByteTrack/BoT-SORT) + um
  módulo de regra por câmera. O estado é pequeno (IDs, timestamps, acumuladores).
- **Eventos, não vídeo**: o que trafega entre serviços são **metadados** (JSON:
  câmera, track_id, classe, bbox, evento) num barramento (NATS/Kafka/Redis
  Streams). Vídeo anotado só quando alguém está vendo (WebRTC/LL-HLS sob demanda).

---

## 2. Alavancas de performance (ordenadas por impacto)

1. **Inferência em batch entre câmeras** — 2-10x throughput. Triton dynamic
   batching ou DeepStream `nvstreammux` (batch-size = nº de câmeras por GPU).
2. **TensorRT + FP16/INT8** — exporte YOLO11 p/ engine (`model.export(format="engine", half=True)`).
   FP16 ~2x, INT8 ~3-4x sobre PyTorch, com VRAM menor. Calibração INT8 por modelo.
3. **Desacoplar fps de decode e de inferência** — decodifica a 30 fps, infere a
   5-10 fps (contagem/permanência não precisam de 30). Corte direto de custo.
4. **Manter o frame na GPU** (o gargalo que medimos): NVDEC → GpuMat → TensorRT
   sem baixar; só os metadados (caixas) descem. Evita o roundtrip de 25 MB/frame.
5. **Densidade de NVDEC** — escolha a GPU pelo nº de engines de vídeo, não só
   FLOPs: **L4** (4×NVDEC, 2×NVENC, barata, 72W) >> T4 p/ muitas câmeras.
6. **ROI / tiling / resolução adaptativa** — processe só a região de interesse e
   em 640; mantenha full-res só p/ gravação/exibição.
7. **Codec certo no ingest** — H.264/H.265 4:2:0 8-bit (o que o NVDEC faz).

---

## 3. Multi-câmera, multi-modelo, multi-solução

Cada câmera é uma **config declarativa** (YAML), o orquestrador resolve:

```yaml
cameras:
  - id: loja-entrada
    source: rtsp://cam1/stream
    model: yolo11n-int8.engine      # modelo/tamanho por câmera
    infer_fps: 8
    roi: [[0,0],[1,0],[1,1],[0,1]]
    solutions:
      - type: line_counting          # contagem (entrada/saída)
        line: [[0,0.6],[1,0.6]]
        classes: [person]
      - type: dwell_time             # tempo de permanência
        region: [[0.2,0.3],[0.8,0.9]]
  - id: doca-fundos
    source: rtsp://cam2/stream
    model: yolo11s.engine
    solutions:
      - type: intrusion              # detecção de invasão
        zone: [[...]]
        schedule: "22:00-06:00"
        alert: webhook
      - type: heatmap                # mapa de calor
        window: 1h
```

**Catálogo de soluções** (todas sobre tracking, plugáveis):

| Solução | Base | Estado por câmera | Saída |
|---|---|---|---|
| **Contagem** (linha/região) | track IDs cruzando linha/ROI | IDs já contados | counter, eventos in/out |
| **Permanência (dwell)** | enter/exit timestamp por track | {track_id: t_entrada} | tempo médio, alerta > limite |
| **Mapa de calor** | acumula centróides/densidade | grade acumuladora | imagem/grade por janela |
| **Invasão** | track entra em polígono (+ horário) | tracks ativos na zona | alerta (webhook/MQTT) |
| **Fluxo/direção** | vetor de movimento dos tracks | trajetórias curtas | matriz origem-destino |

Design: interface `Solution.process(frame_meta, detections, tracks) -> [Event]`.
Modelos via **model registry** (S3/MinIO + versão); carregados sob demanda;
instâncias compartilhadas por (modelo, GPU) e reusadas entre câmeras iguais.

---

## 4. Cloud / Kubernetes / pods

- **Pod de decode+analytics por câmera** (ou N câmeras/pod), **leve em CPU**
  (decode no NVDEC). Inferência num **Deployment separado** (Triton) com HPA.
- **Empacotar mais câmeras por GPU**: time-slicing (oversubscrição simples),
  **MPS** (concorrência real de contexto) ou **MIG** (partições isoladas em
  A100/H100). Para decode-bound, time-slicing/MPS já densificam bem.
- **Autoscaling por demanda real**: **KEDA** escalando por profundidade de fila
  / nº de câmeras ativas; **scale-to-zero** fora de horário. HPA por GPU util.
- **Scheduling**: `nvidia.com/gpu` requests; node pools por tipo de GPU (L4 p/
  vídeo). Spot/preemptível p/ workers stateless = -70% de custo.
- **Plano de mídia**: cameras → gateway (MediaMTX em cluster atrás de LB, ou os
  workers puxam RTSP direto). Saída ao vivo só sob demanda; CDN na frente do
  LL-HLS para audiência grande.
- **Plano de controle**: registry de câmeras + configs + modelos; um operador/
  controller que cria/destrói pods conforme as câmeras ativas (1 CRD `Camera`).
- **Dados**: eventos → Kafka/NATS → DB de séries temporais (counting/dwell) +
  alertas; gravações → object storage (S3) com retenção.

---

## 5. Build vs Buy — recomendação honesta

Para o **núcleo de inferência batched multi-câmera**, o **NVIDIA DeepStream**
(+ **Triton**) é feito exatamente pra isso e economiza meses: `nvstreammux`
(batch de N streams), `nvinfer` (TensorRT batched), `nvtracker` (ByteTrack/NvDCF),
`nvdsanalytics` (linha/ROI/direção = contagem e invasão prontas), tudo na GPU
sem roundtrip. **Triton** sozinho serve os modelos com batching/versionamento se
quisermos manter o pipeline em Python.

**Onde o `gpuvideo` brilha** (e devemos manter): a camada de **ingest/restream
de baixa latência** (NVDEC→NVENC→WebRTC), o **edge** e protótipos, a cola
flexível e a API simples. Estratégia: `gpuvideo` na borda + restream; **Triton/
DeepStream no core de analytics** quando o nº de câmeras justificar.

---

## 6. Observabilidade (obrigatório p/ operar em escala)

Métricas Prometheus por câmera: **fps de decode, fps de inferência, latência
e2e, profundidade de fila, util de GPU/NVDEC/NVENC, VRAM, frames dropados,
eventos/min**. Dashboards Grafana + alertas (câmera caída, fila estourando,
GPU saturada). Tracing distribuído nos eventos (câmera → detecção → alerta).

---

## 7. Próximos passos no `gpuvideo` (priorizados)

1. ✅ **Tracking** (ByteTrack + rastro/trail por ID) — base de todas as soluções.
2. ✅ **Interface `Solution`** + módulos (counting, dwell, heatmap, intrusion).
3. ✅ **Config declarativa por câmera** (YAML) + runner multi-câmera (`gpuvideo analytics`).
   + ✅ **keep-on-GPU** (cudacodec + resize na GPU) → event-only ~120 fps/câmera.
4. **Export TensorRT** + caminho de inferência FP16/INT8.
5. **Inferência em batch** (N câmeras → 1 chamada) — via Triton client ou batch local.
6. **Saída de eventos** (JSON) num barramento (NATS/Redis) — não só vídeo.
7. **Decoupling de fps** (infer a cada N frames) + **keep-on-GPU**.
8. **K8s**: manifests, CRD `Camera`, KEDA, time-slicing/MPS; **Prometheus exporter**.

Sugestão de ordem de entrega: **1→2→3** (vira produto de analytics de verdade),
depois **4→5** (performance/custo), depois **8** (operar em escala).

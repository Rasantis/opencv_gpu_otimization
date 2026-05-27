"""Observabilidade no formato Prometheus — sem dependências (stdlib só).

O formato de exposição do Prometheus é texto simples, então não arrastamos o
`prometheus_client` pra base. Um registry thread-safe coleta counters/gauges e um
servidor HTTP da stdlib serve `/metrics` (e `/healthz`). Pronto p/ scrape do
Prometheus e p/ dimensionar pods (HPA/KEDA) por fps/latência/fila.

    from gpuvideo.metrics import REGISTRY, default_metrics, start_http_server
    default_metrics(REGISTRY)
    start_http_server(9108)
    REGISTRY.inc("gpuvideo_frames_decoded_total", {"camera": "c1"})

O runner multi-câmera liga isso com `gpuvideo analytics cams.yaml --metrics-port 9108`.
"""
from __future__ import annotations

import threading
import time
from typing import Dict, Optional, Tuple


def _esc(v: str) -> str:
    return str(v).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _fmt(v: float) -> str:
    return str(int(v)) if float(v).is_integer() else repr(float(v))


class Metrics:
    """Registry mínimo de counters/gauges com labels, render Prometheus."""

    def __init__(self):
        self._lock = threading.Lock()
        self._data: Dict[str, Dict[Tuple, float]] = {}
        self._meta: Dict[str, Tuple[str, str]] = {}
        self.t0 = time.time()

    @staticmethod
    def _key(labels: Optional[dict]) -> Tuple:
        return tuple(sorted((labels or {}).items()))

    def define(self, name: str, type_: str, help_: str = ""):
        self._meta[name] = (type_, help_)

    def inc(self, name: str, labels: Optional[dict] = None, v: float = 1.0):
        with self._lock:
            d = self._data.setdefault(name, {})
            k = self._key(labels)
            d[k] = d.get(k, 0.0) + v

    def set(self, name: str, labels: Optional[dict] = None, v: float = 0.0):
        with self._lock:
            self._data.setdefault(name, {})[self._key(labels)] = float(v)

    def render(self) -> str:
        out = []
        with self._lock:
            for name in sorted(self._data):
                typ, help_ = self._meta.get(name, ("untyped", ""))
                if help_:
                    out.append(f"# HELP {name} {help_}")
                out.append(f"# TYPE {name} {typ}")
                for k, val in self._data[name].items():
                    lbl = ""
                    if k:
                        lbl = "{" + ",".join(f'{kk}="{_esc(vv)}"' for kk, vv in k) + "}"
                    out.append(f"{name}{lbl} {_fmt(val)}")
        return "\n".join(out) + "\n"


REGISTRY = Metrics()


def default_metrics(reg: Metrics = REGISTRY):
    """Declara HELP/TYPE das métricas padrão do gpuvideo."""
    reg.define("gpuvideo_frames_decoded_total", "counter", "Frames decodificados por câmera")
    reg.define("gpuvideo_inferences_total", "counter", "Inferências executadas por câmera")
    reg.define("gpuvideo_events_total", "counter", "Eventos de analytics por câmera e tipo")
    reg.define("gpuvideo_reconnects_total", "counter", "Reconexões de fonte ao vivo por câmera")
    reg.define("gpuvideo_camera_up", "gauge", "1 se a câmera está processando, 0 se caiu/parou")
    reg.define("gpuvideo_decode_fps", "gauge", "FPS de decode (instantâneo) por câmera")
    reg.define("gpuvideo_inference_fps", "gauge", "FPS de inferência (instantâneo) por câmera")
    reg.define("gpuvideo_inference_latency_ms", "gauge", "Latência do último forward por câmera")
    reg.define("gpuvideo_tracks", "gauge", "Objetos rastreados no último frame por câmera")
    reg.define("gpuvideo_batch_size_avg", "gauge", "Batch médio efetivo por modelo compartilhado")
    reg.define("gpuvideo_uptime_seconds", "gauge", "Tempo de vida do processo")


def start_http_server(port: int, reg: Metrics = REGISTRY):
    """Sobe um servidor HTTP (daemon) servindo /metrics e /healthz. Devolve o server."""
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            path = self.path.split("?")[0]
            if path in ("/metrics", "/"):
                reg.set("gpuvideo_uptime_seconds", None, time.time() - reg.t0)
                body = reg.render().encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif path == "/healthz":
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"ok")
            else:
                self.send_response(404)
                self.end_headers()

        def log_message(self, *_):  # silêncio (não polui o stdout do runner)
            pass

    srv = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    threading.Thread(target=srv.serve_forever, name="metrics-http", daemon=True).start()
    print(f"Métricas Prometheus em http://0.0.0.0:{port}/metrics", flush=True)
    return srv

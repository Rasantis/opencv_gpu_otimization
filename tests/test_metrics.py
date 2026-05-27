"""Testes do registry de métricas (formato Prometheus, sem rede/GPU)."""
from gpuvideo.metrics import Metrics, default_metrics


def test_counter_e_gauge():
    m = Metrics()
    m.define("gpuvideo_frames_decoded_total", "counter", "frames")
    m.inc("gpuvideo_frames_decoded_total", {"camera": "c1"})
    m.inc("gpuvideo_frames_decoded_total", {"camera": "c1"}, 4)
    m.set("gpuvideo_decode_fps", {"camera": "c1"}, 24.5)
    out = m.render()
    assert "# TYPE gpuvideo_frames_decoded_total counter" in out
    assert 'gpuvideo_frames_decoded_total{camera="c1"} 5' in out
    assert 'gpuvideo_decode_fps{camera="c1"} 24.5' in out


def test_labels_ordenados_e_multiplos():
    m = Metrics()
    m.inc("gpuvideo_events_total", {"type": "count_in", "camera": "c2"})
    out = m.render()
    # labels saem em ordem alfabética (camera antes de type)
    assert 'gpuvideo_events_total{camera="c2",type="count_in"} 1' in out


def test_escape_de_valores():
    m = Metrics()
    m.set("g", {"path": 'a"b\\c'}, 1)
    assert 'path="a\\"b\\\\c"' in m.render()


def test_sem_labels():
    m = Metrics()
    m.set("gpuvideo_uptime_seconds", None, 12)
    assert "gpuvideo_uptime_seconds 12" in m.render()


def test_default_metrics_define_tudo():
    m = Metrics()
    default_metrics(m)
    assert "gpuvideo_inference_latency_ms" in m._meta
    assert m._meta["gpuvideo_camera_up"][0] == "gauge"

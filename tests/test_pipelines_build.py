"""Testes dos construtores de pipeline e detecção de fonte (sem GStreamer)."""
import pytest
from gpuvideo.pipelines import classify_source, build_pipeline, SourceKind
from gpuvideo import restream


def test_classify_source():
    assert classify_source("x.mp4") == SourceKind.FILE
    assert classify_source("rtsp://cam/s") == SourceKind.RTSP
    assert classify_source("rtmp://srv/s") == SourceKind.RTMP
    assert classify_source("rtmps://srv/s") == SourceKind.RTMP
    assert classify_source("http://h/v.mp4") == SourceKind.HTTP
    assert classify_source("test") == SourceKind.TEST
    assert classify_source(0) == SourceKind.CAMERA
    assert classify_source("/dev/video0") == SourceKind.CAMERA


def test_build_pipeline_gpu_cpu():
    gpu = build_pipeline("x.mp4", engine="gpu", codec="h264")
    assert "nvh264dec" in gpu and "appsink" in gpu and "qtdemux" in gpu
    cpu = build_pipeline("x.mp4", engine="cpu", codec="h264")
    assert "avdec_h264" in cpu and "nvh264dec" not in cpu


def test_build_pipeline_rtsp_tcp():
    p = build_pipeline("rtsp://cam/stream", engine="gpu", codec="h264")
    assert "rtspsrc" in p and "protocols=tcp" in p and "nvh264dec" in p


def test_build_pipeline_rtmp():
    p = build_pipeline("rtmp://srv/s", engine="gpu", codec="h264")
    assert "rtmp2src" in p and "flvdemux" in p


def test_build_pipeline_codecs():
    assert "nvh265dec" in build_pipeline("x.mkv", engine="gpu", codec="h265")
    assert "nvvp9dec" in build_pipeline("x.webm", engine="gpu", codec="vp9")
    assert "vp9parse" in build_pipeline("x.webm", engine="gpu", codec="vp9")


def test_restream_nvenc_sink():
    enc = restream._nvenc("h264", 4000, 30, low_latency=True)
    assert "nvh264enc" in enc and "low-latency" in enc and "bframes=0" in enc
    assert "nvh265enc" in restream._nvenc("h265", 4000, 30, True)
    rtmp = restream._sink("rtmp", "rtmp://h/s", "h264")
    assert "flvmux" in rtmp and "rtmp2sink" in rtmp
    srt = restream._sink("srt", "srt://h", "h264")
    assert "mpegtsmux" in srt and "srtsink" in srt
    with pytest.raises(ValueError):
        restream._sink("zzz", "x", "h264")

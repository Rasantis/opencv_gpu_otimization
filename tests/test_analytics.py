"""Testes da lógica pura de analytics (sem GPU/cv2/YOLO)."""
from gpuvideo.analytics import (Track, LineCounter, IntrusionZone, DwellZone,
                                _point_in_poly, _color_for_id)


def trk(tid, bbox, name="person"):
    return Track(tid, 0, name, bbox, 0.9)


def test_track_center_foot():
    t = trk(1, (10, 20, 30, 60))
    assert t.center == (20, 40)
    assert t.foot == (20, 60)      # base do bbox (pé no chão)


def test_point_in_poly():
    sq = [(0, 0), (1, 0), (1, 1), (0, 1)]
    assert _point_in_poly((0.5, 0.5), sq)
    assert not _point_in_poly((1.5, 0.5), sq)
    assert not _point_in_poly((-0.1, 0.5), sq)


def test_line_counter_in_out():
    lc = LineCounter([(0, 0.5), (1, 0.5)], name="l")
    # acima -> abaixo = count_in
    lc.process([trk(1, (40, 10, 60, 40))], 100, 100, 0.0)   # foot y=0.40
    ev = lc.process([trk(1, (40, 60, 60, 90))], 100, 100, 1.0)  # foot y=0.90
    assert lc.count_in == 1 and lc.count_out == 0
    assert ev[0].type == "count_in" and ev[0].track_id == 1
    # abaixo -> acima = count_out
    ev = lc.process([trk(1, (40, 10, 60, 40))], 100, 100, 2.0)
    assert lc.count_out == 1 and ev[0].type == "count_out"


def test_line_counter_filtra_classe():
    lc = LineCounter([(0, 0.5), (1, 0.5)], name="l", classes=["car"])
    lc.process([trk(1, (40, 10, 60, 40), "person")], 100, 100, 0.0)
    ev = lc.process([trk(1, (40, 60, 60, 90), "person")], 100, 100, 1.0)
    assert ev == [] and lc.count_in == 0   # person ignorado (só conta car)


def test_intrusion_uma_vez_e_reentrada():
    iz = IntrusionZone([(0, 0), (0.5, 0), (0.5, 1), (0, 1)], name="z")
    ev = iz.process([trk(7, (10, 10, 30, 40))], 100, 100, 0.0)  # foot x=0.2 dentro
    assert len(ev) == 1 and ev[0].type == "intrusion"
    ev = iz.process([trk(7, (10, 10, 30, 40))], 100, 100, 0.1)  # continua dentro
    assert ev == []                                             # não repete
    iz.process([trk(7, (80, 10, 95, 40))], 100, 100, 0.2)       # saiu (x=0.875)
    ev = iz.process([trk(7, (10, 10, 30, 40))], 100, 100, 0.3)  # reentrou
    assert len(ev) == 1 and ev[0].type == "intrusion"


def test_dwell_alerta_e_saida():
    dz = DwellZone([(0, 0), (1, 0), (1, 1), (0, 1)], name="z", alert_s=2.0)
    dz.process([trk(3, (40, 40, 60, 60))], 100, 100, 0.0)   # entra t=0
    ev = dz.process([trk(3, (40, 40, 60, 60))], 100, 100, 2.5)  # dwell 2.5 >= 2 -> alerta
    assert any(e.type == "dwell_alert" for e in ev)
    ev = dz.process([], 100, 100, 3.0)                      # saiu
    exits = [e for e in ev if e.type == "dwell_exit"]
    assert exits and exits[0].data["dwell_s"] == 3.0


def test_color_deterministico():
    assert _color_for_id(1) == _color_for_id(1)
    assert isinstance(_color_for_id(5), tuple) and len(_color_for_id(5)) == 3

"""Съветите към потребителя в отчета на модела (pipeline.tip, glare_share): само по измереното в сканирането."""

import numpy as np
import trimesh
from PIL import Image

from app import pipeline as P


def frames(tmp_path, white: float, n: int = 12):
    """n кадъра 200 x 120 с предмет (кафяв правоъгълник) и `white` дял от него бял отблясък; маските са 1/4 от
    размера, както от app/segment.py."""
    photos = tmp_path / "photos"
    photos.mkdir(parents=True)
    masks = {}
    for k in range(n):
        a = np.full((120, 200, 3), 60, np.uint8)
        a[30:90, 50:150] = (150, 100, 70)
        rows = int(60 * white)
        a[30:30 + rows, 50:150] = 255
        name = f"img_{k + 1:04d}.jpg"
        Image.fromarray(a).save(photos / name, quality=95)
        m = np.zeros((30, 50), np.uint8)
        m[8:22, 13:37] = 255
        masks[name] = m
    return photos, masks


def test_glare_is_measured_on_the_object_only(tmp_path):
    photos, masks = frames(tmp_path / "plain", 0.0)
    assert P.glare_share(photos, masks) < 0.5
    photos, masks = frames(tmp_path / "shiny", 0.3)
    assert P.glare_share(photos, masks) > 20


def test_tips_go_into_the_report_once(tmp_path):
    job = tmp_path / "job"
    job.mkdir()
    P.tip(job, "plain")
    P.tip(job, "plain")
    P.tip(job, "glare")
    P.tip(job, "unknown")  # без текст: не стига до приложението
    mesh = trimesh.creation.box((10, 10, 10))
    report = P.finish(job, mesh)
    assert report["tips"] == [P.TIP_TEXT["plain"], P.TIP_TEXT["glare"]]


def test_no_tips_no_field(tmp_path):
    job = tmp_path / "job"
    job.mkdir()
    assert "tips" not in P.finish(job, trimesh.creation.box((10, 10, 10)))

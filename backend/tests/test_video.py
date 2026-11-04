import subprocess

import numpy as np
import pytest
from PIL import Image

from app import video as V


def test_sharpness_prefers_sharp_frames():
    rng = np.random.default_rng(0)
    sharp = rng.integers(0, 256, (1, 64, 64)).astype(np.uint8)
    blurred = ((sharp[:, :-1, :-1].astype(int) + sharp[:, 1:, 1:]) // 2).astype(np.uint8)[:, :63, :63]
    flat = np.full((1, 64, 64), 128, np.uint8)
    s_sharp, s_blur, s_flat = V.sharpness(sharp)[0], V.sharpness(blurred)[0], V.sharpness(flat)[0]
    assert s_sharp > s_blur > s_flat == 0


def test_score_thumbs_ignores_trailing_partial_frame():
    raw = np.zeros(V.THUMB**2 * 3 + 10, np.uint8)
    assert len(V.score_thumbs(raw)) == 3
    assert len(V.score_thumbs(np.zeros(5, np.uint8))) == 0


def test_pick_one_sharpest_per_slice():
    scores = np.array([1, 5, 2, 9, 3, 3, 0, 7, 1])
    assert V.pick(scores, 3) == [1, 3, 7]
    assert V.pick(scores, len(scores)) == list(range(len(scores)))


def test_motion_measures_change_between_frames():
    a = np.zeros((V.THUMB, V.THUMB), np.uint8)
    b = np.full((V.THUMB, V.THUMB), 100, np.uint8)
    m = V.motion(np.stack([a, a, b, b]).ravel())
    assert m.tolist() == [0, 0, 100, 0]
    assert len(V.motion(np.zeros(5, np.uint8))) == 0


def test_pick_spreads_frames_by_movement_not_time():
    # 10 неподвижни кадъра, после 10 движещи се: почти всички избрани трябва да са от движещата се част.
    moved = np.array([0.0] * 10 + [1.0] * 10)
    scores = np.ones(20)
    picks = V.pick(scores, 10, moved)
    assert sum(p >= 10 for p in picks) >= 8
    assert picks == sorted(set(picks))
    # Нищо не се е движило: обратно към равни участъци по време.
    assert V.pick(scores, 4, np.zeros(20)) == V.pick(scores, 4)


def test_frame_count_bounds():
    assert V.frame_count(30 * V.CANDIDATE_FPS) == 30 * V.FRAMES_PER_S
    assert V.frame_count(5 * V.CANDIDATE_FPS) == V.MIN_FRAMES
    assert V.frame_count(10) == 10  # никога повече кадри от кандидатите
    assert V.frame_count(600 * V.CANDIDATE_FPS) == V.MAX_FRAMES


def test_extract_frames_from_real_video(tmp_path):
    from app.pipeline import FFMPEG, Failed, extract_frames

    def sh(_step, argv):
        subprocess.run([str(a) for a in argv], check=True)

    # Вертикален клип като от телефон, по-голям от ограничението за кадри (4K, за текстурата от снимки), за да се
    # задейства филтърът за мащабиране.
    clip = tmp_path / "video.mp4"
    sh("", [FFMPEG, "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=1800x4000:rate=30:duration=20",
            "-pix_fmt", "yuv420p", clip])  # fmt: skip
    photos, work = tmp_path / "photos", tmp_path / "work"
    photos.mkdir()
    work.mkdir()
    (photos / "img_9999.jpg").write_bytes(b"stale")  # остатък от по-ранно изпълнение

    n = extract_frames(clip, photos, work, sh)
    names = sorted(p.name for p in photos.iterdir())
    assert n == V.frame_count(20 * V.CANDIDATE_FPS) == 20 * V.FRAMES_PER_S
    assert names == [f"img_{i:04d}.jpg" for i in range(1, n + 1)]
    w, h = Image.open(photos / "img_0001.jpg").size
    assert max(w, h) <= 3840 and abs(w / h - 1800 / 4000) < 0.01
    assert not (work / "thumbs.gray").exists()

    short = tmp_path / "short.mp4"
    sh("", [FFMPEG, "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=320x240:rate=30:duration=2",
            "-pix_fmt", "yuv420p", short])  # fmt: skip
    with pytest.raises(Failed, match="твърде кратко"):
        extract_frames(short, photos, work, sh)

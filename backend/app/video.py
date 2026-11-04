"""Orbit video -> the sharpest frame from each equal step of camera movement, so frames are evenly spaced around
the object even when the user walks at an uneven pace or stands still for a moment."""

import numpy as np

# Frames scored per second of video: each kept frame is the sharpest of about three around it (at 6 a second and 120
# frames kept there were only two, and a scan kept frames at a sixth of the median sharpness).
CANDIDATE_FPS = 10
FRAMES_PER_S = 3  # frames kept per second: a 30 s circle gives 90 frames 4° apart
# Two circles in 120 frames are 6° apart: every point of the surface is in more frames, so the fused surface averages
# more measurements (less noise) and the texture has sharper, straighter-on views to pick from. Matching cost grows
# with the square of this (80 frames: 4 min on the server; 120: about 9) and the per-frame steps linearly.
MIN_FRAMES, MAX_FRAMES = 20, 120
MIN_SECONDS = 4  # shorter videos are refused: no circle around anything
# Quick scans (the app's choice): two circles in 48 frames are ~15° apart, still enough overlap to match; matching
# takes about a third of the time and every per-frame step about 60%.
QUICK_FRAMES = 48
THUMB = 256  # square scoring size; squashing the aspect ratio does not change the sharpness ranking
SMALL = 4  # motion is measured on THUMB/SMALL thumbnails, where noise and blur hardly differ
STILL = 0.1  # standing still still counts this fraction of average movement, so time is never ignored
BATCH = 64


def sharpness(frames: np.ndarray) -> np.ndarray:
    """Variance of the Laplacian per frame, (n, h, w) uint8 -> (n,). Motion blur lowers it."""
    f = frames.astype(np.float32)
    lap = f[:, 1:-1, :-2] + f[:, 1:-1, 2:] + f[:, :-2, 1:-1] + f[:, 2:, 1:-1] - 4 * f[:, 1:-1, 1:-1]
    return lap.reshape(len(f), -1).var(axis=1)


def _thumbs(raw: np.ndarray) -> np.ndarray:
    return raw[: len(raw) // THUMB**2 * THUMB**2].reshape(-1, THUMB, THUMB)


def score_thumbs(raw: np.ndarray) -> np.ndarray:
    """Sharpness of each THUMB x THUMB gray frame in a raw byte buffer, in batches to bound memory."""
    thumbs = _thumbs(raw)
    return np.concatenate([sharpness(thumbs[i : i + BATCH]) for i in range(0, len(thumbs), BATCH)] or [np.zeros(0)])


def motion(raw: np.ndarray) -> np.ndarray:
    """How much the picture changed since the previous frame (mean absolute difference of small thumbnails)."""
    t = _thumbs(raw)
    k = THUMB // SMALL
    small = np.concatenate(
        [t[i : i + BATCH].reshape(-1, k, SMALL, k, SMALL).mean((2, 4), dtype=np.float32) for i in range(0, len(t), BATCH)]
        or [np.zeros((0, k, k), np.float32)]
    )
    return np.concatenate([[0.0], np.abs(np.diff(small, axis=0)).mean((1, 2))]) if len(small) else np.zeros(0)


def frame_count(candidates: int, most: int = MAX_FRAMES) -> int:
    n = round(candidates / CANDIDATE_FPS * FRAMES_PER_S)
    return min(candidates, max(MIN_FRAMES, min(most, n)))


def pick(scores: np.ndarray, n: int, moved: np.ndarray | None = None) -> list[int]:
    """Index of the sharpest candidate in each of n slices holding equal camera movement (equal time without
    `moved`). A sudden swing can leave a slice empty; it is skipped."""
    if moved is None or moved.sum() <= 0:
        edges = np.linspace(0, len(scores), n + 1).round().astype(int)
    else:
        progress = np.cumsum(moved + STILL * moved.mean())
        edges = np.searchsorted(progress, np.linspace(0, progress[-1], n + 1)[1:-1], side="right")
        edges = np.concatenate([[0], edges, [len(scores)]])
    return [int(a + np.argmax(scores[a:b])) for a, b in zip(edges[:-1], edges[1:]) if b > a]

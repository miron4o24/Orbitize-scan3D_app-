"""Seam levelling and gap filling of the photo texture (app/texture.py), on a square of two patches."""

import numpy as np

from app import texture as TX


def two_patches():
    """A unit square of two triangles that share an edge on the model but lie apart in the atlas: one patch grey 80,
    the other grey 160 (two frames exposed differently), the rest of the atlas EMPTY."""
    a = np.full((64, 64, 3), TX.EMPTY, np.uint8)
    a[4:28, 4:60] = 80
    a[36:60, 4:60] = 160
    # Model vertices 0..3 (corners of the square); faces (0, 1, 2) and (0, 2, 3) share the edge 0-2.
    uv = np.array([
        [[0.1, 0.9], [0.9, 0.9], [0.9, 0.6]],  # rows 6..26 of the atlas (v up)
        [[0.1, 0.4], [0.9, 0.1], [0.1, 0.1]],  # rows 38..58
    ])  # fmt: skip
    corner = np.array([[0, 1, 2], [0, 2, 3]])
    return a, uv, corner


def test_patches_meet_and_the_gaps_take_their_colours():
    a, uv, corner = two_patches()
    out, covered, note = TX.level(a, uv, corner, np.ones(2, bool))
    assert "seam points" in note
    assert (np.abs(out.astype(int) - TX.EMPTY).sum(2) >= 120).all()  # no EMPTY left between the patches
    low, high = out[covered & (np.arange(64)[:, None] < 32)], out[covered & (np.arange(64)[:, None] >= 32)]
    assert high.mean() - low.mean() < 40  # was 80 apart
    assert 80 < low.mean() <= high.mean() < 160  # both moved towards each other (one face each: they meet)


def test_rasterize_finds_each_texel_once():
    owner, bary = TX.rasterize(np.array([[[0, 0], [1, 0], [0, 1]], [[1, 0], [1, 1], [0, 1]]], float), 8, 8,
                               np.array([0, 1]))  # fmt: skip
    assert (owner >= 0).all()  # the two triangles tile the whole square
    assert np.allclose(bary.sum(2), 1, atol=1e-5)


def test_unseen_faces_are_painted_after_levelling():
    a, uv, corner = two_patches()
    a[63, 63] = TX.EMPTY
    uv = np.r_[uv, [[[0.999, 0.001]] * 3]]
    corner = np.r_[corner, [[1, 2, 3]]]
    empty = TX.unseen_faces(a, uv)
    assert empty.tolist() == [False, False, True]
    out, covered, _ = TX.level(a, uv, corner, ~empty)
    colors = np.full((3, 3, 4), 50, np.uint8)
    img, uv2, share = TX.paint_unseen(TX.Image.fromarray(out), uv, colors, empty, int(np.flatnonzero(covered.any(1))[-1]) + 3)
    c = uv2[2].mean(0)
    pixel = np.asarray(img)[round((1 - c[1]) * (img.size[1] - 1)), round(c[0] * (img.size[0] - 1))]
    assert share == 1 / 3 and (pixel == 50).all()

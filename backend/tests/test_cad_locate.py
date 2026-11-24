"""Изборът на детайла от описанието на Florence-2 (app/cad/locate.py), без самите модели: с отговорите им на снимките
на изоставените детайли от сървъра (ключ на лист, острилка, кука, флашка между пръстите)."""

from app.cad import locate as L

W, H = 2400, 1800


def test_the_subject_of_the_caption_is_what_was_photographed():
    assert L.subject("A bunch of keys sitting on top of a piece of paper.") == "A bunch of keys"
    assert L.subject("A close up of a metal hook on a wooden surface.") == "a metal hook"
    assert L.subject("A green and yellow lighter sitting on top of a wooden table.") == "A green and yellow lighter"


def test_the_paper_under_a_key_is_not_the_part():
    found = {"bboxes": [[992, 450, 1417, 1404], [150, 6, 1928, 1550]], "labels": ["keys", "paper"]}
    assert [box for _, box in L.candidates(found, W, H)] == [(992, 450, 1417, 1404)]


def test_the_hand_holding_the_part_is_not_the_part():
    found = {"bboxes": [[6, 493, 854, 1976], [540, 1107, 873, 1309], [6, 495, 852, 1976]],
             "labels": ["A person", "a pill", "their hand"]}  # fmt: skip
    assert [box for _, box in L.candidates(found, 1800, 2400)] == [(540, 1107, 873, 1309)]


def test_a_box_labelled_with_the_whole_caption_is_judged_by_its_size():
    """Florence-2 понякога дава цялото изречение като име на всяка кутия: последната дума ("surface") не казва нищо,
    кутията с размера на кадъра е повърхността, а другата - куката."""
    caption = "A close up of a metal hook on a wooden surface"
    found = {"bboxes": [[0, 1, 1797, 2305], [539, 594, 1250, 1510]], "labels": [caption, caption]}
    assert [box for _, box in L.candidates(found, 1800, 2400)] == [(539, 594, 1250, 1510)]


def test_only_the_surface_found_gives_no_part():
    found = {"bboxes": [[1, 0, 1717, 1797]], "labels": ["of paper"]}
    assert L.candidates(found, W, H) == []

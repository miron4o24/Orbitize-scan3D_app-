"""Работата върху много кадри наведнъж (app/parallel.py): при недостиг на памет за няколко модела наведнъж
всичко се прави още веднъж с един модел, вместо задачата да загуби всички кадри."""

import threading
import time

import pytest

from app import parallel as PAR


def test_out_of_memory_with_several_models_retries_with_one(monkeypatch):
    monkeypatch.setattr(PAR, "MODELS", 3)
    busy = {"now": 0}
    lock = threading.Lock()
    made = []

    def make():
        made.append(1)
        return object()

    def work(model, item):
        with lock:
            busy["now"] += 1
            many = busy["now"] > 1
        try:
            time.sleep(0.02)
            if many:  # като ONNX Runtime на сървъра: няколко модела наведнъж не се побират в паметта
                raise RuntimeError("[ONNXRuntimeError] : 1 : FAIL : bad allocation")
            return item * 2
        finally:
            with lock:
                busy["now"] -= 1

    assert PAR.each_model(make, work, range(8)) == [i * 2 for i in range(8)]


def test_other_errors_are_not_hidden(monkeypatch):
    monkeypatch.setattr(PAR, "MODELS", 3)

    def work(model, item):
        raise ValueError("broken frame")

    with pytest.raises(ValueError):
        PAR.each_model(object, work, range(4))


def test_ai_model_out_of_memory_is_retried_with_one_model_not_one_frame(tmp_path, monkeypatch):
    """Стъпка на AI модела спря с "bad allocation" (няколко модела наведнъж): още веднъж с един модел наведнъж, а не
    модел от един кадър. Друга грешка: както преди - от един кадър."""
    import os

    from app import ai
    from app import pipeline as P

    calls = []

    def from_video(job, work, photos, n, flat, set_state, sh, tune):
        calls.append(os.environ.get("PAR_MODELS"))
        if len(calls) == 1:
            P.note(job, "onnxruntime_pybind11_state.Fail: [ONNXRuntimeError] : 1 : FAIL : bad allocation")
            raise P.Failed("Обработката спря с грешка. Опитайте отново.")
        return {"ai": "single", "photos_used": 35}

    monkeypatch.setattr(ai, "from_video", from_video)
    monkeypatch.delenv("PAR_MODELS", raising=False)
    work = tmp_path / "work"
    work.mkdir()
    report = ai.all_frames(tmp_path, work, tmp_path, 48, True, lambda s: None, lambda s, a: None, None)
    assert report["photos_used"] == 35 and calls == [None, "1"]
    assert "PAR_MODELS" not in os.environ

    def broken(*args):
        raise P.Failed("Твърде малко кадри.")

    monkeypatch.setattr(ai, "from_video", broken)
    assert ai.all_frames(tmp_path, work, tmp_path, 48, True, lambda s: None, lambda s, a: None, None) is None

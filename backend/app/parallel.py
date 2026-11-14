"""Работа върху много кадри наведнъж, на всички нишки на сървъра: същата работа, същият резултат, за по-малко време.
Нишки, а не процеси: масивите се споделят без копиране, а numpy, scipy и ONNX Runtime освобождават заключването на
Python, докато изчисляват. Никога не умножавайте матрици (`@`, np.dot, решаващите функции на np.linalg) в код, който
работи тук: този OpenBLAS срива целия процес на Windows, когато две нишки го правят едновременно (вижте stress.py).
`turned` прави същото като `X @ R.T` без него. AI моделите работят по няколко наведнъж, всеки с няколко свои нишки: на
процесор един кадър на всички ядра губи голяма част от тях в чакане едно на друго."""

import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Iterable

import numpy as np

CPUS = os.cpu_count() or 1
MODELS = int(os.environ.get("PAR_MODELS") or max(1, CPUS // 4))  # 12 нишки (6 ядра): 3 кадъра наведнъж
MODEL_THREADS = max(1, CPUS // 2 // MODELS)  # по 2 на модел: по една на ядро


def each(work: Callable, items: Iterable, workers: int = CPUS) -> list:
    """[work(item) for item in items], наведнъж; в реда на елементите. Грешка в един се хвърля тук."""
    items = list(items)
    if workers <= 1 or len(items) <= 1:
        return [work(item) for item in items]
    with ThreadPoolExecutor(min(workers, len(items))) as pool:
        return list(pool.map(work, items))


def out_of_memory(e: BaseException) -> bool:
    """Грешката е от недостиг на памет (ONNX Runtime на Windows: "bad allocation")."""
    text = str(e).lower()
    return isinstance(e, MemoryError) or any(w in text for w in ("bad allocation", "bad_alloc", "out of memory",
                                                                   "failed to allocate", "not enough memory"))


def each_model(make: Callable, work: Callable, items: Iterable) -> list:
    """[work(model, item) for item in items], по MODELS наведнъж, всяка работна нишка със собствен модел от make():
    едновременните изпълнения на една ONNX сесия делят единствения ѝ набор от нишки. Ако паметта не стига за
    няколко модела наведнъж, всичко още веднъж с един модел, кадър по кадър: по-бавно, но не се губят всички кадри
    (сканиране с 35 разположени кадъра ставаше модел от един кадър)."""
    items = list(items)
    local = threading.local()

    def one(item):
        if not hasattr(local, "model"):
            local.model = make()
        return work(local.model, item)

    try:
        return each(one, items, MODELS)
    except Exception as e:  # noqa: BLE001
        if MODELS <= 1 or not out_of_memory(e):
            raise
        print(f"out of memory with {MODELS} models at once ({e}): again with one", flush=True)
        import gc

        del local
        gc.collect()
        model = make()
        return [work(model, item) for item in items]


def onnx_options():
    """Настройките на ONNX Runtime за модел, използван чрез each_model."""
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = MODEL_THREADS
    options.add_session_config_entry("session.intra_op.allow_spinning", "0")  # чакащите нишки биха заели реда на другите
    return options


def turned(X: np.ndarray, R: np.ndarray) -> np.ndarray:
    """X @ R.T без OpenBLAS: точки (n, 3) по завъртане (3, 3)."""
    return np.einsum("ij,kj->ik", X, R)

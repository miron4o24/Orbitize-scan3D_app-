"""Work on many frames at once, on all of the server's threads: the same work, the same result, in less time.

Threads, not processes: the arrays are shared without copying, and numpy, scipy and ONNX Runtime let go of Python's lock
while they compute.

Never matrix-multiply (`@`, np.dot, np.linalg's solvers) in code that runs here: this OpenBLAS crashes the whole
process on Windows when two threads do it at once (see stress.py). `turned` does what `X @ R.T` does without it.

The AI models run a few at a time, each with a few threads of its own: on a CPU, one frame at a time on all the cores
spends much of them waiting for each other."""

import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Iterable

import numpy as np

CPUS = os.cpu_count() or 1
MODELS = max(1, CPUS // 4)  # 12 threads (6 cores): 3 frames at once
MODEL_THREADS = max(1, CPUS // 2 // MODELS)  # 2 each: one per core


def each(work: Callable, items: Iterable, workers: int = CPUS) -> list:
    """[work(item) for item in items], at once; in the items' order. A failure in one is raised here."""
    items = list(items)
    if workers <= 1 or len(items) <= 1:
        return [work(item) for item in items]
    with ThreadPoolExecutor(min(workers, len(items))) as pool:
        return list(pool.map(work, items))


def each_model(make: Callable, work: Callable, items: Iterable) -> list:
    """[work(model, item) for item in items], MODELS at once, each worker with its own model from make(): concurrent
    runs of one ONNX session share its one pool of threads."""
    local = threading.local()

    def one(item):
        if not hasattr(local, "model"):
            local.model = make()
        return work(local.model, item)

    return each(one, items, MODELS)


def onnx_options():
    """ONNX Runtime's settings for a model used through each_model."""
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = MODEL_THREADS
    options.add_session_config_entry("session.intra_op.allow_spinning", "0")  # waiting threads would take others' turn
    return options


def turned(X: np.ndarray, R: np.ndarray) -> np.ndarray:
    """X @ R.T without OpenBLAS: points (n, 3) by a rotation (3, 3)."""
    return np.einsum("ij,kj->ik", X, R)

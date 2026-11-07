"""Видео с обиколка -> най-рязкият кадър от всяка равна стъпка от движението на камерата, така че кадрите да са
равномерно разпределени около предмета, дори когато потребителят върви с неравно темпо или за момент спира."""

import numpy as np

# Кадри, оценявани за секунда видео: всеки запазен кадър е най-рязкият от около три около него (при 6 в секунда и 120
# запазени кадъра те бяха само два и едно сканиране запази кадри с една шеста от медианната рязкост).
CANDIDATE_FPS = 10
FRAMES_PER_S = 3  # кадри, запазвани за секунда: кръг от 30 с дава 90 кадъра през 4°
# Два кръга в 120 кадъра са през 6°: всяка точка от повърхността е в повече кадри, затова слятата повърхност осреднява
# повече измервания (по-малко шум), а текстурата има по-резки и по-прави изгледи за избор. Цената на съпоставянето
# расте с квадрата на това (80 кадъра: 4 мин на сървъра; 120: около 9), а стъпките за всеки кадър линейно.
MIN_FRAMES, MAX_FRAMES = 20, 120
MIN_SECONDS = 4  # по-кратките видеа се отказват: няма кръг около нищо
THUMB = 256  # квадратен размер за оценяване; сплескването на пропорциите не променя подредбата по рязкост
SMALL = 4  # движението се измерва върху миниатюри THUMB/SMALL, където шумът и размазването почти не се различават
STILL = 0.1  # неподвижното стоене пак брои този дял от средното движение, така че времето никога не се пренебрегва
BATCH = 64


def sharpness(frames: np.ndarray) -> np.ndarray:
    """Дисперсия на лапласиана за всеки кадър, (n, h, w) uint8 -> (n,). Размазването от движение я намалява."""
    f = frames.astype(np.float32)
    lap = f[:, 1:-1, :-2] + f[:, 1:-1, 2:] + f[:, :-2, 1:-1] + f[:, 2:, 1:-1] - 4 * f[:, 1:-1, 1:-1]
    return lap.reshape(len(f), -1).var(axis=1)


def _thumbs(raw: np.ndarray) -> np.ndarray:
    return raw[: len(raw) // THUMB**2 * THUMB**2].reshape(-1, THUMB, THUMB)


def score_thumbs(raw: np.ndarray) -> np.ndarray:
    """Рязкост на всеки сив кадър THUMB x THUMB в суров буфер от байтове, на порции, за да се ограничи паметта."""
    thumbs = _thumbs(raw)
    return np.concatenate([sharpness(thumbs[i : i + BATCH]) for i in range(0, len(thumbs), BATCH)] or [np.zeros(0)])


def motion(raw: np.ndarray) -> np.ndarray:
    """Колко се е променило изображението спрямо предишния кадър (средна абсолютна разлика на малки миниатюри)."""
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
    """Индекс на най-рязкия кандидат във всеки от n участъка с равно движение на камерата (равно време без `moved`).
    Рязко завъртане може да остави участък празен; той се пропуска."""
    if moved is None or moved.sum() <= 0:
        edges = np.linspace(0, len(scores), n + 1).round().astype(int)
    else:
        progress = np.cumsum(moved + STILL * moved.mean())
        edges = np.searchsorted(progress, np.linspace(0, progress[-1], n + 1)[1:-1], side="right")
        edges = np.concatenate([[0], edges, [len(scores)]])
    return [int(a + np.argmax(scores[a:b])) for a, b in zip(edges[:-1], edges[1:]) if b > a]

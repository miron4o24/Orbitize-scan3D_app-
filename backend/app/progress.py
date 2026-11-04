"""Точки, нива, значки и седмични предизвикателства: това, което прави сканирането приятно (GET /me/progress). Точките
са събития в таблицата xp_events (app/db.py), всяко най-много веднъж за потребител, вид и ref, така че нищо не се
брои двойно и нищо спечелено не се губи при изтриване на сканиране: scan: завършено сканиране (ref: номер на
задачата); publish: сканиране, поставено в общността (номер на задачата); liked: харесване от друг на публикацията
(job:liker), при оттегляне точките остават; stress: тест за здравина от един вид върху модел (job:case,
app/stress.py); print: снимка на отпечатан модел от общността (номер на задачата); printed: снимка от друг на
собствена публикация (job:printer); challenge: изпълнено седмично предизвикателство (week:challenge). Завършените
сканирания, публикациите и харесванията вече се пазят в таблиците scans и likes: стават събития при четене на
напредъка (sync), което дава история и на профили отпреди това, с датата на случването. Тестовете за здравина не
оставят друга следа, затова се записват при случването им. Нивата и значките следват от събитията. Трите
предизвикателства за седмицата са еднакви за всички и се избират от POOL според седмицата."""

import random
import threading
import time
from datetime import datetime, timezone

import sqlalchemy as sa
from fastapi import APIRouter, Depends

from . import db, jobs
from .accounts import Principal, need_db, signed_in

router = APIRouter()

POINTS = {"scan": 50, "publish": 30, "liked": 10, "stress": 20, "print": 40, "printed": 20}

LEVELS = [(0, "Новак"), (100, "Любител"), (300, "Откривател"), (600, "Сканиращ"), (1000, "Майстор"), (1600, "Експерт"),
          (2500, "Виртуоз"), (4000, "Легенда")]  # fmt: skip
LEGEND_STEP = 2000  # past the last level, one more "Легенда" level per this many points

# идентификатор, заглавие, описание, какво се брои, колко, икона (Ionicons)
BADGES = [
    ("first_scan", "Първа стъпка", "Завършете първото си сканиране", "scan", 1, "cube-outline"),
    ("scans_10", "Колекционер", "10 завършени сканирания", "scan", 10, "albums-outline"),
    ("scans_50", "Архивар", "50 завършени сканирания", "scan", 50, "library-outline"),
    ("first_post", "На сцената", "Публикувайте модел в общността", "publish", 1, "people-outline"),
    ("posts_10", "Куратор", "10 модела в общността", "publish", 10, "images-outline"),
    ("liked_10", "Харесван", "Получете 10 харесвания", "liked", 10, "heart-outline"),
    ("liked_100", "Звезда", "Получете 100 харесвания", "liked", 100, "star-outline"),
    ("first_print", "От екрана в ръката", "Качете снимка на отпечатан модел от общността", "print", 1, "print-outline"),
    ("printed_10", "Вдъхновител", "10 души отпечатаха ваши модели", "printed", 10, "ribbon-outline"),
    ("first_stress", "Инженер", "Направете стрес тест", "stress", 1, "construct-outline"),
    ("stress_all", "Краш тест", "И четирите теста на един модел", "stress_all", 1, "flash-outline"),
    ("challenge_1", "Предизвикател", "Изпълнете седмично предизвикателство", "challenge", 1, "flag-outline"),
    ("challenge_10", "Неудържим", "10 изпълнени предизвикателства", "challenge", 10, "trophy-outline"),
]

# идентификатор, какво се брои през седмицата, колко, бонус точки, заглавие, икона
POOL = [
    ("scan2", "scan", 2, 100, "Сканирайте 2 предмета", "scan-outline"),
    ("scan3", "scan", 3, 150, "Сканирайте 3 предмета", "scan-outline"),
    ("post1", "publish", 1, 80, "Публикувайте модел в общността", "people-outline"),
    ("post2", "publish", 2, 140, "Публикувайте 2 модела в общността", "people-outline"),
    ("stress2", "stress", 2, 60, "Направете 2 стрес теста", "construct-outline"),
    ("stress3", "stress", 3, 90, "Направете 3 стрес теста", "construct-outline"),
    ("drop1", "stress:drop", 1, 60, "Изпуснете модел в теста за падане", "arrow-down-outline"),
    ("like5", "like_given", 5, 50, "Харесайте 5 модела в общността", "heart-outline"),
    ("liked3", "liked", 3, 80, "Получете 3 харесвания", "heart"),
]
PER_WEEK = 3
WEEK_OFFSET_S = 2 * 3600  # седмиците започват в понеделник в полунощ в България (зимно време: час разлика през лятото)

LABELS = {"scan": "Завършено сканиране", "publish": "Публикуване в общността", "liked": "Харесване от друг",
          "stress": "Стрес тест", "challenge": "Седмично предизвикателство", "print": "Снимка на отпечатък",
          "printed": "Някой отпечата ваш модел"}  # fmt: skip

_lock = threading.Lock()  # един процес на сървъра: синхронизациите са една по една, така че събитие никога не се добавя двойно


def level_of(points: int) -> dict:
    """Нивото за `points`: номер (от 1), име и точките, от които започват то и следващото."""
    n = sum(1 for floor, _ in LEVELS if points >= floor)
    if n < len(LEVELS):
        return {"number": n, "name": LEVELS[n - 1][1], "from": LEVELS[n - 1][0], "to": LEVELS[n][0]}
    extra = (points - LEVELS[-1][0]) // LEGEND_STEP
    start = LEVELS[-1][0] + extra * LEGEND_STEP
    return {"number": n + extra, "name": LEVELS[-1][1], "from": start, "to": start + LEGEND_STEP}


def week_of(ts: float) -> tuple[str, int, int]:
    """Седмицата, съдържаща `ts`: ключът ѝ ("2026-W44"), начало и край (секунди по Unix)."""
    day = int((ts + WEEK_OFFSET_S) // 86400)
    monday = day - (day + 3) % 7  # 1 януари 1970 г. беше четвъртък
    start = monday * 86400 - WEEK_OFFSET_S
    iso = datetime.fromtimestamp(monday * 86400, timezone.utc).isocalendar()
    return f"{iso.year}-W{iso.week:02d}", start, start + 7 * 86400


def challenges_of(week: str) -> list[tuple]:
    """Предизвикателствата за седмицата: PER_WEEK от POOL, всяко брои нещо различно."""
    rng = random.Random(f"orbitize-{week}")
    families = sorted({c[1].split(":")[0] for c in POOL})
    rng.shuffle(families)
    return [rng.choice([c for c in POOL if c[1].split(":")[0] == f]) for f in families[:PER_WEEK]]


# --- Събития --------------------------------------------------------------------------------------------------------


def _add(c, have: set, user_id: str, kind: str, ref: str, points: int, created: float):
    if (kind, ref) in have:
        return
    c.execute(sa.insert(db.xp_events).values(user_id=user_id, kind=kind, ref=ref[:80], points=points,
                                             created=int(created)))  # fmt: skip
    have.add((kind, ref))


def _have(c, user_id: str) -> set:
    return {(r.kind, r.ref) for r in c.execute(sa.select(db.xp_events.c.kind, db.xp_events.c.ref)
                                               .where(db.xp_events.c.user_id == user_id))}  # fmt: skip


def stress_done(user_id: str, job_id: str, case: str):
    with _lock, need_db().begin() as c:
        _add(c, _have(c, user_id), user_id, "stress", f"{job_id}:{case}", POINTS["stress"], time.time())


def sync(user_id: str):
    """Събития за завършени сканирания, публикации и получени харесвания, които още нямат такива; после бонусът за
    всяко предизвикателство, изпълнено тази или миналата седмица (приложението може да не е отваряно оттогава)."""
    with _lock, need_db().begin() as c:
        have = _have(c, user_id)
        for r in c.execute(sa.select(db.scans).where(db.scans.c.user_id == user_id)).all():
            s = jobs.read(r.job_id) or {}
            if s.get("state") == "done":
                finished = (s.get("created") or r.created) + ((s.get("report") or {}).get("seconds") or 0)
                _add(c, have, user_id, "scan", r.job_id, POINTS["scan"], finished)
            if r.published:
                _add(c, have, user_id, "publish", r.job_id, POINTS["publish"], r.published)
        liked = (sa.select(db.likes.c.job_id, db.likes.c.user_id, db.likes.c.created)
                 .join(db.scans, db.scans.c.job_id == db.likes.c.job_id)
                 .where(db.scans.c.user_id == user_id, db.likes.c.user_id != user_id))  # fmt: skip
        for r in c.execute(liked):
            _add(c, have, user_id, "liked", f"{r.job_id}:{r.user_id}", POINTS["liked"], r.created)
        for r in c.execute(sa.select(db.prints.c.job_id, db.prints.c.created).where(db.prints.c.user_id == user_id)):
            _add(c, have, user_id, "print", r.job_id, POINTS["print"], r.created)
        printed = (sa.select(db.prints.c.job_id, db.prints.c.user_id, db.prints.c.created)
                   .join(db.scans, db.scans.c.job_id == db.prints.c.job_id)
                   .where(db.scans.c.user_id == user_id, db.prints.c.user_id != user_id))  # fmt: skip
        for r in c.execute(printed):
            _add(c, have, user_id, "printed", f"{r.job_id}:{r.user_id}", POINTS["printed"], r.created)

        now = time.time()
        for ts in (now, now - 7 * 86400):
            week, start, end = week_of(ts)
            counts = week_counts(c, user_id, start, end)
            for cid, measure, target, bonus, *_ in challenges_of(week):
                if counts.get(measure, 0) >= target:
                    _add(c, have, user_id, "challenge", f"{week}:{cid}", bonus, min(now, end - 1))


def week_counts(c, user_id: str, start: int, end: int) -> dict:
    """Какво броят предизвикателствата между start и end."""
    rows = c.execute(sa.select(db.xp_events.c.kind, db.xp_events.c.ref)
                     .where(db.xp_events.c.user_id == user_id, db.xp_events.c.created >= start,
                            db.xp_events.c.created < end)).all()  # fmt: skip
    out: dict[str, int] = {}
    for r in rows:
        out[r.kind] = out.get(r.kind, 0) + 1
        if r.kind == "stress" and r.ref.endswith(":drop"):
            out["stress:drop"] = out.get("stress:drop", 0) + 1
    given = (sa.select(sa.func.count()).select_from(db.likes)
             .join(db.scans, db.scans.c.job_id == db.likes.c.job_id)
             .where(db.likes.c.user_id == user_id, db.scans.c.user_id != user_id,
                    db.likes.c.created >= start, db.likes.c.created < end))  # fmt: skip
    out["like_given"] = int(c.execute(given).scalar() or 0)
    return out


# --- Напредъкът -----------------------------------------------------------------------------------------------------


def badges(events: list) -> list[dict]:
    by_kind: dict[str, list[int]] = {}
    for e in events:
        by_kind.setdefault(e.kind, []).append(e.created)
    for times in by_kind.values():
        times.sort()
    # Всеки вид тест за здравина на един модел: кога е направен последният вид, за първия модел, получил всички.
    cases: dict[str, dict[str, int]] = {}
    for e in events:
        if e.kind == "stress":
            job, _, case = e.ref.rpartition(":")
            cases.setdefault(job, {})[case] = e.created
    complete = sorted(max(t.values()) for t in cases.values() if len(t) >= 4)
    by_kind["stress_all"] = complete
    out = []
    for bid, title, description, kind, target, icon in BADGES:
        times = by_kind.get(kind, [])
        done = len(times) >= target
        out.append({"id": bid, "title": title, "description": description, "icon": icon, "target": target,
                    "progress": min(len(times), target), "unlocked": done,
                    "unlocked_at": times[target - 1] if done else None})  # fmt: skip
    return out


def progress(user_id: str) -> dict:
    sync(user_id)
    now = time.time()
    week, start, end = week_of(now)
    with need_db().connect() as c:
        events = c.execute(sa.select(db.xp_events).where(db.xp_events.c.user_id == user_id)
                           .order_by(db.xp_events.c.created.desc(), db.xp_events.c.id.desc())).all()  # fmt: skip
        counts = week_counts(c, user_id, start, end)
    points = sum(e.points for e in events)
    done = {e.ref for e in events if e.kind == "challenge"}
    items = [{"id": cid, "title": title, "icon": icon, "target": target, "points": bonus,
              "progress": min(counts.get(measure, 0), target), "done": f"{week}:{cid}" in done}
             for cid, measure, target, bonus, title, icon in challenges_of(week)]  # fmt: skip
    return {
        "points": points,
        "points_week": sum(e.points for e in events if start <= e.created < end),
        "level": level_of(points),
        "badges": badges(events),
        "challenges": {"week": week, "ends": end, "items": items},
        "recent": [{"kind": e.kind, "label": LABELS.get(e.kind, e.kind), "points": e.points, "created": e.created}
                   for e in events[:10]],  # fmt: skip
    }


def points_of(user_ids: list[str]) -> dict[str, int]:
    """Точки по потребител, за показване на нивата на авторите в общността."""
    if not user_ids:
        return {}
    with need_db().connect() as c:
        rows = c.execute(sa.select(db.xp_events.c.user_id, sa.func.sum(db.xp_events.c.points))
                         .where(db.xp_events.c.user_id.in_(set(user_ids)))
                         .group_by(db.xp_events.c.user_id)).all()  # fmt: skip
    return {r[0]: int(r[1] or 0) for r in rows}


@router.get("/me/progress")
def get_progress(p: Principal = Depends(signed_in)):
    return progress(p.user_id)

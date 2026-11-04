"""Points, levels, badges and weekly challenges: what makes scanning feel rewarding (GET /me/progress).

Points are events in the xp_events table (app/db.py), each at most once per user, kind and ref, so nothing counts
twice and nothing earned is lost when a scan is deleted:

- scan:      a finished scan (ref: the job id)
- publish:   a scan put into the community (the job id)
- liked:     a like from someone else on one's post (job:liker); taken back, the points stay
- stress:    a stress test of one kind on one model (job:case, app/stress.py)
- print:     a photo of a community model one printed (the job id)
- printed:   someone else's photo of one's post (job:printer)
- challenge: a weekly challenge done (week:challenge)

Finished scans, posts and likes are already kept in the scans and likes tables: they become events when the
progress is read (sync), which also gives accounts from before this their history, dated when it happened. Stress
tests leave no other trace, so they are recorded as they happen. Levels and badges follow from the events. The
week's three challenges are the same for everyone, picked from POOL by the week."""

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

# id, title, description, what is counted, how many, icon (Ionicons)
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

# id, what is counted in the week, how many, bonus points, title, icon
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
WEEK_OFFSET_S = 2 * 3600  # weeks start on Monday at midnight in Bulgaria (winter time: an hour off in summer)

LABELS = {"scan": "Завършено сканиране", "publish": "Публикуване в общността", "liked": "Харесване от друг",
          "stress": "Стрес тест", "challenge": "Седмично предизвикателство", "print": "Снимка на отпечатък",
          "printed": "Някой отпечата ваш модел"}  # fmt: skip

_lock = threading.Lock()  # one server process: syncs one at a time, so an event is never added twice


def level_of(points: int) -> dict:
    """The level for `points`: number (from 1), name, and the points where it starts and the next one starts."""
    n = sum(1 for floor, _ in LEVELS if points >= floor)
    if n < len(LEVELS):
        return {"number": n, "name": LEVELS[n - 1][1], "from": LEVELS[n - 1][0], "to": LEVELS[n][0]}
    extra = (points - LEVELS[-1][0]) // LEGEND_STEP
    start = LEVELS[-1][0] + extra * LEGEND_STEP
    return {"number": n + extra, "name": LEVELS[-1][1], "from": start, "to": start + LEGEND_STEP}


def week_of(ts: float) -> tuple[str, int, int]:
    """The week holding `ts`: its key ("2026-W44"), start and end (Unix seconds)."""
    day = int((ts + WEEK_OFFSET_S) // 86400)
    monday = day - (day + 3) % 7  # 1 January 1970 was a Thursday
    start = monday * 86400 - WEEK_OFFSET_S
    iso = datetime.fromtimestamp(monday * 86400, timezone.utc).isocalendar()
    return f"{iso.year}-W{iso.week:02d}", start, start + 7 * 86400


def challenges_of(week: str) -> list[tuple]:
    """The week's challenges: PER_WEEK of POOL, each counting something different."""
    rng = random.Random(f"scanly-{week}")
    families = sorted({c[1].split(":")[0] for c in POOL})
    rng.shuffle(families)
    return [rng.choice([c for c in POOL if c[1].split(":")[0] == f]) for f in families[:PER_WEEK]]


# --- Events -----------------------------------------------------------------------------------------------------


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
    """Events for finished scans, posts and likes received that have none yet; then the bonus of each challenge done
    this week or last week (the app may not have been opened since)."""
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
    """What the challenges count, between start and end."""
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


# --- The progress -----------------------------------------------------------------------------------------------


def badges(events: list) -> list[dict]:
    by_kind: dict[str, list[int]] = {}
    for e in events:
        by_kind.setdefault(e.kind, []).append(e.created)
    for times in by_kind.values():
        times.sort()
    # Every kind of stress test on one model: when its last kind was done, for the first model to get them all.
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
    """Points per user, for showing authors' levels in the community."""
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

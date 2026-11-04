"""Точки, нива, значки и седмични предизвикателства (app/progress.py)."""

import time

# Първо: задава API ключа, който приложението чете при импортиране.
from tests.test_accounts import client, finished_scan, sign_in  # noqa: F401, I001 (client: fixture)
from tests.test_community import setup_two
from app import db
from app import progress as P


def test_levels():
    assert P.level_of(0) == {"number": 1, "name": "Новак", "from": 0, "to": 100}
    assert P.level_of(99)["number"] == 1 and P.level_of(100)["number"] == 2
    assert P.level_of(4000) == {"number": 8, "name": "Легенда", "from": 4000, "to": 6000}
    assert P.level_of(6500)["number"] == 9


def test_weeks_start_on_monday_and_challenges_are_the_same_for_everyone():
    week, start, end = P.week_of(time.mktime((2026, 10, 28, 12, 0, 0, 0, 0, -1)))  # сряда
    assert week == "2026-W44" and end - start == 7 * 86400
    assert time.gmtime(start + P.WEEK_OFFSET_S).tm_wday == 0  # понеделник в полунощ там
    picked = P.challenges_of(week)
    assert picked == P.challenges_of(week) and len(picked) == P.PER_WEEK
    assert len({c[1].split(":")[0] for c in picked}) == P.PER_WEEK  # всяко брои нещо различно


def test_points_from_scans_posts_and_likes(client):
    a, alice, b, _ = setup_two(client)
    first = client.get("/me/progress", headers=a).json()
    assert first["points"] == 0 and first["level"]["number"] == 1 and not any(x["unlocked"] for x in first["badges"])
    assert len(first["challenges"]["items"]) == P.PER_WEEK

    job = finished_scan(client, a)
    got = client.get("/me/progress", headers=a).json()
    assert got["points"] == P.POINTS["scan"]  # по-старо сканиране се брои при четене на напредъка
    badge = {x["id"]: x for x in got["badges"]}
    assert badge["first_scan"]["unlocked"] and badge["first_scan"]["unlocked_at"]
    assert badge["scans_10"]["progress"] == 1 and not badge["scans_10"]["unlocked"]

    client.post(f"/me/scans/{job}/publish", json={"title": "Чаша"}, headers=a)
    client.put(f"/community/{job}/like", headers=b)
    client.put(f"/community/{job}/like", headers=a)  # собственото харесване не носи нищо
    client.delete(f"/community/{job}/like", headers=b)
    client.put(f"/community/{job}/like", headers=b)  # повторното харесване не носи нищо повече
    got = client.get("/me/progress", headers=a).json()
    base = P.POINTS["scan"] + P.POINTS["publish"] + P.POINTS["liked"]
    bonus = got["points"] - base  # предизвикателствата от тази седмица, които случайно са завършени с тези
    done = sum(c["points"] for c in got["challenges"]["items"] if c["done"])
    assert bonus == done and got["points_week"] == got["points"]
    assert client.get("/me/progress", headers=a).json()["points"] == got["points"]  # прочетено отново: същото

    post = client.get(f"/community/{job}", headers=b).json()
    assert post["author"]["level"] == P.level_of(got["points"])["number"]

    assert client.get("/me/progress").status_code == 401
    assert client.delete("/me", headers=a).status_code == 200
    with db.engine().connect() as c:
        assert c.execute(db.xp_events.select().where(db.xp_events.c.user_id == alice["id"])).first() is None

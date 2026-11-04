"""Общността (app/community.py), със същата настройка на SQLite като тестовете за профили."""

from tests.test_accounts import ADMIN, CLIENT_ID, client, finished_scan, sign_in  # noqa: F401 (client: fixture)


def setup_two(client):
    client.put("/admin/settings", json={"google_client_id": CLIENT_ID}, headers=ADMIN)
    a, alice = sign_in(client, "g1", "a@example.com")
    b, bob = sign_in(client, "g2", "b@example.com")
    return a, alice, b, bob


def test_post_like_and_take_back(client):
    a, alice, b, _ = setup_two(client)
    job = finished_scan(client, a)
    assert client.get("/community", headers=b).json() == {"posts": [], "next": None}
    assert client.post(f"/me/scans/{job}/publish", json={"title": "x"}, headers=b).status_code == 404  # не е на Bob
    assert client.post(f"/me/scans/{job}/publish", json={"title": " "}, headers=a).status_code == 400
    r = client.post(f"/me/scans/{job}/publish", json={"title": "  Чаша ", "longest_mm": 82}, headers=a)
    assert r.status_code == 200, r.text
    post = r.json()
    assert post["title"] == "Чаша" and post["longest_mm"] == 82 and post["mine"] and post["likes"] == 0
    assert post["author"]["name"] == "User g1" and post["dimensions_mm"] == [25, 50, 100]

    feed = client.get("/community", headers=b).json()["posts"]
    assert [p["id"] for p in feed] == [job] and not feed[0]["mine"]
    assert client.get("/community").status_code == 401  # само за влезли потребители
    assert client.get(f"/community/{job}/thumb.jpg", headers=b).status_code == 200
    assert client.get(f"/community/{job}/model.glb", headers=b).content[:4] == b"glTF"
    assert client.get("/me/scans", headers=a).json()[0]["published"]
    assert client.get("/me", headers=a).json()["stats"]["published"] == 1

    assert client.put(f"/community/{job}/like", headers=b).json() == {"likes": 1, "liked": True}
    assert client.put(f"/community/{job}/like", headers=b).json() == {"likes": 1, "liked": True}  # веднъж на човек
    assert client.get(f"/community/{job}", headers=a).json()["likes"] == 1
    assert client.delete(f"/community/{job}/like", headers=b).json() == {"likes": 0, "liked": False}

    # Кадрите на автора остават лични: общността показва само модела.
    assert client.get(f"/jobs/{job}/frames.zip", headers=b).status_code == 404

    client.put(f"/community/{job}/like", headers=b)
    assert client.delete(f"/me/scans/{job}/publish", headers=a).status_code == 200
    assert client.get("/community", headers=b).json()["posts"] == []
    assert client.get(f"/community/{job}/model.glb", headers=b).status_code == 404
    again = client.post(f"/me/scans/{job}/publish", json={"title": "Чаша"}, headers=a).json()
    assert again["likes"] == 0  # при повторно публикуване започва отначало


def test_reports_hide_a_post_until_the_admin_looks(client):
    a, _, b, _ = setup_two(client)
    job = finished_scan(client, a)
    client.post(f"/me/scans/{job}/publish", json={"title": "Модел"}, headers=a)
    assert client.post(f"/community/{job}/report", json={}, headers=a).status_code == 400  # собствена публикация
    others = [b] + [sign_in(client, f"g{i}", f"u{i}@example.com")[0] for i in (3, 4)]
    for h in others:
        assert client.post(f"/community/{job}/report", json={"reason": "spam"}, headers=h).status_code == 200
    assert client.get("/community", headers=b).json()["posts"] == []
    assert len(client.get("/community", headers=a).json()["posts"]) == 1  # авторът продължава да я вижда
    listed = client.get("/admin/community/reports", headers=ADMIN).json()
    assert listed[0]["id"] == job and listed[0]["reports"] == 3 and listed[0]["hidden"]
    assert client.get("/admin/community/reports", headers=b).status_code == 403
    client.post(f"/admin/community/{job}/keep", headers=ADMIN)
    assert len(client.get("/community", headers=b).json()["posts"]) == 1
    client.delete(f"/admin/community/{job}", headers=ADMIN)
    assert client.get("/community", headers=a).json()["posts"] == []


def test_deleting_the_scan_or_account_removes_the_post(client):
    a, _, b, _ = setup_two(client)
    job = finished_scan(client, a)
    client.post(f"/me/scans/{job}/publish", json={"title": "Модел"}, headers=a)
    client.put(f"/community/{job}/like", headers=b)
    assert client.delete(f"/jobs/{job}", headers=a).status_code == 200
    assert client.get("/community", headers=b).json()["posts"] == []

    job = finished_scan(client, b)
    client.post(f"/me/scans/{job}/publish", json={"title": "Друг"}, headers=b)
    client.put(f"/community/{job}/like", headers=a)
    assert client.delete("/me", headers=b).status_code == 200
    assert client.get("/community", headers=a).json()["posts"] == []

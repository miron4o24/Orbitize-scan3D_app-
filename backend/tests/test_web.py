"""Уебсайтът (backend/web), сервиран от същия сървър като API-то: страниците и файловете идват от папката, а всеки път
на API-то остава на API-то (сайтът е монтиран последен)."""

from tests.test_accounts import client  # noqa: F401 (client: fixture)


def test_site_pages_and_assets(client):
    for path in ("/", "/index.html", "/login.html", "/engineer.html", "/admin.html", "/scans.html"):
        r = client.get(path)
        assert r.status_code == 200 and r.headers["content-type"].startswith("text/html"), path
        assert "frame-ancestors 'none'" in r.headers["content-security-policy"]
        assert r.headers["cache-control"] == "no-cache"
    css = client.get("/assets/style.css?v=30")
    assert css.status_code == 200 and "immutable" in css.headers["cache-control"]
    assert client.get("/assets/config.js").headers["cache-control"] == "no-cache"  # настройките се виждат веднага


def test_api_is_not_shadowed_by_the_site(client):
    assert client.get("/health").json()["ok"] is True
    assert client.get("/me").status_code == 401
    assert client.post("/auth/login", json={"email": "x@example.com", "password": "nope123"}).status_code in (401, 503)


def test_unknown_page_is_the_sites_404(client):
    r = client.get("/no-such-page.html")
    assert r.status_code == 404 and "<html" in r.text.lower() and "не съществува" in r.text


def test_no_secrets_in_the_public_settings(client):
    text = client.get("/assets/config.js").text
    assert "API_TOKEN=" not in text and "SMTP" not in text and "apiUrl: \"\"" in text

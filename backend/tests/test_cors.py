"""Уебсайтът (website/, на IIS на собствен адрес) вика API-то от браузъра: CORS само за неговите адреси."""

from tests.test_accounts import client  # noqa: F401 (client: fixture)

SITE = "https://orbitize.bg046.com"


def test_the_website_may_call_the_api(client):
    r = client.options("/me", headers={"Origin": SITE, "Access-Control-Request-Method": "GET",
                                       "Access-Control-Request-Headers": "authorization"})  # fmt: skip
    assert r.status_code == 200 and r.headers["access-control-allow-origin"] == SITE
    assert "authorization" in r.headers["access-control-allow-headers"].lower()
    r = client.options("/jobs/x/photos/top.jpg", headers={"Origin": SITE, "Access-Control-Request-Method": "PUT"})
    assert r.headers["access-control-allow-origin"] == SITE
    assert client.get("/health", headers={"Origin": SITE}).headers["access-control-allow-origin"] == SITE


def test_other_sites_may_not(client):
    r = client.options("/me", headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "GET"})
    assert "access-control-allow-origin" not in r.headers
    # Същият адрес без HTTPS също не: само сайтът с HTTPS говори със сървъра.
    r = client.options("/me", headers={"Origin": "http://orbitize.bg046.com", "Access-Control-Request-Method": "GET"})
    assert "access-control-allow-origin" not in r.headers
    assert "access-control-allow-origin" not in client.get("/health", headers={"Origin": "https://evil.example"}).headers

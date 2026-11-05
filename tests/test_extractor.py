import pytest
import requests

from config import Settings
from core.extractor import BlockedURLError, ExtractionError, extract_from_url, extract_text_from_html, validate_url


def public_resolver(host):
    return ["93.184.215.14"]


# ---------------------------------------------------------------- URL validation

@pytest.mark.parametrize(
    "url",
    [
        "http://localhost/privacy",
        "http://127.0.0.1/privacy",
        "http://10.0.0.5/privacy",
        "http://192.168.1.1/",
        "http://172.16.3.4/",
        "http://169.254.169.254/latest/meta-data/",  # cloud metadata
        "http://[::1]/",
        "http://[::ffff:127.0.0.1]/",
        "http://0.0.0.0/",
        "http://printer.local/",
        "ftp://example.com/privacy",
        "file:///etc/passwd",
        "http://user:pass@example.com/",
        "http://example.com:8080/",
    ],
)
def test_blocked_urls(url):
    with pytest.raises(BlockedURLError):
        validate_url(url, resolver=public_resolver)


def test_hostname_resolving_to_private_ip_is_blocked():
    with pytest.raises(BlockedURLError):
        validate_url("https://sneaky.example.com/", resolver=lambda h: ["10.1.2.3"])


def test_mixed_resolution_is_blocked():
    with pytest.raises(BlockedURLError):
        validate_url("https://sneaky.example.com/", resolver=lambda h: ["93.184.215.14", "127.0.0.1"])


def test_public_url_allowed_and_scheme_added():
    assert validate_url("example.com/privacy", resolver=public_resolver) == "https://example.com/privacy"


def test_empty_url():
    with pytest.raises(ExtractionError):
        validate_url("   ")


# ---------------------------------------------------------------- fetching

class FakeResponse:
    def __init__(self, status=200, body=b"", headers=None):
        self.status_code = status
        self._body = body
        self.headers = headers or {"Content-Type": "text/html; charset=utf-8"}
        self.encoding = "utf-8"
        self.apparent_encoding = "utf-8"

    @property
    def is_redirect(self):
        return self.status_code in (301, 302, 303, 307, 308) and "Location" in self.headers

    def iter_content(self, chunk_size=1):
        for i in range(0, len(self._body), chunk_size):
            yield self._body[i:i + chunk_size]

    def close(self):
        pass


class FakeSession:
    def __init__(self, routes):
        self.routes = routes
        self.requested = []

    def get(self, url, **kwargs):
        assert kwargs["allow_redirects"] is False, "redirects must be validated manually"
        self.requested.append(url)
        route = self.routes[url]
        if isinstance(route, Exception):
            raise route
        return route


POLICY_HTML = b"""<html><head><title>Acme Privacy Policy</title><script>alert('x')</script></head><body>
<nav class="site-nav"><a href="/">Home</a> <a href="/pricing">Pricing</a></nav>
<div id="cookie-banner">We use cookies. Accept all?</div>
<main><h1>Privacy Policy</h1>
<h2>Information We Collect</h2><p>We collect your name, email address and device identifiers when you use the service.</p>
<ul><li>Account details you provide</li><li>Usage data collected automatically</li></ul>
<h2>How We Share Information</h2><p>We share information with service providers who help us operate the service and with advertising partners.</p>
<p>""" + b"Additional detail about our practices. " * 40 + b"""</p></main>
<footer>Copyright Acme. Terms. Sitemap.</footer></body></html>"""


def test_html_extraction_removes_chrome_and_keeps_headings():
    text, title = extract_text_from_html(POLICY_HTML.decode())
    assert title == "Acme Privacy Policy"
    assert "## Information We Collect" in text
    assert "- Account details you provide" in text
    assert "alert(" not in text
    assert "Pricing" not in text
    assert "Accept all" not in text
    assert "Sitemap" not in text


def test_fetch_follows_safe_redirect():
    session = FakeSession({
        "https://acme.example/privacy": FakeResponse(301, headers={"Location": "/legal/privacy"}),
        "https://acme.example/legal/privacy": FakeResponse(200, POLICY_HTML),
    })
    policy = extract_from_url("https://acme.example/privacy", session=session, resolver=public_resolver)
    assert policy.source_url == "https://acme.example/legal/privacy"
    assert "Information We Collect" in policy.text


def test_redirect_to_internal_address_is_blocked():
    session = FakeSession({
        "https://acme.example/privacy": FakeResponse(302, headers={"Location": "http://169.254.169.254/latest"}),
    })
    with pytest.raises(BlockedURLError):
        extract_from_url("https://acme.example/privacy", session=session, resolver=public_resolver)
    assert session.requested == ["https://acme.example/privacy"]


def test_too_many_redirects():
    session = FakeSession({"https://acme.example/a": FakeResponse(302, headers={"Location": "/a"})})
    with pytest.raises(ExtractionError, match="redirected too many"):
        extract_from_url("https://acme.example/a", session=session, resolver=public_resolver)


def test_response_size_limit():
    big = FakeResponse(200, b"<p>" + b"x" * 5000 + b"</p>")
    session = FakeSession({"https://acme.example/big": big})
    with pytest.raises(ExtractionError, match="too large"):
        extract_from_url("https://acme.example/big", settings=Settings(fetch_max_bytes=1000), session=session,
                         resolver=public_resolver)


def test_unsupported_content_type():
    pdf = FakeResponse(200, b"%PDF", headers={"Content-Type": "application/pdf"})
    with pytest.raises(ExtractionError, match="Unsupported content type"):
        extract_from_url("https://acme.example/p.pdf", session=FakeSession({"https://acme.example/p.pdf": pdf}),
                         resolver=public_resolver)


def test_http_error_and_network_error_are_friendly():
    session = FakeSession({
        "https://acme.example/404": FakeResponse(404),
        "https://acme.example/down": requests.ConnectionError("boom"),
        "https://acme.example/slow": requests.Timeout("slow"),
    })
    with pytest.raises(ExtractionError, match="HTTP 404"):
        extract_from_url("https://acme.example/404", session=session, resolver=public_resolver)
    with pytest.raises(ExtractionError, match="could not be reached"):
        extract_from_url("https://acme.example/down", session=session, resolver=public_resolver)
    with pytest.raises(ExtractionError, match="too long"):
        extract_from_url("https://acme.example/slow", session=session, resolver=public_resolver)


def test_javascript_only_page_reports_no_text():
    js_only = FakeResponse(200, b"<html><body><div id='root'></div><script>render()</script></body></html>")
    with pytest.raises(ExtractionError, match="No readable text"):
        extract_from_url("https://acme.example/spa", session=FakeSession({"https://acme.example/spa": js_only}),
                         resolver=public_resolver)

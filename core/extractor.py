"""Safe fetching and text extraction for privacy policy pages.

Security posture (SSRF mitigation):
* Only http/https URLs, no embedded credentials, standard ports only.
* Every hostname is resolved and *all* resolved addresses must be public
  (no loopback, private, link-local, multicast, reserved or unspecified).
* Redirects are followed manually so each hop is re-validated.
* Responses are streamed with a hard byte cap and a timeout.
* Only HTML / plain text content types are accepted.
* JavaScript from the fetched page is never executed — we only parse markup.

Known limitation: a DNS answer could change between validation and connection
(DNS rebinding). For a local/hackathon deployment this residual risk is
documented in the README rather than solved with a custom transport.
"""

from __future__ import annotations

import ipaddress
import re
import socket
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup, Comment, Tag

from config import Settings, get_settings

USER_AGENT = "PrivacyLens/1.0 (+privacy policy reader; educational project)"
ALLOWED_PORTS = {None, 80, 443}
ALLOWED_CONTENT_TYPES = ("text/html", "application/xhtml+xml", "text/plain")

# Elements that never contain policy text.
_STRIP_TAGS = [
    "script", "style", "noscript", "template", "iframe", "svg", "canvas", "form",
    "button", "input", "select", "nav", "footer", "header", "aside", "dialog", "picture", "video", "audio",
]
# id/class fragments typical of banners, menus and other chrome.
_BOILERPLATE_RE = re.compile(
    r"(cookie|consent|gdpr-banner|onetrust|cc-banner|newsletter|subscribe|breadcrumb|"
    r"site-nav|navbar|nav-bar|menu|sidebar|skip-link|share|social|modal|popup|advert|banner)",
    re.I,
)
_BLOCK_TAGS = ["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "td", "th", "dt", "dd", "pre", "blockquote"]


class ExtractionError(Exception):
    """User-facing error raised when a policy cannot be fetched or parsed."""


class BlockedURLError(ExtractionError):
    """Raised when a URL targets a disallowed destination."""


@dataclass
class ExtractedPolicy:
    text: str
    title: str | None
    source_url: str | None


# ---------------------------------------------------------------------------
# URL validation
# ---------------------------------------------------------------------------

def _is_public_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    return not (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
        or not ip.is_global
    )


def _resolve(host: str) -> list[str]:
    try:
        infos = socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        raise ExtractionError(f"Could not resolve the website address '{host}'.") from exc
    return sorted({info[4][0] for info in infos})


def validate_url(url: str, resolver=_resolve) -> str:
    """Return a normalised URL or raise BlockedURLError / ExtractionError."""
    url = (url or "").strip()
    if not url:
        raise ExtractionError("Please enter a URL.")
    if "://" not in url:
        url = "https://" + url
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise BlockedURLError("Only http:// and https:// URLs are supported.")
    if parsed.username or parsed.password:
        raise BlockedURLError("URLs containing credentials are not allowed.")
    host = parsed.hostname
    if not host:
        raise ExtractionError("That doesn't look like a valid URL.")
    try:
        port = parsed.port
    except ValueError as exc:
        raise ExtractionError("That URL has an invalid port.") from exc
    if port not in ALLOWED_PORTS:
        raise BlockedURLError("Only standard web ports (80 and 443) are allowed.")
    lowered = host.lower().rstrip(".")
    if lowered == "localhost" or lowered.endswith((".localhost", ".local", ".internal")):
        raise BlockedURLError("Local and internal addresses cannot be fetched.")

    try:  # Literal IP in the URL
        addresses = [str(ipaddress.ip_address(host.strip("[]")))]
    except ValueError:
        addresses = resolver(host)
    if not addresses:
        raise ExtractionError(f"Could not resolve the website address '{host}'.")
    for addr in addresses:
        if not _is_public_ip(ipaddress.ip_address(addr.split("%")[0])):
            raise BlockedURLError("That address points to a private or internal network and cannot be fetched.")
    return parsed.geturl()


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------

def fetch_html(url: str, settings: Settings | None = None, session: requests.Session | None = None,
               resolver=_resolve) -> tuple[str, str, str]:
    """Fetch a URL safely. Returns (body, final_url, content_type)."""
    settings = settings or get_settings()
    session = session or requests.Session()
    current = validate_url(url, resolver=resolver)

    for _ in range(settings.fetch_max_redirects + 1):
        try:
            resp = session.get(
                current,
                headers={"User-Agent": USER_AGENT, "Accept": "text/html,text/plain;q=0.9"},
                timeout=settings.fetch_timeout_s,
                allow_redirects=False,
                stream=True,
            )
        except requests.Timeout as exc:
            raise ExtractionError("The website took too long to respond.") from exc
        except requests.RequestException as exc:
            raise ExtractionError("The website could not be reached. Check the URL and try again.") from exc

        try:
            if resp.is_redirect or resp.status_code in (301, 302, 303, 307, 308):
                location = resp.headers.get("Location")
                if not location:
                    raise ExtractionError("The website returned a redirect without a destination.")
                current = validate_url(urljoin(current, location), resolver=resolver)
                continue

            if resp.status_code in (401, 403):
                raise ExtractionError(
                    "The website refused automated access (HTTP %d). Try pasting the policy text instead."
                    % resp.status_code
                )
            if resp.status_code >= 400:
                raise ExtractionError(f"The website returned an error (HTTP {resp.status_code}).")

            content_type = (resp.headers.get("Content-Type") or "").split(";")[0].strip().lower()
            if content_type and not content_type.startswith(ALLOWED_CONTENT_TYPES):
                raise ExtractionError(
                    f"Unsupported content type '{content_type}'. PrivacyLens reads HTML or plain-text pages."
                )
            declared = resp.headers.get("Content-Length")
            if declared and declared.isdigit() and int(declared) > settings.fetch_max_bytes:
                raise ExtractionError("That page is too large to process.")

            body = bytearray()
            for chunk in resp.iter_content(chunk_size=65536):
                body.extend(chunk)
                if len(body) > settings.fetch_max_bytes:
                    raise ExtractionError("That page is too large to process.")
            encoding = resp.encoding or "utf-8"
            if encoding.lower() == "iso-8859-1" and content_type != "text/plain":
                encoding = resp.apparent_encoding or "utf-8"
            text = bytes(body).decode(encoding, errors="replace")
            return text, current, content_type or "text/html"
        finally:
            resp.close()

    raise ExtractionError("The website redirected too many times.")


# ---------------------------------------------------------------------------
# HTML → structured text
# ---------------------------------------------------------------------------

def _looks_like_boilerplate(tag: Tag) -> bool:
    if not isinstance(tag, Tag) or tag.attrs is None:
        return False
    if tag.name in ("html", "body", "main", "article"):
        return False
    attrs = " ".join(
        [tag.get("id") or "", " ".join(tag.get("class") or []), tag.get("role") or "", tag.get("aria-label") or ""]
    )
    if tag.get("role") in ("navigation", "banner", "contentinfo", "dialog", "alertdialog"):
        return True
    if tag.get("aria-hidden") == "true" or tag.get("hidden") is not None:
        return True
    return bool(attrs.strip()) and bool(_BOILERPLATE_RE.search(attrs))


def _pick_root(soup: BeautifulSoup) -> Tag:
    for selector in ("main", "article", "[role=main]", "#content", ".content", "#main"):
        candidates = soup.select(selector)
        if candidates:
            best = max(candidates, key=lambda t: len(t.get_text(" ", strip=True)))
            if len(best.get_text(" ", strip=True).split()) >= 150:
                return best
    return soup.body or soup


def extract_text_from_html(html: str) -> tuple[str, str | None]:
    """Convert HTML into heading-preserving plain text. Returns (text, title)."""
    soup = BeautifulSoup(html, "html.parser")
    title = None
    if soup.title and soup.title.string:
        title = soup.title.string.strip()

    for comment in soup.find_all(string=lambda s: isinstance(s, Comment)):
        comment.extract()
    for tag in soup.find_all(_STRIP_TAGS):
        tag.decompose()
    for tag in list(soup.find_all(True)):
        if getattr(tag, "decomposed", False):
            continue
        if _looks_like_boilerplate(tag):
            tag.decompose()

    root = _pick_root(soup)
    if title is None:
        h1 = root.find("h1")
        if h1:
            title = h1.get_text(" ", strip=True) or None

    lines: list[str] = []
    seen_blocks: set[int] = set()
    for el in root.find_all(_BLOCK_TAGS):
        # Skip blocks nested inside an already-emitted block (e.g. <p> in <li>).
        if any(id(parent) in seen_blocks for parent in el.parents):
            continue
        seen_blocks.add(id(el))
        text = re.sub(r"\s+", " ", el.get_text(" ", strip=True)).strip()
        if not text:
            continue
        if el.name in ("h1", "h2", "h3", "h4", "h5", "h6"):
            level = int(el.name[1])
            lines.append("")
            lines.append("#" * min(level, 3) + " " + text)
        elif el.name == "li":
            lines.append("- " + text)
        else:
            lines.append(text)

    if len(" ".join(lines).split()) < 50:  # unusual markup: fall back to all text
        raw = root.get_text("\n", strip=True)
        lines = [re.sub(r"\s+", " ", ln).strip() for ln in raw.splitlines() if ln.strip()]

    return "\n".join(lines).strip(), title


def extract_from_url(url: str, settings: Settings | None = None, session: requests.Session | None = None,
                     resolver=_resolve) -> ExtractedPolicy:
    body, final_url, content_type = fetch_html(url, settings=settings, session=session, resolver=resolver)
    if content_type == "text/plain":
        text, title = body, None
    else:
        text, title = extract_text_from_html(body)
    if not text.strip():
        raise ExtractionError(
            "No readable text was found on that page. It may require JavaScript — try pasting the policy text instead."
        )
    return ExtractedPolicy(text=text, title=title, source_url=final_url)

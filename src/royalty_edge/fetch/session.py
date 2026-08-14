"""Authenticated session construction.

The index endpoint answered while you were logged in, and the detail payload
contains offer ladders and buyer ids that are certainly not public. Assume
every endpoint here needs a session and design for it rather than discovering
mid-crawl that 2,000 detail fetches all returned a login redirect.

Cookie handling is deliberately manual. There is no OAuth or API key on this
platform, so the only way in is the session cookie your browser already holds.
Export it once, keep it out of git, and re-export when it expires.

To export:
  DevTools -> Application -> Cookies -> https://auctions.royaltyexchange.com
  Copy the whole cookie header, or use a "copy as cURL" on any XHR and pull
  the -H 'cookie: ...' value out of it. Save to secrets/cookie.txt.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path

import requests

log = logging.getLogger(__name__)

DEFAULT_COOKIE_PATH = Path("secrets/cookie.txt")
BASE = "https://auctions.royaltyexchange.com"


def load_cookie(path: str | Path | None = None) -> str | None:
    """Cookie string from file, or the RE_COOKIE env var, or None."""
    env = os.environ.get("RE_COOKIE")
    if env:
        return env.strip()
    p = Path(path) if path else DEFAULT_COOKIE_PATH
    if p.exists():
        raw = p.read_text().strip()
        # tolerate a pasted "cookie: a=b; c=d" line
        return re.sub(r"^\s*cookie:\s*", "", raw, flags=re.I)
    return None


def build_session(cookie: str | None = None, *, referer: str = BASE + "/") -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": referer,
        "X-Requested-With": "XMLHttpRequest",
    })
    if cookie:
        s.headers["Cookie"] = cookie
        m = re.search(r"csrftoken=([^;]+)", cookie)
        if m:
            # Django REST convention; harmless if the platform ignores it
            s.headers["X-CSRFToken"] = m.group(1)
    return s


def looks_like_auth_failure(status: int, body: bytes) -> bool:
    """A login redirect that returns 200 with an HTML shell is the dangerous
    case: it looks like success and parses as garbage. Catch it explicitly."""
    if status in (401, 403):
        return True
    head = body[:400].lstrip().lower()
    return head.startswith(b"<!doctype html") or head.startswith(b"<html")

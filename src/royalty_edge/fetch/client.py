"""Content-addressed, resumable HTTP client.

Design rules, in priority order:

1. Never fetch the same URL twice in a run, and never refetch across runs
   unless explicitly asked. Bandwidth is not the constraint; goodwill is.
2. Persist raw bytes before any parsing. Parsers will be wrong on the first
   three attempts. Refetching to fix a parser bug is the failure mode this
   module exists to prevent.
3. Respect robots.txt and a conservative rate limit by default. If the
   platform publishes an API or will hand over an export, this module
   should end up mostly unused -- that is the good outcome.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import logging
import time
import urllib.robotparser
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urljoin, urlparse

import requests

log = logging.getLogger(__name__)

USER_AGENT = (
    "royalty-edge-research/0.1 (personal investment research; "
    "contact: SET_YOUR_EMAIL_HERE)"
)


@dataclass
class FetchConfig:
    landing_dir: Path
    min_interval_s: float = 3.0          # one request per 3s. Slower than you want.
    timeout_s: float = 30.0
    max_retries: int = 3
    backoff_base_s: float = 5.0
    respect_robots: bool = True
    user_agent: str = USER_AGENT
    extra_headers: Mapping[str, str] = field(default_factory=dict)


@dataclass
class FetchResult:
    url: str
    fetched_at: datetime
    http_status: int
    content_type: str | None
    content_sha256: str
    content_bytes: int
    payload_path: str
    from_cache: bool


class PoliteFetcher:
    def __init__(self, cfg: FetchConfig, session: requests.Session | None = None):
        self.cfg = cfg
        self.cfg.landing_dir.mkdir(parents=True, exist_ok=True)
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": cfg.user_agent, **cfg.extra_headers})
        self._last_request_at = 0.0
        self._robots: dict[str, urllib.robotparser.RobotFileParser] = {}

    # -- robots -------------------------------------------------------
    # We fetch robots.txt ourselves with our own session so the server sees
    # the same User-Agent as the actual requests. The stdlib RobotFileParser
    # uses urllib with its own UA, which can get a 403 or redirect and then
    # silently treats the entire site as Disallow: / as a safe fallback.

    def _robots_for(self, url: str) -> urllib.robotparser.RobotFileParser:
        root = "{0.scheme}://{0.netloc}".format(urlparse(url))
        if root not in self._robots:
            rp = urllib.robotparser.RobotFileParser()
            robots_url = urljoin(root, "/robots.txt")
            try:
                resp = self.session.get(robots_url, timeout=self.cfg.timeout_s)
                if resp.status_code == 200:
                    rp.set_url(robots_url)
                    rp.parse(resp.text.splitlines())
                elif resp.status_code in (401, 403, 404):
                    # No robots.txt or gated behind auth: treat as allow-all
                    log.debug("robots.txt %s for %s; treating as allow-all",
                              resp.status_code, root)
                else:
                    log.warning("robots.txt unexpected status %s for %s; allowing",
                                resp.status_code, root)
            except Exception as exc:
                log.warning("robots.txt fetch failed for %s (%s); allowing", root, exc)
            self._robots[root] = rp
        return self._robots[root]

    def allowed(self, url: str) -> bool:
        if not self.cfg.respect_robots:
            return True
        return self._robots_for(url).can_fetch(self.cfg.user_agent, url)

    # -- landing zone -------------------------------------------------

    def _payload_path(self, sha: str) -> Path:
        # content-addressed: identical bytes stored once, cheap change detection
        return self.cfg.landing_dir / sha[:2] / sha[2:4] / f"{sha}.gz"

    def _write_payload(self, content: bytes) -> tuple[str, Path]:
        sha = hashlib.sha256(content).hexdigest()
        path = self._payload_path(sha)
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".gz.tmp")
            tmp.write_bytes(gzip.compress(content))
            tmp.rename(path)          # atomic; no half-written payloads
        return sha, path

    def read_payload(self, payload_path: str | Path) -> bytes:
        return gzip.decompress(Path(payload_path).read_bytes())

    def read_json(self, payload_path: str | Path) -> Any:
        return json.loads(self.read_payload(payload_path))

    # -- fetch --------------------------------------------------------

    def _throttle(self) -> None:
        wait = self.cfg.min_interval_s - (time.monotonic() - self._last_request_at)
        if wait > 0:
            time.sleep(wait)
        self._last_request_at = time.monotonic()

    def fetch(
        self,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        method: str = "GET",
        json_body: Any = None,
    ) -> FetchResult:
        if not self.allowed(url):
            raise PermissionError(f"robots.txt disallows {url}")

        last_exc: Exception | None = None
        for attempt in range(self.cfg.max_retries):
            self._throttle()
            try:
                resp = self.session.request(
                    method, url, params=params, json=json_body,
                    timeout=self.cfg.timeout_s,
                )
            except requests.RequestException as exc:
                last_exc = exc
                time.sleep(self.cfg.backoff_base_s * (2 ** attempt))
                continue

            # 429/503 mean slow down, not retry harder
            if resp.status_code in (429, 503):
                retry_after = float(resp.headers.get("Retry-After", 0) or 0)
                sleep_s = max(retry_after, self.cfg.backoff_base_s * (2 ** attempt))
                log.warning("throttled (%s) on %s; sleeping %.0fs", resp.status_code, url, sleep_s)
                time.sleep(sleep_s)
                continue

            sha, path = self._write_payload(resp.content)
            return FetchResult(
                url=url,
                fetched_at=datetime.now(timezone.utc),
                http_status=resp.status_code,
                content_type=resp.headers.get("Content-Type"),
                content_sha256=sha,
                content_bytes=len(resp.content),
                payload_path=str(path),
                from_cache=False,
            )

        raise RuntimeError(f"exhausted retries for {url}") from last_exc

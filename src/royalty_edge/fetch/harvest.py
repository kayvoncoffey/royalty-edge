"""Detail harvest.

Drains fetch_queue. Designed to be interrupted: state lives in the database
after every single fetch, so Ctrl-C, a laptop lid, or an expired cookie all
leave a resumable job rather than a corrupt one.

Two failure modes get special handling because they are the ones that quietly
ruin a crawl:

  * Auth expiry mid-run. A session cookie that dies at request 900 turns the
    remaining 1,600 into HTML shells that parse as nothing. The harvester
    aborts the whole run on the first auth failure rather than burning through
    the queue marking everything failed.

  * Parse failure on a subset. A listing type we have never seen (an NFT lot,
    a syndicate, a non-music asset) may have a different payload shape. Those
    are marked failed with the error text and skipped, not retried forever.
    Inspect them afterwards with the failed-queue report; they are usually the
    most interesting rows in the dataset.
"""

from __future__ import annotations

import logging
import signal
import time
from dataclasses import dataclass
from datetime import datetime, timezone

from ..db.load import insert_listing_observation, record_fetch
from ..parse.listing import parse_listing_payload, reconcile
from .session import looks_like_auth_failure

log = logging.getLogger(__name__)


@dataclass
class HarvestStats:
    attempted: int = 0
    ok: int = 0
    failed: int = 0
    skipped: int = 0
    reconcile_warnings: int = 0
    stopped_reason: str = "queue_empty"


class _Interrupt:
    """Finish the request in flight, then stop cleanly."""

    def __init__(self) -> None:
        self.requested = False
        self._prev = signal.signal(signal.SIGINT, self._handle)

    def _handle(self, *_):
        if self.requested:
            raise KeyboardInterrupt
        self.requested = True
        log.warning("interrupt received; finishing current request then stopping. "
                    "Press Ctrl-C again to abort immediately.")

    def restore(self) -> None:
        signal.signal(signal.SIGINT, self._prev)


def pending_count(con) -> int:
    return con.execute(
        "SELECT count(*) FROM fetch_queue WHERE state = 'pending'").fetchone()[0]


def harvest(con, fetcher, *, run_id: str, limit: int | None = None,
            max_runtime_s: float | None = None, max_attempts: int = 3,
            retry_failed: bool = False) -> HarvestStats:
    stats = HarvestStats()
    started = time.monotonic()
    interrupt = _Interrupt()

    states = "('pending','failed')" if retry_failed else "('pending')"
    try:
        while True:
            if interrupt.requested:
                stats.stopped_reason = "interrupted"
                break
            if limit is not None and stats.attempted >= limit:
                stats.stopped_reason = "limit"
                break
            if max_runtime_s and (time.monotonic() - started) > max_runtime_s:
                stats.stopped_reason = "max_runtime"
                break

            row = con.execute(
                f"SELECT url, endpoint_kind, listing_id, attempts FROM fetch_queue "
                f"WHERE state IN {states} AND attempts < ? "
                f"ORDER BY discovered_at LIMIT 1", [max_attempts]).fetchone()
            if not row:
                break
            url, kind, listing_id, attempts = row

            con.execute("UPDATE fetch_queue SET last_attempt_at = ?, attempts = ? "
                        "WHERE url = ?",
                        [datetime.now(timezone.utc), attempts + 1, url])
            stats.attempted += 1

            try:
                res = fetcher.fetch(url)
            except Exception as exc:
                _fail(con, url, f"request error: {exc}")
                stats.failed += 1
                continue

            body = fetcher.read_payload(res.payload_path)

            if looks_like_auth_failure(res.http_status, body):
                _fail(con, url, f"auth failure (HTTP {res.http_status})")
                stats.failed += 1
                stats.stopped_reason = "auth_failure"
                log.error("auth failure on %s -- aborting run so the rest of the "
                          "queue is not burned. Refresh the cookie and resume.", url)
                break

            if res.http_status != 200:
                _fail(con, url, f"HTTP {res.http_status}")
                stats.failed += 1
                continue

            fetch_id = record_fetch(
                con, url=res.url, fetched_at=res.fetched_at,
                http_status=res.http_status, content_type=res.content_type,
                content_sha256=res.content_sha256, content_bytes=res.content_bytes,
                payload_path=res.payload_path, endpoint_kind=kind, run_id=run_id)

            try:
                obs = parse_listing_payload(body, observed_at=res.fetched_at)
                insert_listing_observation(con, obs, fetch_id=fetch_id)
            except Exception as exc:
                _fail(con, url, f"parse/load error: {type(exc).__name__}: {exc}")
                stats.failed += 1
                log.warning("listing %s failed to parse: %s", listing_id, exc)
                continue

            problems = reconcile(obs)
            if problems:
                stats.reconcile_warnings += 1
                con.execute(
                    "UPDATE fetch_queue SET state='done', last_error=? WHERE url=?",
                    ["; ".join(problems), url])
            else:
                con.execute(
                    "UPDATE fetch_queue SET state='done', last_error=NULL WHERE url=?",
                    [url])
            stats.ok += 1

            if stats.attempted % 25 == 0:
                rate = stats.attempted / max(time.monotonic() - started, 1e-9)
                left = pending_count(con)
                log.info("harvest: %d ok / %d failed / %d attempted; %d pending "
                         "(~%.0f min left at %.2f req/s)",
                         stats.ok, stats.failed, stats.attempted, left,
                         left / rate / 60 if rate else 0, rate)
    finally:
        interrupt.restore()

    log.info("harvest finished: %d ok, %d failed, %d reconcile warnings (%s)",
             stats.ok, stats.failed, stats.reconcile_warnings, stats.stopped_reason)
    return stats


def _fail(con, url: str, msg: str) -> None:
    con.execute("UPDATE fetch_queue SET state='failed', last_error=? WHERE url=?",
                [msg, url])


def queue_report(con) -> list[tuple]:
    return con.execute(
        "SELECT state, count(*), min(discovered_at), max(last_attempt_at) "
        "FROM fetch_queue GROUP BY state ORDER BY 2 DESC").fetchall()


def failure_report(con, limit: int = 20) -> list[tuple]:
    return con.execute(
        "SELECT listing_id, attempts, last_error FROM fetch_queue "
        "WHERE state='failed' ORDER BY listing_id LIMIT ?", [limit]).fetchall()

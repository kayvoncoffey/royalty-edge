"""Index walk.

Follows the API's own `next` cursor rather than incrementing `page`. That
matters: if the server changes page_size mid-crawl, or a listing is inserted
while we walk, page-number arithmetic silently skips rows while the cursor
does not.

Every index page is persisted to the landing zone and every row is appended
to obs_index. The queue of detail URLs is a side effect, not the product --
if the detail endpoint is never resolved, the index alone still gives 2,500
rows of clearing prices, LTM, dollar age, term and kind.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone

from ..db.load import insert_index_observation, record_fetch
from ..parse.listing import parse_index_payload
from .probe import INDEX_FILTER_PARAMS, INDEX_URL
from .session import looks_like_auth_failure

log = logging.getLogger(__name__)


@dataclass
class DiscoverStats:
    pages: int = 0
    rows: int = 0
    queued: int = 0
    total_results: int | None = None
    stopped_reason: str = "complete"


def discover(con, fetcher, *, run_id: str, page_size: int = 15,
             detail_url_template: str | None = None,
             max_pages: int | None = None,
             include_states: tuple[str, ...] = ("filled", "pending", "closed"),
             ) -> DiscoverStats:
    stats = DiscoverStats()
    params = [("filter{state.in}", s) for s in include_states] + [
        p for p in INDEX_FILTER_PARAMS if p[0] != "filter{state.in}"
    ] + [("page", "1"), ("page_size", str(page_size))]

    url: str | None = INDEX_URL
    use_params: list[tuple[str, str]] | None = params

    while url:
        if max_pages is not None and stats.pages >= max_pages:
            stats.stopped_reason = "max_pages"
            break

        res = fetcher.fetch(url, params=use_params)
        body = fetcher.read_payload(res.payload_path)

        if looks_like_auth_failure(res.http_status, body):
            stats.stopped_reason = "auth_failure"
            log.error("index page %d returned an HTML shell or auth error; "
                      "refresh the session cookie", stats.pages + 1)
            break

        fetch_id = record_fetch(
            con, url=res.url, fetched_at=res.fetched_at, http_status=res.http_status,
            content_type=res.content_type, content_sha256=res.content_sha256,
            content_bytes=res.content_bytes, payload_path=res.payload_path,
            endpoint_kind="index", run_id=run_id,
            request_params=dict(use_params) if use_params else None,
        )

        rows, next_url, meta = parse_index_payload(body, observed_at=res.fetched_at)
        stats.total_results = meta.get("total_results", stats.total_results)

        for r in rows:
            insert_index_observation(con, r, fetch_id=fetch_id)
            stats.rows += 1
            if detail_url_template:
                stats.queued += _queue(
                    con, detail_url_template.format(id=r.listing_id),
                    "listing_detail", r.listing_id, res.fetched_at)

        stats.pages += 1
        if stats.pages % 10 == 0:
            log.info("index: %d pages, %d rows (of %s)", stats.pages, stats.rows,
                     stats.total_results)

        url, use_params = next_url, None   # cursor carries its own params

    log.info("index walk finished: %d pages, %d rows, %d queued (%s)",
             stats.pages, stats.rows, stats.queued, stats.stopped_reason)
    return stats


def _queue(con, url: str, kind: str, listing_id: int, now: datetime) -> int:
    """Insert if absent. Never resets an existing row's state -- that is what
    makes a re-run of discover cheap instead of a full refetch."""
    existing = con.execute("SELECT 1 FROM fetch_queue WHERE url = ?", [url]).fetchone()
    if existing:
        return 0
    con.execute(
        "INSERT INTO fetch_queue (url, endpoint_kind, listing_id, discovered_at) "
        "VALUES (?,?,?,?)", [url, kind, listing_id, now])
    return 1


def queue_details_from_index(con, *, detail_url_template: str,
                             only_missing: bool = True) -> int:
    """Backfill the queue from obs_index rows already in the database, for the
    case where discover ran before the detail endpoint was known."""
    sql = "SELECT DISTINCT listing_id FROM obs_index"
    if only_missing:
        sql += (" WHERE listing_id NOT IN (SELECT listing_id FROM obs_listing "
                "WHERE listing_id IS NOT NULL)")
    now = datetime.now(timezone.utc)
    n = 0
    for (lid,) in con.execute(sql).fetchall():
        n += _queue(con, detail_url_template.format(id=lid), "listing_detail", lid, now)
    return n

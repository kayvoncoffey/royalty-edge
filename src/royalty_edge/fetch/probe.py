"""Endpoint probes.

Two unknowns remain after the first recon pass, and both are cheap to settle
with a handful of requests. Settling them before a 2,500-request crawl is the
difference between a two-hour job and a two-hour job done twice.

  1. The detail endpoint URL. We have a detail *payload* but not the URL that
     produced it. The SPA route is /orderbook/asset-detail/<id>/, which is a
     client-side route, not the API. Candidates below are ranked by how the
     index endpoint is shaped; probe_detail_url tries each and accepts the
     first that parses AND reconciles.

  2. The maximum page_size. The UI requests 15. Django REST Framework
     installations usually cap this server-side and silently return the cap
     rather than erroring, so the only way to know is to ask and count.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable

from ..parse.listing import parse_index_payload, parse_listing_payload, reconcile
from .session import BASE, looks_like_auth_failure

log = logging.getLogger(__name__)

INDEX_URL = f"{BASE}/inventory/listings/"

# Ranked guesses. The index lives at /inventory/listings/, so a DRF detail
# route at the same collection is by far the most likely.
DETAIL_URL_CANDIDATES = [
    f"{BASE}/inventory/listings/{{id}}/",
    f"{BASE}/orderbook/api/listings/{{id}}/",
    f"{BASE}/inventory/listing/{{id}}/",
    f"{BASE}/orderbook/api/listing-detail/{{id}}/",
]

INDEX_FILTER_PARAMS = [
    ("filter{state.in}", "filled"),
    ("filter{state.in}", "pending"),
    ("filter{state.in}", "closed"),
    ("only_favorited", "0"),
    ("sort[]", "-deal_date"),
    ("sort[]", "-published_date"),
]


@dataclass
class ProbeResult:
    ok: bool
    detail_url_template: str | None
    max_page_size: int
    total_results: int | None
    notes: list[str]


def probe_page_size(fetcher, *, candidates: Iterable[int] = (15, 50, 100, 250, 500)
                    ) -> tuple[int, int | None, list[str]]:
    """Return (largest page size actually honoured, total_results, notes)."""
    notes: list[str] = []
    best, total = 15, None
    for size in candidates:
        params = INDEX_FILTER_PARAMS + [("page", "1"), ("page_size", str(size))]
        res = fetcher.fetch(INDEX_URL, params=params)
        body = fetcher.read_payload(res.payload_path)
        if looks_like_auth_failure(res.http_status, body):
            notes.append(f"page_size={size}: auth failure or HTML shell returned")
            break
        try:
            rows, _, meta = parse_index_payload(body)
        except Exception as exc:
            notes.append(f"page_size={size}: unparseable ({exc})")
            break
        total = meta.get("total_results", total)
        got = len(rows)
        notes.append(f"page_size={size}: asked {size}, got {got}, per_page={meta.get('per_page')}")
        if got >= size:
            best = size
        else:
            # server capped us; the cap is what we actually received
            best = max(best, got)
            break
    return best, total, notes


def probe_detail_url(fetcher, *, listing_id: int,
                     candidates: Iterable[str] = tuple(DETAIL_URL_CANDIDATES)
                     ) -> tuple[str | None, list[str]]:
    """Try each candidate against a known listing id. Accept only a payload
    that parses to the right id and passes reconciliation -- a 200 that
    parses but reconciles badly means we found a different, thinner endpoint."""
    notes: list[str] = []
    for tmpl in candidates:
        url = tmpl.format(id=listing_id)
        try:
            res = fetcher.fetch(url)
        except Exception as exc:
            notes.append(f"{url}: request failed ({exc})")
            continue
        body = fetcher.read_payload(res.payload_path)
        if res.http_status != 200:
            notes.append(f"{url}: HTTP {res.http_status}")
            continue
        if looks_like_auth_failure(res.http_status, body):
            notes.append(f"{url}: HTML shell / auth failure")
            continue
        try:
            obs = parse_listing_payload(body)
        except Exception as exc:
            notes.append(f"{url}: parsed as JSON but not a listing ({exc})")
            continue
        if obs.listing_id != listing_id:
            notes.append(f"{url}: returned listing {obs.listing_id}, expected {listing_id}")
            continue
        problems = reconcile(obs)
        if problems:
            notes.append(f"{url}: parsed but failed reconciliation: {problems}")
            continue
        notes.append(f"{url}: OK -- {len(obs.offers)} offers, {len(obs.quarters)} quarters")
        return tmpl, notes
    return None, notes


def run_probe(fetcher, *, listing_id: int) -> ProbeResult:
    size, total, n1 = probe_page_size(fetcher)
    tmpl, n2 = probe_detail_url(fetcher, listing_id=listing_id)
    return ProbeResult(
        ok=tmpl is not None,
        detail_url_template=tmpl,
        max_page_size=size,
        total_results=total,
        notes=n1 + n2,
    )

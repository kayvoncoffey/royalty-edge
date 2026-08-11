"""Raw payload -> typed observation records.

Written against live payloads captured 2026-08-11 from:
  index   GET /inventory/listings/?filter{state.in}=...&page=N&page_size=15
  detail  GET /inventory/listings/<id>/   (shape confirmed from asset-detail XHR)

Bump PARSER_VERSION on every change. It is stamped on every observation row
so a mixed-version database can be diagnosed instead of discarded.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any, Iterator

PARSER_VERSION = "0.2.0"

# --------------------------------------------------------------------
# Coercion
# --------------------------------------------------------------------

def _f(x: Any) -> float | None:
    """Numeric fields arrive as both strings and floats in the same payload.
    Missing stays None -- a missing price and a zero price are different
    facts and collapsing them corrupts every downstream mean."""
    if x is None or x == "":
        return None
    if isinstance(x, bool):
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _i(x: Any) -> int | None:
    v = _f(x)
    return None if v is None else int(v)


def _ts(x: Any) -> datetime | None:
    if not x:
        return None
    s = str(x).replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        return None


def _d(x: Any) -> date | None:
    if not x:
        return None
    try:
        return date.fromisoformat(str(x)[:10])
    except ValueError:
        return None


# --------------------------------------------------------------------
# Term taxonomy
# --------------------------------------------------------------------

# Observed values: life_of_rights, 10_year, 30_year, partial_10_year,
# fixed_return. fixed_return is a contracted sequence of advance payments,
# NOT a decaying royalty stream -- it must never enter the decay model, and
# its "multiple" is not comparable to a perpetuity multiple.
_TERM_FAMILY = {
    "life_of_rights": "perpetual",
    "fixed_return": "fixed_return",
}


def term_family(term: str | None) -> str | None:
    if not term:
        return None
    if term in _TERM_FAMILY:
        return _TERM_FAMILY[term]
    if re.search(r"\d+_year$", term):
        return "fixed_term"
    return "unknown"


def term_years(term: str | None) -> float | None:
    if not term:
        return None
    m = re.search(r"(\d+)_year", term)
    return float(m.group(1)) if m else None


def is_partial_share(term: str | None) -> bool:
    return bool(term and term.startswith("partial_"))


def buyer_fee(final_price: float | None, *, all_access: bool = False,
              opened_bidding: bool = False) -> float:
    """1% of final price, $500 minimum. Waived for All Access members and for
    whoever placed the opening bid on a listing they go on to win."""
    if final_price is None:
        return 0.0
    if all_access or opened_bidding:
        return 0.0
    return max(500.0, 0.01 * final_price)


# --------------------------------------------------------------------
# Records
# --------------------------------------------------------------------

@dataclass
class OfferRecord:
    offer_id: int
    listing_id: int
    buyer_id: int | None = None
    bidder_index: int | None = None
    amount: float | None = None
    multiple: float | None = None
    multiple_source: str | None = None
    kind: str | None = None
    state: str | None = None
    term: str | None = None
    created_at: datetime | None = None
    expiration: datetime | None = None
    countered_offer_id: int | None = None
    counteroffer_id: int | None = None
    incentive_pool_qualifying: bool | None = None


@dataclass
class QuarterRecord:
    period_start: date
    domestic: float | None = None
    intl: float | None = None
    unreported: float | None = None
    total: float | None = None


@dataclass
class YearRecord:
    dimension: str
    year_index: int
    member: str
    amount: float | None


@dataclass
class BreakdownRecord:
    dimension: str
    member: str
    amount: float | None
    track_count: int | None = None


@dataclass
class TrackRecord:
    track_id: str
    parent_track_id: str | None
    title: str | None


@dataclass
class BuyNowRecord:
    seq: int
    created_at: datetime | None
    buy_now_price: float | None


@dataclass
class IndexObservation:
    listing_id: int
    observed_at: datetime
    raw_extra: dict = field(default_factory=dict)
    # populated dynamically; declared for the loader's column list
    fields: dict = field(default_factory=dict)


@dataclass
class ListingObservation:
    listing_id: int
    observed_at: datetime
    fields: dict = field(default_factory=dict)
    offers: list[OfferRecord] = field(default_factory=list)
    quarters: list[QuarterRecord] = field(default_factory=list)
    years: list[YearRecord] = field(default_factory=list)
    breakdowns: list[BreakdownRecord] = field(default_factory=list)
    tracks: list[TrackRecord] = field(default_factory=list)
    buy_now_history: list[BuyNowRecord] = field(default_factory=list)
    raw_extra: dict = field(default_factory=dict)


# ====================================================================
# Index adapter
# ====================================================================

INDEX_FIELDS = [
    "title", "url", "state", "kind", "term", "term_remaining", "term_expiration",
    "seller_id", "ltm", "dollar_age", "deal_amount", "deal_date", "close_date",
    "published_date", "list_price", "list_price_multiple",
    "list_price_multiple_source", "marketplace_median",
    "marketplace_median_multiplier", "marketplace_median_source",
    "is_marketplace_median_disabled", "highest_open_offer_amount",
    "accepting_final_offers_start_at", "accepting_final_offers_end_at",
    "is_open_auction", "is_nft", "is_featured", "currency",
]

_INDEX_COERCE = {
    "term_remaining": _f, "seller_id": _i, "ltm": _f, "dollar_age": _f,
    "deal_amount": _f, "list_price": _f, "list_price_multiple": _f,
    "marketplace_median": _f, "marketplace_median_multiplier": _f,
    "highest_open_offer_amount": _f,
    "term_expiration": _ts, "deal_date": _ts, "close_date": _ts,
    "published_date": _ts, "accepting_final_offers_start_at": _ts,
    "accepting_final_offers_end_at": _ts,
}


def parse_index_payload(payload: bytes, *, observed_at: datetime | None = None
                        ) -> tuple[list[IndexObservation], str | None, dict]:
    """Returns (rows, next_url, meta). next_url is the API's own cursor --
    use it rather than incrementing page, so a mid-crawl page-size change
    cannot silently skip listings."""
    obj = json.loads(payload)
    observed_at = observed_at or datetime.now(timezone.utc)
    rows: list[IndexObservation] = []
    known = set(INDEX_FIELDS) | {"listing_id", "tags"}

    for r in obj.get("results", []):
        lid = _i(r.get("listing_id"))
        if lid is None:
            continue
        vals = {}
        for k in INDEX_FIELDS:
            fn = _INDEX_COERCE.get(k)
            vals[k] = fn(r.get(k)) if fn else r.get(k)
        vals["tags"] = json.dumps(r.get("tags") or [])
        rows.append(IndexObservation(
            listing_id=lid, observed_at=observed_at, fields=vals,
            raw_extra={k: v for k, v in r.items() if k not in known},
        ))

    meta = obj.get("meta", {}) or {}
    meta.setdefault("count", obj.get("count"))
    return rows, obj.get("next"), meta


def listing_ids_from_index(payload: bytes) -> Iterator[int]:
    for r in json.loads(payload).get("results", []):
        lid = _i(r.get("listing_id"))
        if lid is not None:
            yield lid


# ====================================================================
# Detail adapter
# ====================================================================

_DETAIL_SCALARS = {
    "title": (("title",), None),
    "state": (("state",), None),
    "kind": (("kind",), None),
    "term": (("term",), None),
    "term_expiration": (("term_expiration",), _ts),
    "term_return_amount": (("term_return_amount",), _f),
    "seller_id": (("seller_id",), _i),
    "published_date": (("published_date",), _ts),
    "minimum_price": (("minimum_price",), _f),
    "default_buy_now_price": (("default_buy_now_price",), _f),
    "buy_now_price": (("buy_now_price",), _f),
    "proxy_increment_amount": (("proxy_increment_amount",), _f),
    "offers_received_count": (("offers_received_count",), _i),
    "unique_bidder_count": (("unique_bidder_count",), _i),
    "accepting_final_offers_start_at": (("accepting_final_offers_start_at",), _ts),
    "accepting_final_offers_end_at": (("accepting_final_offers_end_at",), _ts),
    "accepting_final_offers_initial_offer": (("accepting_final_offers_initial_offer",), _i),
    "marketplace_median": (("marketplace_median",), _f),
    "marketplace_median_multiplier": (("marketplace_median_multiplier",), _f),
    "track_count": (("track_count",), _i),
    "is_track_list_hidden": (("is_track_list_hidden",), None),
    "valuation_id": (("active_listing_valuation_id",), _i),
    "asset_id": (("asset", "id"), _i),
    "asset_sale_date": (("asset", "sale_date"), _ts),
    "display_state_msg": (("display_state", "message"), None),
    "ltm": (("valuation", "ltm"), _f),
    "lifetime_amount": (("valuation", "lifetime_amount"), _f),
    "three_years_average": (("valuation", "three_years_average"), _f),
    "dollar_age": (("valuation", "dollar_age"), _f),
    "first_earnings_date": (("valuation", "first_earnings_date"), _ts),
    "distribution_frequency": (("valuation", "distribution_frequency"), _i),
    "statistics_last_updated": (("valuation", "statistics_last_updated"), _ts),
    "vd_prompt_id": (("valuation_description", "prompt_id"), None),
    "vd_prompt_version": (("valuation_description", "prompt_version"), _i),
    "deal_offer_id": (("deal", "id"), _i),
    "deal_amount": (("deal", "amount"), _f),
    "deal_buyer_id": (("deal", "buyer_id"), _i),
    "deal_multiple": (("deal", "multiple"), _f),
    "deal_multiple_source": (("deal", "multiple_source"), None),
    "deal_state": (("deal", "state"), None),
    "deal_created_at": (("deal", "created_at"), _ts),
}

# top_earnings_by_year sub-key -> dimension label
_YEAR_DIMS = {
    "top_songs": "song",
    "top_income_types": "income_type",
    "top_sources": "source",
    "top_music_users": "music_user",
    "top_royalty_payors": "royalty_payor",
}
_LTM_DIMS = {
    "top_songs": "song",
    "top_income_types": "income_type",
    "top_sources": "source",
    "top_music_users": "music_user",
}


def _dig(obj: dict, path: tuple[str, ...]) -> Any:
    cur: Any = obj
    for p in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(p)
    return cur


def parse_listing_payload(payload: bytes, *, observed_at: datetime | None = None
                          ) -> ListingObservation:
    obj = json.loads(payload)
    observed_at = observed_at or datetime.now(timezone.utc)
    lid = _i(obj.get("id"))
    if lid is None:
        raise ValueError("listing detail payload has no id")

    vals: dict[str, Any] = {}
    for col, (path, fn) in _DETAIL_SCALARS.items():
        raw = _dig(obj, path)
        vals[col] = fn(raw) if fn else raw
    for col in ("tags", "media_urls"):
        vals[col] = json.dumps(obj.get(col) or [])
    vals["royalty_payors"] = json.dumps(
        [{"name": p.get("name"), "display_name": p.get("display_name"),
          "kind": p.get("kind")}
         for p in (_dig(obj, ("asset", "royalty_payors")) or [])]
    )

    obs = ListingObservation(listing_id=lid, observed_at=observed_at, fields=vals)

    # -- offers -------------------------------------------------------
    for o in obj.get("offers") or []:
        oid = _i(o.get("id"))
        if oid is None:
            continue
        obs.offers.append(OfferRecord(
            offer_id=oid, listing_id=lid,
            buyer_id=_i(o.get("buyer_id")), bidder_index=_i(o.get("bidder_index")),
            amount=_f(o.get("amount")), multiple=_f(o.get("multiple")),
            multiple_source=o.get("multiple_source"), kind=o.get("kind"),
            state=o.get("state"), term=o.get("term"),
            created_at=_ts(o.get("created_at")), expiration=_ts(o.get("expiration")),
            countered_offer_id=_i(o.get("countered_offer_id")),
            counteroffer_id=_i(o.get("counteroffer_id")),
            incentive_pool_qualifying=o.get("incentive_pool_qualifying"),
        ))

    # -- buy-now ask trajectory ---------------------------------------
    for i, h in enumerate(obj.get("buy_now_price_history") or []):
        obs.buy_now_history.append(BuyNowRecord(
            seq=i, created_at=_ts(h.get("created_at")),
            buy_now_price=_f(h.get("buy_now_price")),
        ))

    val = obj.get("valuation") or {}

    # -- quarterly panel ----------------------------------------------
    # shape: [["2019-10-01", {"domestic":..,"intl":..,"unreported":..,"total":..}], ...]
    for entry in val.get("earnings_by_region") or []:
        if not (isinstance(entry, list) and len(entry) == 2):
            continue
        d0, m = entry
        pd_ = _d(d0)
        if pd_ is None or not isinstance(m, dict):
            continue
        obs.quarters.append(QuarterRecord(
            period_start=pd_, domestic=_f(m.get("domestic")), intl=_f(m.get("intl")),
            unreported=_f(m.get("unreported")), total=_f(m.get("total")),
        ))

    # -- annual panel, four taxonomies --------------------------------
    # shape: {"top_songs":[{"year":"year_1","SONG A":123.0,...}, ...], ...}
    teby = val.get("top_earnings_by_year") or {}
    for key, dim in _YEAR_DIMS.items():
        for row in teby.get(key) or []:
            if not isinstance(row, dict):
                continue
            yi = _i(str(row.get("year", "")).replace("year_", ""))
            if yi is None:
                continue
            for member, amt in row.items():
                if member == "year":
                    continue
                obs.years.append(YearRecord(dim, yi, str(member), _f(amt)))

    # -- LTM-window breakdown -----------------------------------------
    for key, dim in _LTM_DIMS.items():
        for row in val.get(key) or []:
            if not isinstance(row, dict):
                continue
            obs.breakdowns.append(BreakdownRecord(
                dimension=dim, member=str(row.get("name")),
                amount=_f(row.get("earnings")),
                track_count=_i(row.get("track_count")),
            ))

    # -- tracks -------------------------------------------------------
    for t in val.get("track_list") or []:
        tid = t.get("id")
        if not tid:
            continue
        obs.tracks.append(TrackRecord(
            track_id=str(tid), parent_track_id=(t.get("parent_id") or None),
            title=t.get("title"),
        ))

    consumed = {"id", "offers", "deal", "buy_now_price_history", "valuation",
                "valuation_description", "asset", "display_state", "tags",
                "media_urls"} | set(_DETAIL_SCALARS)
    obs.raw_extra = {k: v for k, v in obj.items() if k not in consumed}
    return obs


# --------------------------------------------------------------------
# Reconciliation -- run on every parsed detail payload
# --------------------------------------------------------------------

def reconcile(obs: ListingObservation, *, tol: float = 0.02) -> list[str]:
    """The quarterly panel must sum to lifetime_amount, and its last four
    quarters must sum to LTM. When that holds, the panel is complete and the
    decay model can be fitted on it. When it does not, something is being
    withheld or restated and the listing should not enter the decay sample."""
    problems: list[str] = []
    q = sorted(obs.quarters, key=lambda x: x.period_start)
    if not q:
        return ["no quarterly panel"]

    life = obs.fields.get("lifetime_amount")
    ltm = obs.fields.get("ltm")
    tot = sum(x.total or 0.0 for x in q)
    if life and abs(tot - life) > tol * max(abs(life), 1.0):
        problems.append(f"quarterly sum {tot:,.2f} != lifetime {life:,.2f}")
    if ltm and len(q) >= 4:
        last4 = sum(x.total or 0.0 for x in q[-4:])
        if abs(last4 - ltm) > tol * max(abs(ltm), 1.0):
            problems.append(f"last 4 quarters {last4:,.2f} != ltm {ltm:,.2f}")

    n_off = obs.fields.get("offers_received_count")
    if n_off is not None and len(obs.offers) != n_off:
        problems.append(f"offers list has {len(obs.offers)}, count says {n_off}")

    uniq = obs.fields.get("unique_bidder_count")
    if uniq is not None and obs.offers:
        seen = len({o.buyer_id for o in obs.offers if o.buyer_id is not None})
        if seen != uniq:
            problems.append(f"distinct buyer_id {seen} != unique_bidder_count {uniq}")
    return problems

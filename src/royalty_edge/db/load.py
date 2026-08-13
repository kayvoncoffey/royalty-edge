"""DuckDB loader (v0.2).

L1 (obs_*) is append-only and is the only thing the harvester writes.
L2 (dim_/fact_*) is a pure function of L1 and is dropped and rebuilt.

Single-writer note: DuckDB takes an exclusive lock on the file, which is why
harvest (writes payloads to disk) and load (takes the lock for seconds) are
separate commands.
"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb

from ..parse.listing import (PARSER_VERSION, INDEX_FIELDS, IndexObservation,
                             ListingObservation, buyer_fee)

SCHEMA_PATH = Path(__file__).with_name("schema.sql")


def connect(db_path: str | Path, *, read_only: bool = False):
    con = duckdb.connect(str(db_path), read_only=read_only)
    if not read_only:
        con.execute(SCHEMA_PATH.read_text())
    return con


def record_fetch(con, *, url, fetched_at, http_status, content_type, content_sha256,
                 content_bytes, payload_path, endpoint_kind, run_id,
                 request_params=None, notes=None) -> int:
    return int(con.execute(
        """INSERT INTO raw_fetch (url, request_params, fetched_at, http_status,
             content_type, content_sha256, content_bytes, payload_path,
             endpoint_kind, run_id, notes)
           VALUES (?,?,?,?,?,?,?,?,?,?,?) RETURNING fetch_id""",
        [url, json.dumps(request_params or {}), fetched_at, http_status, content_type,
         content_sha256, content_bytes, str(payload_path), endpoint_kind, run_id, notes],
    ).fetchone()[0])


# --------------------------------------------------------------------
# L1
# --------------------------------------------------------------------

def insert_index_observation(con, obs: IndexObservation, *, fetch_id: int) -> int:
    cols = ["fetch_id", "parser_version", "observed_at", "listing_id"] + INDEX_FIELDS + ["tags", "raw_extra"]
    vals = [fetch_id, PARSER_VERSION, obs.observed_at, obs.listing_id]
    vals += [obs.fields.get(k) for k in INDEX_FIELDS]
    vals += [obs.fields.get("tags"), json.dumps(obs.raw_extra)]
    ph = ",".join("?" * len(cols))
    return int(con.execute(
        f"INSERT INTO obs_index ({','.join(cols)}) VALUES ({ph}) RETURNING obs_id", vals
    ).fetchone()[0])


_DETAIL_COLS = [
    "listing_id", "asset_id", "valuation_id", "title", "state", "display_state_msg",
    "kind", "term", "term_expiration", "term_return_amount", "seller_id",
    "published_date", "asset_sale_date", "minimum_price", "default_buy_now_price",
    "buy_now_price", "proxy_increment_amount", "offers_received_count",
    "unique_bidder_count", "accepting_final_offers_start_at",
    "accepting_final_offers_end_at", "accepting_final_offers_initial_offer",
    "deal_offer_id", "deal_amount", "deal_buyer_id", "deal_multiple",
    "deal_multiple_source", "deal_state", "deal_created_at", "marketplace_median",
    "marketplace_median_multiplier", "ltm", "lifetime_amount", "three_years_average",
    "dollar_age", "first_earnings_date", "distribution_frequency",
    "statistics_last_updated", "track_count", "is_track_list_hidden",
    "vd_prompt_id", "vd_prompt_version", "tags", "royalty_payors", "media_urls",
]


def insert_listing_observation(con, obs: ListingObservation, *, fetch_id: int) -> int:
    cols = ["fetch_id", "parser_version", "observed_at"] + _DETAIL_COLS + ["raw_extra"]
    vals = [fetch_id, PARSER_VERSION, obs.observed_at]
    vals += [obs.listing_id if c == "listing_id" else obs.fields.get(c) for c in _DETAIL_COLS]
    vals += [json.dumps(obs.raw_extra, default=str)]
    ph = ",".join("?" * len(cols))
    obs_id = int(con.execute(
        f"INSERT INTO obs_listing ({','.join(cols)}) VALUES ({ph}) RETURNING obs_id", vals
    ).fetchone()[0])

    if obs.offers:
        con.executemany(
            "INSERT OR REPLACE INTO obs_offer VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            [[obs_id, o.offer_id, o.listing_id, o.buyer_id, o.bidder_index, o.amount,
              o.multiple, o.multiple_source, o.kind, o.state, o.term, o.created_at,
              o.expiration, o.countered_offer_id, o.counteroffer_id,
              o.incentive_pool_qualifying] for o in obs.offers])
    if obs.buy_now_history:
        con.executemany("INSERT OR REPLACE INTO obs_buy_now_history VALUES (?,?,?,?)",
            [[obs_id, h.seq, h.created_at, h.buy_now_price] for h in obs.buy_now_history])
    if obs.quarters:
        con.executemany("INSERT OR REPLACE INTO obs_earnings_quarter VALUES (?,?,?,?,?,?)",
            [[obs_id, q.period_start, q.domestic, q.intl, q.unreported, q.total]
             for q in obs.quarters])
    if obs.years:
        con.executemany("INSERT OR REPLACE INTO obs_earnings_year VALUES (?,?,?,?,?)",
            [[obs_id, y.dimension, y.year_index, y.member, y.amount] for y in obs.years])
    if obs.breakdowns:
        con.executemany("INSERT OR REPLACE INTO obs_ltm_breakdown VALUES (?,?,?,?,?)",
            [[obs_id, b.dimension, b.member, b.amount, b.track_count] for b in obs.breakdowns])
    if obs.tracks:
        con.executemany("INSERT OR REPLACE INTO obs_track VALUES (?,?,?,?)",
            [[obs_id, t.track_id, t.parent_track_id, t.title] for t in obs.tracks])
    return obs_id


# --------------------------------------------------------------------
# L2 rebuild
# --------------------------------------------------------------------

def rebuild_analytic(con, *, all_access: bool = False) -> None:
    for t in ("dim_listing", "fact_outcome", "fact_earnings_panel",
              "fact_source_mix", "fact_offer", "data_quality_flag"):
        con.execute(f"DELETE FROM {t}")

    # latest detail observation per listing
    con.execute("""
        CREATE OR REPLACE TEMP VIEW _det AS
        SELECT * FROM (
          SELECT *, row_number() OVER (PARTITION BY listing_id ORDER BY observed_at DESC) rn,
                 count(*) OVER (PARTITION BY listing_id) n_obs,
                 min(observed_at) OVER (PARTITION BY listing_id) first_obs,
                 max(observed_at) OVER (PARTITION BY listing_id) last_obs
          FROM obs_listing) WHERE rn = 1
    """)
    # latest index observation per listing
    con.execute("""
        CREATE OR REPLACE TEMP VIEW _idx AS
        SELECT * FROM (
          SELECT *, row_number() OVER (PARTITION BY listing_id ORDER BY observed_at DESC) rn,
                 count(*) OVER (PARTITION BY listing_id) n_obs_idx
          FROM obs_index) WHERE rn = 1
    """)

    # dim_listing: index is the spine (all listings), detail enriches
    con.execute("""
        INSERT INTO dim_listing
        SELECT
          coalesce(i.listing_id, d.listing_id),
          d.asset_id,
          coalesce(d.title, i.title),
          i.url,
          coalesce(d.state, i.state),
          coalesce(d.kind, i.kind),
          coalesce(d.term, i.term)                                  AS term,
          CASE
            WHEN coalesce(d.term, i.term) = 'life_of_rights' THEN 'perpetual'
            WHEN coalesce(d.term, i.term) = 'fixed_return'   THEN 'fixed_return'
            WHEN regexp_matches(coalesce(d.term, i.term), '\\d+_year$') THEN 'fixed_term'
            ELSE 'unknown' END,
          try_cast(regexp_extract(coalesce(d.term, i.term), '(\\d+)_year', 1) AS DOUBLE),
          coalesce(d.term, i.term) LIKE 'partial\\_%' ESCAPE '\\',
          coalesce(d.seller_id, i.seller_id),
          coalesce(d.published_date, i.published_date),
          i.deal_date,
          i.close_date,
          coalesce(d.ltm, i.ltm),
          d.three_years_average,
          d.lifetime_amount,
          coalesce(d.dollar_age, i.dollar_age),
          d.first_earnings_date,
          CASE WHEN d.first_earnings_date IS NOT NULL
               THEN date_diff('day', d.first_earnings_date,
                              coalesce(i.deal_date, d.published_date, now())) / 365.25 END,
          d.track_count,
          d.minimum_price,
          d.minimum_price / nullif(coalesce(d.ltm, i.ltm), 0),
          coalesce(d.marketplace_median_multiplier, i.marketplace_median_multiplier),
          i.is_marketplace_median_disabled,
          coalesce(d.n_obs, 0) + coalesce(i.n_obs_idx, 0),
          least(coalesce(d.first_obs, i.observed_at), i.observed_at),
          greatest(coalesce(d.last_obs, i.observed_at), i.observed_at),
          d.listing_id IS NOT NULL,
          coalesce(d.tags, i.tags)
        FROM _idx i FULL OUTER JOIN _det d USING (listing_id)
    """)

    # fact_offer with proxy-increment detection and ladder rank
    con.execute("""
        INSERT INTO fact_offer
        SELECT o.listing_id, o.offer_id, o.buyer_id, o.bidder_index, o.amount,
               o.multiple, o.created_at, o.state,
               o.state = 'awaiting_transfer' OR o.offer_id = d.deal_offer_id,
               o.state = 'fail_offer_increased',
               row_number() OVER (PARTITION BY o.listing_id ORDER BY o.amount)
        FROM obs_offer o
        JOIN _det d USING (obs_id)
    """)

    fee = "0.0" if all_access else "greatest(500.0, 0.01 * price)"
    con.execute(f"""
        INSERT INTO fact_outcome
        WITH base AS (
          SELECT l.listing_id, l.ltm, l.three_years_average,
                 l.marketplace_median_multiplier,
                 coalesce(d.deal_amount, i.deal_amount) AS price,
                 d.deal_buyer_id, d.offers_received_count, d.unique_bidder_count,
                 l.state, l.minimum_price
          FROM dim_listing l
          LEFT JOIN _det d USING (listing_id)
          LEFT JOIN _idx i USING (listing_id)
        ), winner AS (
          SELECT listing_id, any_value(buyer_id) AS winner_buyer_id
          FROM fact_offer WHERE is_winning GROUP BY 1
        ), ladder AS (
          SELECT f.listing_id,
                 max(f.amount) FILTER (WHERE NOT f.is_winning) AS second_price,
                 min(f.amount)                                 AS opening_price,
                 count(*) FILTER (WHERE f.buyer_id = w.winner_buyer_id) AS n_bids_by_winner
          FROM fact_offer f LEFT JOIN winner w USING (listing_id)
          GROUP BY f.listing_id
        )
        SELECT b.listing_id,
               b.price IS NOT NULL AND b.price > 0,
               CASE WHEN b.price > 0 THEN 'sold'
                    WHEN coalesce(b.offers_received_count,0) = 0 THEN 'no_offers'
                    ELSE 'unknown' END,
               b.price, b.deal_buyer_id, b.offers_received_count, b.unique_bidder_count,
               ld.second_price,
               b.price - ld.second_price,
               ld.opening_price, ld.n_bids_by_winner,
               CASE WHEN b.price IS NULL THEN NULL ELSE {fee} END,
               CASE WHEN b.price IS NULL THEN NULL ELSE b.price + {fee} END,
               b.price / nullif(b.ltm, 0),
               CASE WHEN b.price IS NULL THEN NULL
                    ELSE (b.price + {fee}) / nullif(b.ltm, 0) END,
               b.price / nullif(b.three_years_average, 0),
               (b.price / nullif(b.ltm, 0)) / nullif(b.marketplace_median_multiplier, 0),
               b.three_years_average,
               CASE WHEN b.price IS NULL THEN 'unsold'
                    WHEN b.minimum_price IS NOT NULL
                         AND abs(b.price - b.minimum_price) < 1e-6 THEN 'left_at_reserve'
                    ELSE 'none' END
        FROM base b LEFT JOIN ladder ld USING (listing_id)
    """)

    con.execute("""
        INSERT INTO fact_earnings_panel
        SELECT d.listing_id, d.asset_id, q.period_start, q.total, q.domestic,
               q.intl, q.unreported,
               date_diff('month', min(q.period_start) OVER (PARTITION BY d.listing_id),
                         q.period_start) / 3,
               date_diff('day', l.deal_date, q.period_start) / 91.31
        FROM _det d
        JOIN obs_earnings_quarter q USING (obs_id)
        JOIN dim_listing l USING (listing_id)
    """)

    con.execute("""
        INSERT INTO fact_source_mix
        SELECT d.listing_id, b.dimension, b.member, b.amount,
               b.amount / nullif(sum(b.amount) OVER (PARTITION BY d.listing_id, b.dimension), 0)
        FROM _det d JOIN obs_ltm_breakdown b USING (obs_id)
    """)

    con.execute("""
        INSERT INTO dim_asset
        SELECT l.asset_id, any_value(l.title), count(*),
               count(*) FILTER (WHERE o.sold),
               min(l.published_date), max(l.published_date),
               count(*) FILTER (WHERE o.sold) > 1
        FROM dim_listing l LEFT JOIN fact_outcome o USING (listing_id)
        WHERE l.asset_id IS NOT NULL GROUP BY l.asset_id
    """)

    _flag_quality(con)


_RULES = [
    ("missing_ltm", "exclude",
     "SELECT listing_id, 'LTM absent or non-positive' FROM dim_listing WHERE ltm IS NULL OR ltm <= 0"),
    ("fixed_return_instrument", "exclude",
     "SELECT listing_id, 'fixed-return advance; multiple not comparable, no decay' "
     "FROM dim_listing WHERE term_family = 'fixed_return'"),
    ("no_detail_payload", "warn",
     "SELECT listing_id, 'index-only; no offer ladder or earnings panel' "
     "FROM dim_listing WHERE NOT has_detail"),
    ("no_earnings_panel", "warn",
     "SELECT l.listing_id, 'no quarterly panel; cannot fit decay' FROM dim_listing l "
     "LEFT JOIN fact_earnings_panel p USING (listing_id) WHERE p.listing_id IS NULL"),
    ("ltm_far_above_3yr", "info",
     "SELECT listing_id, 'LTM more than 50% above 3-year average; suspect sync or breakout' "
     "FROM dim_listing WHERE three_years_average > 0 AND ltm / three_years_average > 1.5"),
    ("dollar_age_exceeds_catalog_age", "info",
     "SELECT listing_id, 'dollar_age exceeds age of catalog; not a weighted average in years' "
     "FROM dim_listing WHERE dollar_age IS NOT NULL AND catalog_age_years IS NOT NULL "
     "AND dollar_age > catalog_age_years + 0.51"),
    ("secondary_listing", "info",
     "SELECT listing_id, 'investor resale, not a creator sale' FROM dim_listing "
     "WHERE kind = 'secondary_listing'"),
    ("single_bidder", "warn",
     "SELECT listing_id, 'one unique bidder; price is the reserve, not a market clearing' "
     "FROM fact_outcome WHERE n_unique_bidders = 1"),
    ("fee_material_vs_price", "warn",
     "SELECT listing_id, 'buyer fee exceeds 2% of price' FROM fact_outcome "
     "WHERE buyer_fee_standard > 0.02 * clearing_price"),
]


def _flag_quality(con) -> None:
    con.execute("DELETE FROM data_quality_flag")
    for flag, sev, sql in _RULES:
        con.execute(
            "INSERT OR REPLACE INTO data_quality_flag "
            f"SELECT q.a, '{flag}', '{sev}', q.b, now() FROM ({sql}) AS q(a,b)")

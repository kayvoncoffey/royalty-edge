-- =====================================================================
-- royalty_edge :: DuckDB schema  (v0.2 -- rewritten against live payloads)
--
-- L0  raw_*      immutable fetch manifest; payload bytes on disk
-- L1  obs_*      append-only, one row per (entity, scrape observation)
-- L2  dim_/fact_ deduplicated analytic layer, fully rebuildable
--
-- Changes from v0.1, all forced by what the API actually returns:
--   * asset_id is a real stable key across listings. The fuzzy catalog
--     matcher is deleted; repeat listings of the same asset join exactly.
--   * offers are fully disclosed, including losing bids and persistent
--     buyer_id. The bid ladder is a first-class table, not an optional one.
--   * earnings_by_region is a complete quarterly panel from first earnings
--     to present and reconciles exactly to LTM and lifetime_amount. This is
--     the decay panel; it does not require the CSV download.
--   * income_type and general_source are two different taxonomies in the
--     source data (PERFORMANCE vs STREAMING). v0.1 conflated them.
--   * marketplace_median_multiplier is a platform-published anchor shown to
--     bidders before they bid. It is an endogenous regressor, kept separate.
-- =====================================================================

-- ---------------------------------------------------------------------
-- L0
-- ---------------------------------------------------------------------

CREATE SEQUENCE IF NOT EXISTS seq_fetch_id START 1;

CREATE TABLE IF NOT EXISTS raw_fetch (
    fetch_id        BIGINT PRIMARY KEY DEFAULT nextval('seq_fetch_id'),
    url             VARCHAR NOT NULL,
    request_params  JSON,
    fetched_at      TIMESTAMPTZ NOT NULL,
    http_status     INTEGER,
    content_type    VARCHAR,
    content_sha256  VARCHAR NOT NULL,
    content_bytes   BIGINT,
    payload_path    VARCHAR NOT NULL,
    endpoint_kind   VARCHAR NOT NULL,   -- index | listing_detail | cashflows | earnings_csv
    run_id          VARCHAR NOT NULL,
    notes           VARCHAR
);

CREATE TABLE IF NOT EXISTS scrape_run (
    run_id          VARCHAR PRIMARY KEY,
    started_at      TIMESTAMPTZ NOT NULL,
    finished_at     TIMESTAMPTZ,
    code_version    VARCHAR,
    n_requests      INTEGER,
    n_new_payloads  INTEGER,
    status          VARCHAR,
    notes           VARCHAR
);

CREATE TABLE IF NOT EXISTS fetch_queue (
    url             VARCHAR PRIMARY KEY,
    endpoint_kind   VARCHAR NOT NULL,
    listing_id      INTEGER,
    discovered_at   TIMESTAMPTZ NOT NULL,
    last_attempt_at TIMESTAMPTZ,
    attempts        INTEGER NOT NULL DEFAULT 0,
    state           VARCHAR NOT NULL DEFAULT 'pending',
    last_error      VARCHAR
);

-- ---------------------------------------------------------------------
-- L1 :: index rows (15 per request, covers all listings cheaply)
-- ---------------------------------------------------------------------

CREATE SEQUENCE IF NOT EXISTS seq_obs_id START 1;

CREATE TABLE IF NOT EXISTS obs_index (
    obs_id              BIGINT PRIMARY KEY DEFAULT nextval('seq_obs_id'),
    fetch_id            BIGINT NOT NULL,
    parser_version      VARCHAR NOT NULL,
    observed_at         TIMESTAMPTZ NOT NULL,

    listing_id          INTEGER NOT NULL,
    title               VARCHAR,
    url                 VARCHAR,
    state               VARCHAR,            -- pending | filled | closed | ...
    kind                VARCHAR,            -- direct_listing | secondary_listing
    term                VARCHAR,            -- life_of_rights | 10_year | 30_year | partial_10_year | fixed_return
    term_remaining      DOUBLE,
    term_expiration     TIMESTAMPTZ,
    seller_id           INTEGER,

    ltm                 DOUBLE,
    dollar_age          DOUBLE,
    deal_amount         DOUBLE,
    deal_date           TIMESTAMPTZ,
    close_date          TIMESTAMPTZ,
    published_date      TIMESTAMPTZ,
    list_price          DOUBLE,
    list_price_multiple DOUBLE,
    list_price_multiple_source VARCHAR,

    marketplace_median            DOUBLE,
    marketplace_median_multiplier DOUBLE,
    marketplace_median_source     VARCHAR,
    is_marketplace_median_disabled BOOLEAN,

    highest_open_offer_amount DOUBLE,
    accepting_final_offers_start_at TIMESTAMPTZ,
    accepting_final_offers_end_at   TIMESTAMPTZ,
    is_open_auction     BOOLEAN,
    is_nft              BOOLEAN,
    is_featured         BOOLEAN,
    currency            VARCHAR,
    tags                JSON,
    raw_extra           JSON
);

-- ---------------------------------------------------------------------
-- L1 :: listing detail
-- ---------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS obs_listing (
    obs_id              BIGINT PRIMARY KEY DEFAULT nextval('seq_obs_id'),
    fetch_id            BIGINT NOT NULL,
    parser_version      VARCHAR NOT NULL,
    observed_at         TIMESTAMPTZ NOT NULL,

    listing_id          INTEGER NOT NULL,
    asset_id            INTEGER,            -- stable across relistings and resales
    valuation_id        INTEGER,
    title               VARCHAR,
    state               VARCHAR,
    display_state_msg   VARCHAR,
    kind                VARCHAR,
    term                VARCHAR,
    term_expiration     TIMESTAMPTZ,
    term_return_amount  DOUBLE,
    seller_id           INTEGER,
    published_date      TIMESTAMPTZ,
    asset_sale_date     TIMESTAMPTZ,

    minimum_price       DOUBLE,             -- reserve
    default_buy_now_price DOUBLE,
    buy_now_price       DOUBLE,
    proxy_increment_amount DOUBLE,
    offers_received_count INTEGER,
    unique_bidder_count INTEGER,
    accepting_final_offers_start_at TIMESTAMPTZ,
    accepting_final_offers_end_at   TIMESTAMPTZ,
    accepting_final_offers_initial_offer BIGINT,

    deal_offer_id       BIGINT,
    deal_amount         DOUBLE,
    deal_buyer_id       INTEGER,
    deal_multiple       DOUBLE,
    deal_multiple_source VARCHAR,
    deal_state          VARCHAR,
    deal_created_at     TIMESTAMPTZ,

    marketplace_median            DOUBLE,
    marketplace_median_multiplier DOUBLE,

    ltm                 DOUBLE,
    lifetime_amount     DOUBLE,
    three_years_average DOUBLE,             -- sync-normalization denominator
    dollar_age          DOUBLE,
    first_earnings_date TIMESTAMPTZ,
    distribution_frequency INTEGER,
    statistics_last_updated TIMESTAMPTZ,
    track_count         INTEGER,
    is_track_list_hidden BOOLEAN,

    -- the narrative is machine-generated; prompt_version says which model
    vd_prompt_id        VARCHAR,
    vd_prompt_version   INTEGER,

    tags                JSON,
    royalty_payors      JSON,
    media_urls          JSON,
    raw_extra           JSON
);

CREATE TABLE IF NOT EXISTS obs_offer (
    obs_id          BIGINT NOT NULL,
    offer_id        BIGINT NOT NULL,
    listing_id      INTEGER NOT NULL,
    buyer_id        INTEGER,
    bidder_index    INTEGER,
    amount          DOUBLE,
    multiple        DOUBLE,
    multiple_source VARCHAR,
    kind            VARCHAR,
    state           VARCHAR,
    term            VARCHAR,
    created_at      TIMESTAMPTZ,
    expiration      TIMESTAMPTZ,
    countered_offer_id BIGINT,
    counteroffer_id BIGINT,
    incentive_pool_qualifying BOOLEAN,
    PRIMARY KEY (obs_id, offer_id)
);

CREATE TABLE IF NOT EXISTS obs_buy_now_history (
    obs_id          BIGINT NOT NULL,
    seq             INTEGER NOT NULL,
    created_at      TIMESTAMPTZ,
    buy_now_price   DOUBLE,      -- -1 encodes "no buy-now set"
    PRIMARY KEY (obs_id, seq)
);

CREATE TABLE IF NOT EXISTS obs_earnings_quarter (
    obs_id          BIGINT NOT NULL,
    period_start    DATE NOT NULL,
    domestic        DOUBLE,
    intl            DOUBLE,
    unreported      DOUBLE,
    total           DOUBLE,
    PRIMARY KEY (obs_id, period_start)
);

CREATE TABLE IF NOT EXISTS obs_earnings_year (
    obs_id          BIGINT NOT NULL,
    dimension       VARCHAR NOT NULL,   -- song | income_type | source | music_user | royalty_payor
    year_index      INTEGER NOT NULL,   -- 5 == most recent == LTM window
    member          VARCHAR NOT NULL,
    amount          DOUBLE,
    PRIMARY KEY (obs_id, dimension, year_index, member)
);

CREATE TABLE IF NOT EXISTS obs_ltm_breakdown (
    obs_id          BIGINT NOT NULL,
    dimension       VARCHAR NOT NULL,
    member          VARCHAR NOT NULL,
    amount          DOUBLE,
    track_count     INTEGER,
    PRIMARY KEY (obs_id, dimension, member)
);

CREATE TABLE IF NOT EXISTS obs_track (
    obs_id          BIGINT NOT NULL,
    track_id        VARCHAR NOT NULL,
    parent_track_id VARCHAR,        -- non-empty => alternate version of a parent work
    title           VARCHAR,
    PRIMARY KEY (obs_id, track_id)
);

-- ---------------------------------------------------------------------
-- L2
-- ---------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS dim_asset (
    asset_id            INTEGER PRIMARY KEY,
    canonical_title     VARCHAR,
    n_listings          INTEGER,
    n_sold_listings     INTEGER,
    first_listed        TIMESTAMPTZ,
    last_listed         TIMESTAMPTZ,
    has_repeat_sale     BOOLEAN
);

CREATE TABLE IF NOT EXISTS dim_listing (
    listing_id          INTEGER PRIMARY KEY,
    asset_id            INTEGER,
    title               VARCHAR,
    url                 VARCHAR,
    state               VARCHAR,
    kind                VARCHAR,
    term                VARCHAR,
    term_family         VARCHAR,    -- perpetual | fixed_term | fixed_return
    term_years          DOUBLE,
    is_partial_share    BOOLEAN,
    seller_id           INTEGER,
    published_date      TIMESTAMPTZ,
    deal_date           TIMESTAMPTZ,
    close_date          TIMESTAMPTZ,

    ltm                 DOUBLE,
    three_years_average DOUBLE,
    lifetime_amount     DOUBLE,
    dollar_age          DOUBLE,
    first_earnings_date TIMESTAMPTZ,
    catalog_age_years   DOUBLE,
    track_count         INTEGER,

    minimum_price       DOUBLE,
    reserve_multiple    DOUBLE,
    marketplace_median_multiplier DOUBLE,
    is_marketplace_median_disabled BOOLEAN,

    n_observations      INTEGER,
    first_observed_at   TIMESTAMPTZ,
    last_observed_at    TIMESTAMPTZ,
    has_detail          BOOLEAN,    -- FALSE => index-only, no offer ladder
    tags                JSON
);

CREATE TABLE IF NOT EXISTS fact_outcome (
    listing_id          INTEGER PRIMARY KEY,
    sold                BOOLEAN NOT NULL,
    outcome_reason      VARCHAR,
    clearing_price      DOUBLE,
    winning_buyer_id    INTEGER,
    n_offers            INTEGER,
    n_unique_bidders    INTEGER,

    second_price        DOUBLE,      -- highest losing bid
    winner_margin_over_second DOUBLE,
    opening_price       DOUBLE,
    n_bids_by_winner    INTEGER,

    buyer_fee_standard  DOUBLE,
    all_in_cost_standard DOUBLE,
    multiple_gross      DOUBLE,
    multiple_all_in     DOUBLE,
    multiple_normalized DOUBLE,      -- vs three_years_average
    multiple_vs_median  DOUBLE,      -- clearing multiple / published anchor
    normalized_ltm      DOUBLE,
    censoring_flag      VARCHAR
);

CREATE TABLE IF NOT EXISTS fact_earnings_panel (
    listing_id      INTEGER NOT NULL,
    asset_id        INTEGER,
    period_start    DATE NOT NULL,
    total           DOUBLE,
    domestic        DOUBLE,
    intl            DOUBLE,
    unreported      DOUBLE,
    quarters_since_first_earnings INTEGER,
    quarters_before_listing DOUBLE,
    PRIMARY KEY (listing_id, period_start)
);

CREATE TABLE IF NOT EXISTS fact_source_mix (
    listing_id      INTEGER NOT NULL,
    dimension       VARCHAR NOT NULL,
    member          VARCHAR NOT NULL,
    ltm_amount      DOUBLE,
    ltm_share       DOUBLE,
    PRIMARY KEY (listing_id, dimension, member)
);

CREATE TABLE IF NOT EXISTS fact_offer (
    listing_id      INTEGER NOT NULL,
    offer_id        BIGINT NOT NULL,
    buyer_id        INTEGER,
    bidder_index    INTEGER,
    amount          DOUBLE,
    multiple        DOUBLE,
    created_at      TIMESTAMPTZ,
    state           VARCHAR,
    is_winning      BOOLEAN,
    is_proxy_increment BOOLEAN,
    rank_in_ladder  INTEGER,
    PRIMARY KEY (listing_id, offer_id)
);

CREATE TABLE IF NOT EXISTS data_quality_flag (
    listing_id      INTEGER NOT NULL,
    flag            VARCHAR NOT NULL,
    severity        VARCHAR NOT NULL,
    detail          VARCHAR,
    flagged_at      TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (listing_id, flag)
);

-- ---------------------------------------------------------------------
-- Views
-- ---------------------------------------------------------------------

CREATE OR REPLACE VIEW v_pricing_frame AS
SELECT
    l.listing_id, l.asset_id, l.kind, l.term, l.term_family, l.term_years,
    l.is_partial_share, l.seller_id,
    l.ltm, l.three_years_average, l.dollar_age, l.catalog_age_years,
    l.track_count, l.reserve_multiple,
    l.marketplace_median_multiplier, l.is_marketplace_median_disabled,
    l.published_date, l.deal_date,
    date_trunc('quarter', l.deal_date) AS deal_quarter,
    l.ltm / nullif(l.three_years_average, 0) AS ltm_vs_3yr,
    o.sold, o.clearing_price, o.multiple_gross, o.multiple_all_in,
    o.multiple_normalized, o.multiple_vs_median,
    o.n_offers, o.n_unique_bidders, o.second_price,
    o.winner_margin_over_second, o.censoring_flag,
    (SELECT count(*) FROM data_quality_flag q
      WHERE q.listing_id = l.listing_id AND q.severity = 'exclude') AS n_exclude_flags
FROM dim_listing l
LEFT JOIN fact_outcome o USING (listing_id);

CREATE OR REPLACE VIEW v_field_coverage AS
SELECT date_trunc('year', deal_date) AS deal_year, kind,
       count(*) AS n,
       sum(CASE WHEN sold THEN 1 ELSE 0 END) AS n_sold,
       avg(CASE WHEN dollar_age IS NOT NULL THEN 1.0 ELSE 0 END) AS cov_dollar_age,
       avg(CASE WHEN three_years_average IS NOT NULL THEN 1.0 ELSE 0 END) AS cov_3yr,
       avg(CASE WHEN n_unique_bidders IS NOT NULL THEN 1.0 ELSE 0 END) AS cov_bidders,
       avg(CASE WHEN marketplace_median_multiplier IS NOT NULL THEN 1.0 ELSE 0 END) AS cov_median_anchor
FROM v_pricing_frame GROUP BY 1,2 ORDER BY 1,2;

-- Is dollar_age a weighted average in years or an unnormalized total?
-- A weighted average cannot exceed the age of the oldest earning track, so
-- dollar_age > catalog_age_years is diagnostic of a different construction.
CREATE OR REPLACE VIEW v_dollar_age_check AS
SELECT
    count(*) AS n,
    corr(dollar_age, ln(nullif(ltm,0)))              AS corr_with_log_ltm,
    corr(dollar_age, catalog_age_years)              AS corr_with_catalog_age,
    sum(CASE WHEN dollar_age > catalog_age_years + 0.51 THEN 1 ELSE 0 END) AS n_exceeds_catalog_age,
    median(dollar_age)                               AS median_dollar_age,
    max(dollar_age)                                  AS max_dollar_age
FROM dim_listing WHERE dollar_age IS NOT NULL AND ltm > 0;

-- Repeat sales matched by track overlap (asset_id is NOT shared across
-- listings -- the platform creates a fresh asset wrapper per listing even
-- for resales). Two listings are a repeat pair when their track sets share
-- >= min_overlap_pct of the smaller set and the later one is secondary_listing.
-- This is the only direct observation of realized holding-period returns.
CREATE OR REPLACE VIEW v_repeat_sales AS
WITH track_sets AS (
    SELECT t.obs_id, o.listing_id,
           list(t.track_id ORDER BY t.track_id) AS tracks,
           count(t.track_id) AS n_tracks
    FROM obs_track t
    JOIN obs_listing o USING (obs_id)
    -- parser stores '' as NULL, so coalesce before comparing: NULL = '' is
    -- never true and would silently filter out every root work.
    WHERE coalesce(t.track_id,'') != '' AND coalesce(t.parent_track_id,'') = ''
    GROUP BY t.obs_id, o.listing_id
), pairs AS (
    SELECT a.listing_id AS first_listing_id,
           b.listing_id AS later_listing_id,
           -- Jaccard-style: shared / min(|A|, |B|)
           list_aggregate(
               [x for x in a.tracks if list_contains(b.tracks, x)],
               'count'
           )::DOUBLE / least(a.n_tracks, b.n_tracks) AS overlap_pct
    FROM track_sets a
    JOIN track_sets b ON a.listing_id < b.listing_id
    WHERE least(a.n_tracks, b.n_tracks) >= 2
)
SELECT p.first_listing_id, p.later_listing_id,
       round(p.overlap_pct, 3)                          AS track_overlap,
       f.deal_date                                      AS first_deal_date,
       t.deal_date                                      AS later_deal_date,
       date_diff('day', f.deal_date, t.deal_date) / 365.25 AS years_held,
       f.clearing_price                                 AS first_price,
       t.clearing_price                                 AS later_price,
       t.clearing_price / nullif(f.clearing_price, 0)  AS price_ratio,
       f.ltm                                            AS first_ltm,
       t.ltm                                            AS later_ltm,
       t.ltm / nullif(f.ltm, 0)                        AS ltm_ratio,
       dl_t.kind                                        AS later_kind
FROM pairs p
JOIN v_pricing_frame f ON f.listing_id = p.first_listing_id AND f.sold
JOIN v_pricing_frame t ON t.listing_id = p.later_listing_id AND t.sold
                      AND t.deal_date > f.deal_date
JOIN dim_listing dl_t ON dl_t.listing_id = p.later_listing_id
WHERE p.overlap_pct >= 0.80;
